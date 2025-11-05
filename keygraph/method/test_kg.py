import argparse, os, math, time, contextlib, gc
import torch, torch.nn.functional as F
from statistics import median
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- keygraph imports (local first, then package layout) ---
try:
    from kg_cache import KeygraphCache, KeygraphCacheConfig
except Exception:
    from keygraph.method.kg_cache import KeygraphCache, KeygraphCacheConfig

try:
    from kg_patch import KeygraphAttentionPatch, PatchConfig
except Exception:
    from keygraph.method.kg_patch import KeygraphAttentionPatch, PatchConfig

try:
    from keygraph_core import build_descriptors_unrope
except Exception:
    from keygraph.method.keygraph_core import build_descriptors_unrope


# ===================== utils / mem =====================
def mb(x): return x / (1024**2)
def gb(x): return x / (1024**3)

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def reset_peak_and_cache():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()

def mem_now():
    if not torch.cuda.is_available(): return 0, 0
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()

def mem_peak():
    if not torch.cuda.is_available(): return 0
    return torch.cuda.max_memory_allocated()

def _dtype_nbytes(dtype):
    if dtype in (torch.float16, torch.bfloat16): return 2
    if dtype == torch.float32: return 4
    if dtype == torch.float64: return 8
    if dtype == torch.int32: return 4
    if dtype == torch.int64: return 8
    return 4

def _mb(x_bytes):  # MiB
    return float(x_bytes) / (1024.0 * 1024.0)

@contextlib.contextmanager
def measure_block(tag, flush_after=False):
    reset_peak_and_cache()
    t0 = time.time()
    try:
        yield
    finally:
        cuda_sync()
        alloc, reserv = mem_now()
        peak = mem_peak()
        dt = time.time() - t0
        print(f"\n[{tag}] time={dt:.2f}s  now_alloc={mb(alloc):,.1f} MB  now_reserved={mb(reserv):,.1f} MB  peak_alloc={mb(peak):,.1f} MB")
        if flush_after:
            reset_peak_and_cache()
            alloc2, reserv2 = mem_now()
            print(f"[{tag}] after empty_cache(): alloc={mb(alloc2):,.1f} MB  reserved={mb(reserv2):,.1f} MB")


def _make_gqa_map(cfg, device):
    Hq  = int(getattr(cfg, "num_attention_heads", 0) or getattr(cfg, "n_head", 0))
    Hkv = int(getattr(cfg, "num_key_value_heads", Hq))
    group = max(1, Hq // max(1, Hkv))
    gqa_map = torch.div(torch.arange(Hq, device=device), group, rounding_mode="floor")  # [Hq], values in [0, Hkv-1]
    return gqa_map, Hq, Hkv


# ===================== dashboards =====================
def _cluster_size_stats(labels: torch.Tensor, _C_hint: int):
    if labels.device.type != "cpu":
        labels = labels.detach().to("cpu")
    uniq, cnt = torch.unique(labels.to(torch.long), return_counts=True)
    if cnt.numel() == 0:
        return (0.0, 0.0, 0.0, 0.0), cnt
    vals_sorted = sorted(cnt.tolist())
    n = len(vals_sorted)
    p95_idx = min(n - 1, int(0.95 * (n - 1)))
    return (
        float(vals_sorted[0]),
        float(median(vals_sorted)),
        float(vals_sorted[p95_idx]),
        float(vals_sorted[-1]),
    ), cnt

def _layer_reps_bytes(layer) -> dict:
    C, Hkv, D = layer.repsK.shape
    bytes_per_scalar_KV = _dtype_nbytes(layer.repsK.dtype)
    kv_bytes = C * Hkv * D * 2 * bytes_per_scalar_KV  # K + V
    _, rp_dim = layer.repsPhi.shape
    phi_bytes = C * rp_dim * _dtype_nbytes(layer.repsPhi.dtype)
    return {
        "C": C, "Hkv": Hkv, "D": D, "rp_dim": int(rp_dim),
        "kv_bytes": int(kv_bytes),
        "phi_bytes": int(phi_bytes),
        "total_bytes": int(kv_bytes + phi_bytes),
    }

def print_keygraph_dashboard(kg_cache, est_kv_mb: float | None = None, warn_if_big_clusters=True):
    layers = getattr(kg_cache, "layers", None)
    if layers is None:
        layers = list(kg_cache)

    print("\n================== KEYGRAPH COMPRESSION DASHBOARD ==================")
    total_tokens = 0
    total_clusters = 0
    total_kv_bytes = 0
    total_phi_bytes = 0

    for li, layer in enumerate(layers):
        S = int(layer.num_tokens()) if hasattr(layer, "num_tokens") else int(layer.labels.numel())
        C = int(layer.num_clusters()) if hasattr(layer, "num_clusters") else int(layer.repsK.shape[0])
        rep_info = _layer_reps_bytes(layer)
        (c_min, c_med, c_p95, c_max), _cnt = _cluster_size_stats(layer.labels, C)
        avg_sz = (S / max(1, C)) if C > 0 else 0.0

        total_tokens += S
        total_clusters += C
        total_kv_bytes += rep_info["kv_bytes"]
        total_phi_bytes += rep_info["phi_bytes"]

        print(f"[L{li:02d}] S={S:6d}  C={C:5d}  avg={avg_sz:7.1f}  "
              f"min/med/p95/max={c_min:.0f}/{c_med:.0f}/{c_p95:.0f}/{c_max:.0f}  "
              f"| K*+V*={_mb(rep_info['kv_bytes']):7.2f} MB  Φ={_mb(rep_info['phi_bytes']):6.2f} MB  "
              f"(rp_dim={rep_info['rp_dim']})")

        if warn_if_big_clusters and C > 0 and avg_sz > 1024:
            print(f"   ⚠️  Over-merged? avg {avg_sz:.0f} tokens/cluster. Consider raising τ / rp_dim / tokens_assign_tau.")

    grand_total_mb = _mb(total_kv_bytes + total_phi_bytes)
    print("---------------------------------------------------------------------")
    print(f"[Totals] S={total_tokens}  C={total_clusters}  "
          f"Reps: K*+V*={_mb(total_kv_bytes):.2f} MB  Φ={_mb(total_phi_bytes):.2f} MB  "
          f"All reps={grand_total_mb:.2f} MB")

    if est_kv_mb is not None and est_kv_mb > 0:
        ratio = est_kv_mb / max(1e-9, grand_total_mb)
        print(f"[Compression vs full KV] ~{est_kv_mb:.2f} MB  →  {grand_total_mb:.2f} MB  "
              f"(~{ratio:.1f}× smaller)")
    print("=====================================================================\n")


# ===================== exact attn (attention-only mode) =====================
def exact_attention(Q, K, V, mask=None):
    B, Hq, Tq, D = Q.shape
    _, Hkv, S, _ = K.shape
    scale = 1.0 / math.sqrt(D)
    out = torch.empty(B, Hq, Tq, D, device=Q.device, dtype=Q.dtype)
    for hq in range(Hq):
        kvh = hq % Hkv
        K_h = K[:, kvh]  # [B,S,D]
        V_h = V[:, kvh]
        logits = torch.matmul(Q[:, hq], K_h.transpose(-1, -2)) * scale  # [B,Tq,S]
        if mask is not None:
            m = mask.squeeze(1)
            logits = logits.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else logits + m
        attn = F.softmax(logits.to(torch.float32), dim=-1).to(Q.dtype)   # [B,Tq,S]
        out[:, hq] = torch.matmul(attn, V_h)                             # [B,Tq,D]
    return out

def kv_cache_size_bytes(num_layers, n_kv_heads, head_dim, seq_len, dtype=torch.float16, batch=1):
    bytes_per = torch.finfo(dtype).bits // 8
    return 2 * num_layers * batch * n_kv_heads * seq_len * head_dim * bytes_per


# ===================== attention-only test =====================
def run_attention_only(device, dtype, ann_backend, force_exact, B=1, Hq=16, Hkv=8, Tq=1, S=32768, D=128,
                       rp_dim=64, top_clusters=32, rope_base=10000.0,
                       args_tau=0.35, args_ann_k=32, args_no_mutual=False, args_assign_tau=0.35, probe_per_cluster=8):
    print(f"\n== ATTENTION-ONLY TEST ==")
    print(f"Shapes: B={B}, Hq={Hq}, Hkv={Hkv}, Tq={Tq}, S={S}, D={D}, rp_dim={rp_dim}, topC={top_clusters}, ann={ann_backend}")

    if ann_backend == "exact" and S > 8192 and not force_exact:
        print(f"[WARN] exact kNN with S={S} is O(S^2). Switching ann_backend='faiss'. Use --force-exact to override.")
        ann_backend = "faiss"

    torch.manual_seed(0)
    Q = torch.randn(B, Hq, Tq, D, device=device, dtype=dtype)
    K = torch.randn(B, Hkv, S, D, device=device, dtype=dtype)
    V = torch.randn(B, Hkv, S, D, device=device, dtype=dtype)

    with measure_block("ExactAttention (baseline)", flush_after=True):
        _ = exact_attention(Q, K, V)

    # Build KeyGraph cache from K/V (convert to [H,S,D] per layer)
    K_l = K[0].to(dtype)   # [Hkv,S,D]
    V_l = V[0].to(dtype)
    pos_idx = torch.arange(S, device=device, dtype=torch.long)

    kg_cfg = KeygraphCacheConfig(
        rp_dim=rp_dim, tau=args_tau, ann_k=args_ann_k, mutual=(not args_no_mutual),
        ann_backend=ann_backend, probe_per_cluster=probe_per_cluster, probe_on_cpu=True,
        tokens_assign_tau=args_assign_tau, assign_mode="centroid",
        keep_descriptors=False, store_full_kv=False,   # keep full-KV OFF
        dtype=dtype, device=device
    )
    kg = KeygraphCache(kg_cfg)

    with measure_block("KeyGraph build_layer (prefill → reps cache)", flush_after=True):
        layer = kg.build_layer(K=K_l, V=V_l, pos_idx=pos_idx, rope_base=float(rope_base),
                               rp_matrix=None, device=device, dtype=dtype)
    print_keygraph_dashboard([layer], est_kv_mb=None)

    patch_cfg = PatchConfig(
        top_clusters=top_clusters, mass_alpha=1.0,
        use_representatives_only=True, enable_rescue=False,
        compute_dtype=dtype, small_S_exact_fallback=64, gqa_map=None,
    )
    patch = KeygraphAttentionPatch(attn_module=None, layer_cache=layer, cfg=patch_cfg).to(device)

    with measure_block("KeyGraph RepsOnly Attention (ours)", flush_after=True):
        _ = patch(
            Q_bhtd=Q, K_bhsd=None, V_bhsd=None,
            position_ids=torch.zeros(B, Tq, device=device, dtype=torch.long),
            attn_mask=None
        )


# ===================== probe-rescue builder =====================
def _build_probes_and_attach(K_eff, V_eff, pos_idx, rope_base, layer, probe_per_cluster, device, dtype):
    """
    Build token descriptors (using the same RP) and attach per-cluster probe K/V onto the layer:
      layer.K_probe: [C, Hkv, m, D], layer.V_probe: [C, Hkv, m, D], layer.probe_idx: list[C] of LongTensor[m]
    Kept on CPU pinned by default (tiny), DMA'ed on demand by the patch.
    """
    if probe_per_cluster <= 0:
        return layer

    # 1) compute token descriptors with the SAME RP used for reps
    #    K_eff: [Hkv, S_eff, D], pos_idx: [S_eff]
    with torch.inference_mode():
        Phi_tokens, _ = build_descriptors_unrope(
            keys_per_head=K_eff,
            pos_idx=pos_idx.to(torch.long),
            rp_matrix=getattr(layer, "rp_matrix", None),     # reuse existing RP
            r=int(layer.repsPhi.shape[1]),                   # same r as reps
            base=float(getattr(layer, "rope_base", rope_base)))
    

    labels = layer.labels.to(torch.long)               # [S_eff]
    repsPhi = layer.repsPhi                            # [C, r]
    C = repsPhi.shape[0]
    Hkv, S_eff, D = K_eff.shape
    m = min(int(probe_per_cluster),  max(1, (labels.bincount(minlength=C).min().item() if C > 0 else 1)))

    # 2) for each cluster, select top-m tokens by cosine to centroid descriptor
    Kp_list, Vp_list, idx_list = [], [], []
    # precompute per-cluster token lists
    for c in range(C):
        tok_idx_c = (labels == c).nonzero(as_tuple=False).squeeze(-1)
        if tok_idx_c.numel() == 0:
            # empty cluster (rare) → skip placeholder
            idx_list.append(torch.empty(0, dtype=torch.long))
            Kp_list.append(torch.zeros(Hkv, 0, D, device=K_eff.device, dtype=dtype))
            Vp_list.append(torch.zeros(Hkv, 0, D, device=K_eff.device, dtype=dtype))
            continue
        # cosine ~ dot since Φ are unit-norm
        sims = (Phi_tokens.index_select(0, tok_idx_c) @ repsPhi[c].to(Phi_tokens.dtype))
        topk = min(m, tok_idx_c.numel())
        top_vals, top_pos = torch.topk(sims, k=topk, largest=True, sorted=False)
        pick = tok_idx_c.index_select(0, top_pos)    # [topk]
        idx_list.append(pick.cpu())
        Kp_list.append(K_eff.index_select(1, pick))  # [Hkv, topk, D]
        Vp_list.append(V_eff.index_select(1, pick))  # [Hkv, topk, D]

    # pad to fixed m per cluster for easier stacking
    Kp_fixed, Vp_fixed = [], []
    for Kc, Vc in zip(Kp_list, Vp_list):
        topk = Kc.shape[1]
        if topk < m:
            pad_k = torch.zeros(Kc.shape[0], m - topk, Kc.shape[2], device=Kc.device, dtype=dtype)
            pad_v = torch.zeros(Vc.shape[0], m - topk, Vc.shape[2], device=Vc.device, dtype=dtype)
            Kc = torch.cat([Kc, pad_k], dim=1)
            Vc = torch.cat([Vc, pad_v], dim=1)
        Kp_fixed.append(Kc)
        Vp_fixed.append(Vc)

    K_probe = torch.stack(Kp_fixed, dim=0)  # [C, Hkv, m, D]
    V_probe = torch.stack(Vp_fixed, dim=0)  # [C, Hkv, m, D]

    # keep off-VRAM by default; move on demand in patch
    K_probe = K_probe.to("cpu").pin_memory()
    V_probe = V_probe.to("cpu").pin_memory()

    layer.K_probe = K_probe
    layer.V_probe = V_probe
    layer.probe_idx = idx_list
    layer.rope_base = float(getattr(layer, "rope_base", rope_base))
    return layer


# ===================== build KG from past (+attach probes) =====================
def _build_keygraph_from_past(past, attn_mask_1xS, rope_base, device, dtype, kg_cfg, probe_per_cluster):
    """
    Build KeyGraph layers from HF past_key_values. We must retain full per-layer K/V so that
    rescue can expand a cluster to ALL its members without keeping full KV on VRAM.
    Strategy: store_full_kv=True for prefill build, then move K/V to pinned CPU.
    """
    # Ensure we store full K/V for this build (prefill only)
    kg_cfg.store_full_kv = True
    kg = KeygraphCache(kg_cfg)

    layers = []
    valid_idx = attn_mask_1xS.nonzero(as_tuple=False).squeeze(-1)  # [S_eff]
    for (k, v) in past:
        # k,v: [B, Hkv, S, D] (B==1)
        K_full, V_full = k[0].contiguous().to(device=device, dtype=dtype), v[0].contiguous().to(device=device, dtype=dtype)
        K_eff = K_full.index_select(1, valid_idx)   # [Hkv,S_eff,D]
        V_eff = V_full.index_select(1, valid_idx)
        pos_idx = valid_idx.to(torch.long)         # [S_eff]

        layer_cache = kg.build_layer(
            K=K_eff, V=V_eff, pos_idx=pos_idx, rope_base=float(rope_base),
            rp_matrix=None, device=device, dtype=dtype
        )

        # Move full K/V OFF VRAM → pinned CPU (tiny compared to full HF KV, keeps decode VRAM low)
        if getattr(layer_cache, "K", None) is not None:
            layer_cache.K = layer_cache.K.to("cpu").pin_memory()
        if getattr(layer_cache, "V", None) is not None:
            layer_cache.V = layer_cache.V.to("cpu").pin_memory()

        # Attach small per-cluster probe K/V (already CPU pinned inside helper)
        layer_cache = _build_probes_and_attach(K_eff, V_eff, pos_idx, rope_base,
                                               layer_cache, probe_per_cluster, device, dtype)
        layers.append(layer_cache)
    return layers



# ===================== patching LLaMA attention =====================
def _make_llama_keygraph_forward(attn_mod, layer_cache, patch_cfg):
    # tiny helpers kept local
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]; x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def _rope_apply(x_bhtd: torch.Tensor, position_ids: torch.Tensor, rope_base: float) -> torch.Tensor:
        # x_bhtd: [B,H,T,D] (queries or keys), apply RoPE
        B, H, T, D = x_bhtd.shape
        assert D % 2 == 0
        half = D // 2
        idx = torch.arange(0, half, device=x_bhtd.device, dtype=torch.float32)
        inv = (1.0 / (rope_base ** (2.0 * idx / float(D)))).to(torch.float32)
        pos = position_ids.to(device=x_bhtd.device, dtype=torch.float32)  # [B,T]
        freqs = torch.einsum('bt,d->btd', pos, inv)  # [B,T,half]
        cos = torch.cat([freqs.cos(), freqs.cos()], -1)[:, None, :, :].to(x_bhtd.dtype)
        sin = torch.cat([freqs.sin(), freqs.sin()], -1)[:, None, :, :].to(x_bhtd.dtype)
        return (x_bhtd * cos) + (_rotate_half(x_bhtd) * sin)

    def _ret2(res):
        if isinstance(res, tuple):
            return (res + (None,))[:2]
        return res, None

    patch = KeygraphAttentionPatch(attn_mod, layer_cache, patch_cfg).to(layer_cache.repsK.device)
    rope_base = float(getattr(layer_cache, "rope_base", 10000.0))

    def kg_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs
    ):
        B, T, _ = hidden_states.shape
        # if prefill sneaks in, punt to HF
        if T != 1:
            return _ret2(attn_mod._kg_orig_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs
            ))

        # project Q/K/V for the current step
        q = attn_mod.q_proj(hidden_states)  # [B,1,Hq*Dh]
        Dh = getattr(attn_mod, "head_dim", None)
        if Dh is None:
            Dh = q.shape[-1] // max(1, getattr(attn_mod, "num_attention_heads", 1))
        Hq = q.shape[-1] // Dh
        Q = q.view(B, T, Hq, Dh).permute(0, 2, 1, 3).contiguous()          # [B,Hq,1,Dh]
        if position_ids is None:
            position_ids = torch.zeros(B, T, dtype=torch.long, device=hidden_states.device)
        Q_rope = _rope_apply(Q, position_ids, rope_base=rope_base)          # RoPE(Q)

        # base K/V from KeyGraph prefill cache (S_eff)
        K_bhsd = getattr(layer_cache, "K", None)
        V_bhsd = getattr(layer_cache, "V", None)
        if K_bhsd is not None and V_bhsd is not None:
            K_bhsd = K_bhsd.unsqueeze(0)  # [1,Hkv,S_eff,D]
            V_bhsd = V_bhsd.unsqueeze(0)  # [1,Hkv,S_eff,D]

        # ---- NEW: compute current-step self K/V ----
        k_cur = attn_mod.k_proj(hidden_states)  # [B,1,Hkv*Dh]
        v_cur = attn_mod.v_proj(hidden_states)  # [B,1,Hkv*Dh]
        Hkv = k_cur.shape[-1] // Dh
        K_cur = k_cur.view(B, T, Hkv, Dh).permute(0, 2, 1, 3).contiguous()  # [B,Hkv,1,D]
        V_cur = v_cur.view(B, T, Hkv, Dh).permute(0, 2, 1, 3).contiguous()  # [B,Hkv,1,D]
        K_cur = _rope_apply(K_cur, position_ids, rope_base=rope_base)       # RoPE(K)

        # ---- REMOVED THE K_aug/V_aug LOGIC ----
        
        # Call patch with separate PAST and CURRENT K/V
        out_bhtd = patch(
            Q_bhtd=Q_rope,
            K_bhsd=K_bhsd,      # Past K (full, for rescue)
            V_bhsd=V_bhsd,      # Past V (full, for rescue)
            K_cur_bhsd=K_cur,   # Current K
            V_cur_bhsd=V_cur,   # Current V
            position_ids=position_ids,
            attn_mask=None,
        )

        out = out_bhtd.permute(0, 2, 1, 3).contiguous().view(B, T, Hq * Dh)
        out = attn_mod.o_proj(out)
        return out, None

    return kg_forward



def _patch_model_with_keygraph(model, layers, patch_cfg):
    blocks = getattr(model, "model", model).layers
    assert len(blocks) == len(layers), "Layer count mismatch between model and KeyGraph caches."
    for i, blk in enumerate(blocks):
        attn = getattr(blk, "self_attn", None)
        if attn is None:
            raise RuntimeError("Expected LLaMA-style blocks with .self_attn")
        if not hasattr(attn, "_kg_orig_forward"):
            attn._kg_orig_forward = attn.forward  # backup original
        attn.forward = _make_llama_keygraph_forward(attn, layers[i], patch_cfg)
    return model


# ===================== generation helpers =====================
@torch.inference_mode()
def _baseline_generate(model, tok, prompt_ids, max_new_tokens, device):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    t0 = time.perf_counter()

    out = model(input_ids=prompt_ids, use_cache=True)
    past = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1,1]
    gen = []

    for _ in range(max_new_tokens):
        out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok_id = next_token.item()
        gen.append(tok_id)
        if tok_id == tok.eos_token_id:
            break

    dt = time.perf_counter() - t0
    text = tok.decode(gen, skip_special_tokens=True)
    toks = max(1, len(gen))
    metrics = dict(tokens_generated=len(gen),
                   tokens_per_second=(len(gen)/dt) if dt > 0 else 0.0,
                   latency_per_token=(dt/toks),
                   peak_vram_mb=(torch.cuda.max_memory_allocated(device)/(1024*1024)) if torch.cuda.is_available() else 0.0)
    return text, metrics, past, out

@torch.inference_mode()
def _keygraph_generate_from_past(model, tok, prompt_ids, past, attn_mask, rope_base, dtype, device,
                                 rp_dim, tau, ann_k, mutual, ann_backend, probe_per_cluster,
                                 tokens_assign_tau, top_clusters, mass_alpha, enable_rescue,
                                 rescue_var_eps, rescue_tokens_per_cluster, max_new_tokens):
    # Build KG caches (+ attach per-cluster probes)
    kg_cfg = KeygraphCacheConfig(
        rp_dim=rp_dim, tau=tau, ann_k=ann_k, mutual=mutual,
        ann_backend=ann_backend, probe_per_cluster=probe_per_cluster, probe_on_cpu=True,
        tokens_assign_tau=tokens_assign_tau, assign_mode="centroid",
        keep_descriptors=False, store_full_kv=False,   # <- keep full-KV OFF
        dtype=dtype, device=device
    )
    with measure_block("Build KeyGraph caches from HF past_key_values", flush_after=True):
        layers = _build_keygraph_from_past(past, attn_mask[0].to(torch.bool), rope_base, device, dtype, kg_cfg, probe_per_cluster)
    print_keygraph_dashboard(layers)

    # Patch attention (rescue enabled per cfg)
    gqa_map, Hq, Hkv = _make_gqa_map(model.config, device)
    patch_cfg = PatchConfig(
        top_clusters=top_clusters, mass_alpha=mass_alpha,
        use_representatives_only=(not enable_rescue), enable_rescue=enable_rescue,
        rescue_var_eps=rescue_var_eps, rescue_tokens_per_cluster=rescue_tokens_per_cluster,
        small_S_exact_fallback=64,
        gqa_map=gqa_map,                  # <- pass it in
        compute_dtype=dtype, attn_dropout_p=0.0
    )
    print(gqa_map.tolist()[:16])
    _patch_model_with_keygraph(model, layers, patch_cfg)

    # Free baseline KV for fair VRAM view
    del past
    cuda_sync(); reset_peak_and_cache()

    # Decode with KG (no HF cache; position_ids manually tracked)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    t0 = time.perf_counter()

    cur_pos = prompt_ids.shape[1]
    # seed next token from last prompt token
    out0 = model(input_ids=prompt_ids[:, -1:], attention_mask=None,
                 position_ids=torch.tensor([[cur_pos-1]], device=device, dtype=torch.long), use_cache=False)
    next_token = out0.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    gen = []

    for _ in range(max_new_tokens):
        out = model(input_ids=next_token, attention_mask=None,
                    position_ids=torch.tensor([[cur_pos]], device=device, dtype=torch.long), use_cache=False)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok_id = next_token.item()
        gen.append(tok_id)
        cur_pos += 1
        if tok_id == tok.eos_token_id:
            break

    dt = time.perf_counter() - t0
    text = tok.decode(gen, skip_special_tokens=True)
    toks = max(1, len(gen))
    metrics = dict(tokens_generated=len(gen),
                   tokens_per_second=(len(gen)/dt) if dt > 0 else 0.0,
                   latency_per_token=(dt/toks),
                   peak_vram_mb=(torch.cuda.max_memory_allocated(device)/(1024*1024)) if torch.cuda.is_available() else 0.0)
    return text, metrics, layers


# ===================== end2end driver =====================
def _prepare_prompt_text(args, tok):
    if args.text is not None:
        return args.text
    if args.text_file is not None:
        with open(args.text_file, "r", encoding="utf-8") as f:
            return f.read()
    # fallback: synthetic long-ish text
    prefix = "Once upon a time"
    chunks = [f"{prefix} {i}." for i in range(10 * args.seq_len // 8)]
    return " ".join(chunks)

def run_end2end_with_generation(device, dtype, model_id, ann_backend, force_exact,
                                rp_dim, top_clusters, args_tau, args_ann_k, args_no_mutual,
                                args_assign_tau, probe_per_cluster, seq_len, max_new_tokens,
                                mass_alpha=1.0, enable_rescue=False, rescue_var_eps=0.08, rescue_tokens_per_cluster=16,
                                prompt_text=None, show_outputs=True):
    print(f"\n== END-TO-END (your prompt) + GENERATION ==")
    print(f"Model: {model_id} | rp_dim={rp_dim} | topC={top_clusters} | ann={ann_backend}")

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="auto")
    model.eval()

    cfg = model.config
    rope_base = getattr(cfg, "rope_theta", 10000.0)
    n_layers = int(getattr(cfg, "num_hidden_layers", 0) or getattr(cfg, "n_layer", 0) or 0)
    n_heads = int(getattr(cfg, "num_attention_heads", 0) or getattr(cfg, "n_head", 0) or 0)
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads))
    head_dim = int(getattr(cfg, "hidden_size", 0) // max(1, n_heads))

    # Build prompt
    prompt_text = prompt_text if prompt_text is not None else "Hello!"
    enc = tok(prompt_text, return_tensors="pt", truncation=True, max_length=seq_len)
    input_ids = enc.input_ids.to(device)  # [1,S]
    attn_mask = enc.attention_mask.to(device) if "attention_mask" in enc else torch.ones_like(input_ids, device=device)
    S = input_ids.shape[1]

    est_kv = kv_cache_size_bytes(n_layers, n_kv_heads, head_dim, S, dtype=dtype, batch=1)
    print(f"[Est] KV cache for this prompt ≈ {mb(est_kv):,.1f} MB (fp{'16' if dtype==torch.float16 else '32'}, "
          f"L={n_layers}, kv_heads={n_kv_heads}, d_head={head_dim}, S={S})")

    # -------- Baseline full-KV generate --------
    with measure_block("Baseline generate (HF full KV)", flush_after=True):
        base_text, base_metrics, past, out_prefill = _baseline_generate(
            model, tok, input_ids, max_new_tokens=max_new_tokens, device=device
        )
    if show_outputs:
        print("\n[Baseline output]")
        print(base_text)

    # -------- KeyGraph (reps + probe-rescue) generate --------
    if ann_backend == "exact" and S > 8192 and not force_exact:
        print(f"[WARN] S={S} too large for exact build; switching to ann='faiss'. Use --force-exact to override.")
        ann_backend = "faiss"

    with measure_block("KeyGraph generate (reps + probe-rescue)", flush_after=True):
        kg_text, kg_metrics, layers = _keygraph_generate_from_past(
            model, tok, input_ids, past, attn_mask, rope_base, dtype, device,
            rp_dim=rp_dim, tau=args_tau, ann_k=args_ann_k, mutual=(not args_no_mutual),
            ann_backend=ann_backend, probe_per_cluster=probe_per_cluster,
            tokens_assign_tau=args_assign_tau if args_assign_tau is not None else args_tau,
            top_clusters=top_clusters, mass_alpha=mass_alpha,
            enable_rescue=enable_rescue, rescue_var_eps=rescue_var_eps,
            rescue_tokens_per_cluster=rescue_tokens_per_cluster, max_new_tokens=max_new_tokens
        )

    if show_outputs:
        print("\n[KeyGraph output]")
        print(kg_text)

    # -------- Summary --------
    print("\n=== Generation metrics ===")
    print(f"Baseline:  tok/s={base_metrics['tokens_per_second']:.2f}  "
          f"lat/tok={base_metrics['latency_per_token']:.4f}s  "
          f"tokens={base_metrics['tokens_generated']}  "
          f"peakVRAM={base_metrics['peak_vram_mb']:.1f} MB")
    print(f"KeyGraph:  tok/s={kg_metrics['tokens_per_second']:.2f}  "
          f"lat/tok={kg_metrics['latency_per_token']:.4f}s  "
          f"tokens={kg_metrics['tokens_generated']}  "
          f"peakVRAM={kg_metrics['peak_vram_mb']:.1f} MB")


# ===================== main =====================
def main():
    parser = argparse.ArgumentParser(description="VRAM + generation test for KeyGraph (reps + probe-rescue)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--mode", type=str, default="end2end", choices=["attention", "end2end", "both"])

    # end2end / prompt + gen
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--seq-len", type=int, default=8192, help="truncate prompt to this length")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--text", type=str, default=None, help="prompt text for end2end generation")
    parser.add_argument("--text-file", type=str, default=None, help="path to UTF-8 file with prompt text")
    parser.add_argument("--show-outputs", action="store_true", help="print generated text for both cases")

    # ANN backend
    parser.add_argument("--ann", type=str, default="faiss", choices=["faiss", "torch_ivf", "exact"])
    parser.add_argument("--force-exact", action="store_true", help="allow exact build with large S")

    # attention-only knobs
    parser.add_argument("--S", type=int, default=32768)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--Hq", type=int, default=16)
    parser.add_argument("--Hkv", type=int, default=8)
    parser.add_argument("--Tq", type=int, default=1)
    parser.add_argument("--rp-dim", type=int, default=64)
    parser.add_argument("--topC", type=int, default=32)

    # unified KeyGraph knobs
    parser.add_argument("--tau", type=float, default=0.35, help="cosine edge threshold for graph (0.25-0.55 good)")
    parser.add_argument("--ann-k", type=int, default=32, help="k for kNN search")
    parser.add_argument("--tokens-assign-tau", type=float, default=None, help="decode-time assign threshold; defaults to --tau if omitted")
    parser.add_argument("--probe-per-cluster", type=int, default=8, help="#probe tokens to store per cluster")
    parser.add_argument("--no-mutual", action="store_true", help="disable mutual-k filtering for kNN graph")

    # rescue/mass knobs
    parser.add_argument("--mass-alpha", type=float, default=1.0)
    parser.add_argument("--enable-rescue", action="store_true")
    parser.add_argument("--rescue-var-eps", type=float, default=0.08)
    parser.add_argument("--rescue-tokens-per-cluster", type=int, default=16)

    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available; running on CPU won’t show VRAM numbers.")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    torch.backends.cuda.matmul.allow_tf32 = True

    assign_tau = args.tokens_assign_tau if args.tokens_assign_tau is not None else args.tau

    if args.mode in ("attention", "both"):
        run_attention_only(
            device=device, dtype=dtype, ann_backend=args.ann, force_exact=args.force_exact,
            B=1, Hq=args.Hq, Hkv=args.Hkv, Tq=args.Tq,
            S=args.S, D=args.D, rp_dim=args.rp_dim, top_clusters=args.topC, rope_base=10000.0,
            args_tau=args.tau, args_ann_k=args.ann_k, args_no_mutual=args.no_mutual,
            args_assign_tau=assign_tau, probe_per_cluster=args.probe_per_cluster
        )

    if args.mode in ("end2end", "both"):
        prompt_text = None
        if args.text is not None:
            prompt_text = args.text
        elif args.text_file is not None:
            with open(args.text_file, "r", encoding="utf-8") as f:
                prompt_text = f.read()

        run_end2end_with_generation(
            device=device, dtype=dtype, model_id=args.model, ann_backend=args.ann, force_exact=args.force_exact,
            rp_dim=args.rp_dim, top_clusters=args.topC, args_tau=args.tau, args_ann_k=args.ann_k, args_no_mutual=args.no_mutual,
            args_assign_tau=assign_tau, probe_per_cluster=args.probe_per_cluster,
            seq_len=args.seq_len, max_new_tokens=args.max_new_tokens,
            mass_alpha=args.mass_alpha, enable_rescue=args.enable_rescue,
            rescue_var_eps=args.rescue_var_eps, rescue_tokens_per_cluster=args.rescue_tokens_per_cluster,
            prompt_text=prompt_text, show_outputs=args.show_outputs
        )

if __name__ == "__main__":
    main()
