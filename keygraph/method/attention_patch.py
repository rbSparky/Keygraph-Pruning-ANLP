import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from keygraph.method.keygraph_cache import KeyGraphCache


def _expand_attn_mask_for_prefix(attn_mask: torch.Tensor, prefix_len: int):
    """
    Prepend zeros for a 'prefix_len' KV prefix so caller's mask aligns after we
    add KeyGraph tokens (reps/rescues/tail/current-step).
    """
    if attn_mask is None or prefix_len == 0:
        return attn_mask
    bsz, one, q_len, kv_len = attn_mask.shape
    zeros = torch.zeros(bsz, 1, q_len, prefix_len, device=attn_mask.device, dtype=attn_mask.dtype)
    return torch.cat([zeros, attn_mask], dim=-1)


def _make_rep_bias(cluster_sizes: torch.Tensor, q_len: int) -> torch.Tensor:
    """
    log|C| additive bias for the kept representative positions (broadcastable [B,1,Q,C_keep]).
    """
    if cluster_sizes.numel() == 0:
        return None
    log_mass = torch.log(cluster_sizes.clamp_min(1.0).to(torch.float32))  # build in fp32
    return log_mass.view(1, 1, 1, -1).expand(1, 1, q_len, -1)              # [1,1,Q,C]


def _expand_kv_view(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Zero-copy logical expansion from [B, H_kv, T, D] -> [B, H_kv*groups, T, D] for GQA.
    """
    if groups == 1:
        return x
    B, Hkv, T, D = x.shape
    return x.unsqueeze(2).expand(B, Hkv, groups, T, D).reshape(B, Hkv * groups, T, D)


def _sdpa_with_keygraph(
    query_states,        # [B, H_q, Q, D]
    key_states,          # [B, H_kv, S, D]  (current block)
    value_states,        # [B, H_kv, S, D]
    attention_mask,      # [B, 1, Q, S] or None (additive)
    keygraph_cache: KeyGraphCache,
    layer_idx: int,
    epsilon_var: float = 1e-3,
):
    """
    Assemble KV as: kept reps + rescued members + tail (+ current block),
    expand K/V to match query heads under GQA, add log-mass bias on kept reps,
    then run SDPA once.
    """
    B, Hq, Q, D = query_states.shape
    Hkv = key_states.shape[1]
    assert Hq % Hkv == 0, f"GQA head mismatch: Hq={Hq}, Hkv={Hkv}"
    groups = Hq // Hkv
    device = query_states.device
    qdtype = query_states.dtype

    rep = keygraph_cache.get_layer_representatives(layer_idx)

    # If layer is not patched by KeyGraph, just run SDPA over current block (expanded to Hq)
    if rep is None or "K_star" not in rep:
        K_cur = _expand_kv_view(key_states, groups)
        V_cur = _expand_kv_view(value_states, groups)
        attn_mask_total = attention_mask.to(qdtype) if attention_mask is not None else None
        return F.scaled_dot_product_attention(
            query_states, K_cur, V_cur, attn_mask=attn_mask_total, dropout_p=0.0, is_causal=False
        )

    # Representatives (stored per K/V head) -> expand to Hq
    K_star = rep["K_star"].to(device=device, dtype=qdtype)     # [H_kv, C, D]
    V_star = rep["V_star"].to(device=device, dtype=qdtype)     # [H_kv, C, D]
    cluster_sizes = rep["cluster_sizes"].to(device=device)      # [C] (float)

    members_padded = rep["members_padded"]  # [C, max_m] (long, -1 padded)
    sizes_long     = rep["sizes_long"]      # [C] (long)
    probe_idx      = rep["probe_idx"]       # [C, P] (long, -1 padded)

    C = K_star.shape[1]
    # [B,H_kv,C,D] -> repeat to [B,H_q,C,D] (zero-copy view)
    K_rep = _expand_kv_view(K_star.unsqueeze(0).expand(B, -1, -1, -1), groups)
    V_rep = _expand_kv_view(V_star.unsqueeze(0).expand(B, -1, -1, -1), groups)

    # ---- Variance probe (decide which clusters to expand) ----
    if probe_idx.numel() > 0:
        P = probe_idx.shape[1]
        probe_clamped = torch.clamp(probe_idx, min=0)  # [C,P]

        # Unique probe members, fetch once from CPU originals
        uniq = torch.unique(probe_clamped.view(-1))
        if uniq.numel() > 0:
            K_probe_all, _ = keygraph_cache.gather_original(
                layer_idx, uniq, device=device, dtype=qdtype
            )  # [H_kv, Muniq, D]
            # Map probe_clamped to uniq indices fast
            S = rep["original_k_cpu"].shape[1]
            idxmap = torch.full((S,), -1, device=device, dtype=torch.long)
            idxmap[uniq] = torch.arange(uniq.numel(), device=device, dtype=torch.long)
            map_idx = idxmap[probe_clamped]                      # [C,P] (=-1 pad)
            map_idx_safe = torch.clamp(map_idx, min=0)
            d_k = K_probe_all.shape[-1]
            K_probe = K_probe_all.index_select(1, map_idx_safe.view(-1)).view(
                Hkv, C, P, d_k
            )  # [H_kv,C,P,D]
        else:
            d_k = D
            K_probe = torch.empty(Hkv, C, P, d_k, device=device, dtype=qdtype)

        # Query per K/V head: average Q heads within each group to compare to K/V head
        if groups > 1:
            q = query_states.view(B, Hkv, groups, Q, D).mean(dim=2)  # [B,H_kv,Q,D]
        else:
            q = query_states  # [B,H_kv,Q,D]

        valid = (probe_idx >= 0).to(torch.float32).view(1, 1, 1, C, P)   # [1,1,1,C,P]
        scores = torch.einsum("bhqd,hcpd->bhqcp", q, K_probe) / (float(D) ** 0.5)
        sum_w  = valid.sum(dim=-1).clamp_min(1.0)                        # [1,1,1,C]
        mean   = (scores * valid).sum(dim=-1) / sum_w                    # [B,H_kv,Q,C]
        mean2  = (scores * scores * valid).sum(dim=-1) / sum_w           # [B,H_kv,Q,C]
        var    = (mean2 - mean * mean).relu()                             # [B,H_kv,Q,C]
        expand_mask_c = (var > epsilon_var).any(dim=(0, 1, 2))           # [C] bool
    else:
        expand_mask_c = torch.zeros(C, dtype=torch.bool, device=device)

    keep_mask_c = ~expand_mask_c
    keep_idx = keep_mask_c.nonzero(as_tuple=True)[0]   # [C_keep]
    exp_idx  = expand_mask_c.nonzero(as_tuple=True)[0] # [C_exp]

    # Kept reps
    if keep_idx.numel() > 0:
        K_keep = K_rep.index_select(2, keep_idx)            # [B,H_q,C_keep,D]
        V_keep = V_rep.index_select(2, keep_idx)
        sizes_keep = cluster_sizes.index_select(0, keep_idx) # [C_keep]
    else:
        K_keep = K_rep[:, :, :0, :]
        V_keep = V_rep[:, :, :0, :]
        sizes_keep = cluster_sizes[:0]

    # Rescued members (fetch from CPU originals, expand to H_q)
    if exp_idx.numel() > 0:
        mem_rows = members_padded.index_select(0, exp_idx)  # [C_exp, max_m]
        m_sizes  = sizes_long.index_select(0, exp_idx)      # [C_exp]
        max_m = mem_rows.shape[1]
        valid_m = (torch.arange(max_m, device=mem_rows.device).view(1, -1) < m_sizes.view(-1, 1))
        sel = mem_rows[valid_m]  # [M_total] flat indices
        if sel.numel() > 0:
            K_mem_HMD, V_mem_HMD = keygraph_cache.gather_original(
                layer_idx, sel, device=device, dtype=qdtype
            )  # [H_kv,M,D]
            K_mem = _expand_kv_view(K_mem_HMD.unsqueeze(0).expand(B, -1, -1, -1), groups)  # [B,H_q,M,D]
            V_mem = _expand_kv_view(V_mem_HMD.unsqueeze(0).expand(B, -1, -1, -1), groups)
        else:
            K_mem = K_rep[:, :, :0, :]
            V_mem = V_rep[:, :, :0, :]
    else:
        K_mem = K_rep[:, :, :0, :]
        V_mem = V_rep[:, :, :0, :]

    # Tail buffer (recent tokens), stored per K/V head -> expand to H_q
    K_tail, V_tail = keygraph_cache.get_tail(layer_idx)  # [B,H_kv,T,D] or None
    if K_tail is not None:
        K_tail = _expand_kv_view(K_tail, groups)
        V_tail = _expand_kv_view(V_tail, groups)

    # Current block K/V: expand to H_q
    K_cur = _expand_kv_view(key_states, groups)   # [B,H_q,S,D]
    V_cur = _expand_kv_view(value_states, groups)

    # Concatenate prefix + current
    catK = [K_keep, K_mem]
    catV = [V_keep, V_mem]
    if K_tail is not None:
        catK.append(K_tail)
        catV.append(V_tail)
    catK.append(K_cur)
    catV.append(V_cur)

    K_full = torch.cat(catK, dim=2)                   # [B,H_q,KV,D]
    V_full = torch.cat(catV, dim=2)
    prefix_len = sum(x.shape[2] for x in catK[:-1])   # exclude current block

    # Attention mask (add zeros for prefix) — ensure dtype matches query dtype
    attn_mask_total = None
    if attention_mask is not None:
        attn_mask_total = _expand_attn_mask_for_prefix(attention_mask, prefix_len).to(qdtype)

    # log|C| bias only on kept reps (first C_keep of prefix) — ensure dtype & device match
    if sizes_keep.numel() > 0:
        rep_bias = _make_rep_bias(sizes_keep, Q).to(device=K_full.device, dtype=qdtype)
        kv_total = K_full.shape[2]
        pad = kv_total - rep_bias.shape[-1]
        if pad > 0:
            rep_bias = torch.cat(
                [rep_bias, torch.zeros(rep_bias.shape[:-1] + (pad,), device=K_full.device, dtype=qdtype)],
                dim=-1,
            )
        attn_mask_total = rep_bias if attn_mask_total is None else (attn_mask_total + rep_bias)

    # Final SDPA
    return F.scaled_dot_product_attention(
        query_states, K_full, V_full, attn_mask=attn_mask_total, dropout_p=0.0, is_causal=False
    )


class RotatoryPositionalEncoding(nn.Module):
    """
    Lazy-initialized per attention module (embedding_dim=head_dim). max_len from model config.
    """
    def __init__(self, embedding_dim, max_len, base=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.base = base
        self._create_rotary_embedding()

    def reset_parameters(self):
        self._create_rotary_embedding()

    def _create_rotary_embedding(self):
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.embedding_dim, 2)[: (self.embedding_dim // 2)].float() / self.embedding_dim)
        )
        self.register_buffer("theta", theta)
        seq_len = torch.arange(self.max_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum("i,j->ij", seq_len, self.theta).float()
        cache = torch.stack([idx_theta.cos(), idx_theta.sin()], dim=-1)
        self.register_buffer("cache", cache)

    def forward(self, x):
        B, S, H, d_k = x.shape
        x = x.view(B, S, H, d_k // 2, 2)
        rope = self.cache[:S].unsqueeze(0).unsqueeze(2)  # [1,S,1,d/2,2]
        cos, sin = rope[..., 0], rope[..., 1]
        x1, x2 = x[..., 0], x[..., 1]
        x_rotated = torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return x_rotated.flatten(-2)


@contextmanager
def keygraph_attention_patch(model, keygraph_cache: KeyGraphCache, rescue_threshold: float = 0.5):
    """
    Patch LlamaAttention.forward to:
      - support GQA (expand KV & all KeyGraph KV to H_q via zero-copy views),
      - run one SDPA with reps/rescues/tail/current,
      - NEVER return present_key_value (stop HF KV growth),
      - bypass the first long prefill under the patch,
      - **CRITICAL FIX:** after prefill, attend with ONLY the last token each step.
      - return EXACTLY 2 outputs to match HF when use_cache=False.
    """
    original_forward_methods = {}
    max_rope_len = int(getattr(model.config, "max_position_embeddings", 32768))

    def create_patched_forward(layer_idx):
        def patched_forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()

            # -------- Robust GQA attributes --------
            hidden_size = getattr(self, "hidden_size", getattr(self, "embed_dim", getattr(self.o_proj, "in_features", model.config.hidden_size)))
            num_heads_q = int(getattr(self, "num_heads", getattr(self, "num_attention_heads", model.config.num_attention_heads)))
            num_heads_kv = int(getattr(self, "num_key_value_heads", getattr(self, "num_kv_heads", getattr(model.config, "num_key_value_heads", num_heads_q))))
            head_dim = int(getattr(self, "head_dim", hidden_size // max(1, num_heads_q)))
            assert hidden_size == num_heads_q * head_dim, "hidden_size != num_heads * head_dim"

            # -------- Projections --------
            q_proj = self.q_proj(hidden_states)
            k_proj = self.k_proj(hidden_states)
            v_proj = self.v_proj(hidden_states)

            # Shape to heads (GQA-aware)
            query_states = q_proj.view(bsz, q_len, num_heads_q, head_dim).transpose(1, 2)    # [B,H_q,Q,D]
            key_states   = k_proj.view(bsz, q_len, num_heads_kv, head_dim).transpose(1, 2)   # [B,H_kv,Q,D]
            value_states = v_proj.view(bsz, q_len, num_heads_kv, head_dim).transpose(1, 2)   # [B,H_kv,Q,D]

            # -------- Lazy RoPE (per module; matches head_dim) --------
            if not hasattr(self, "_kg_rope") or getattr(self._kg_rope, "embedding_dim", None) != head_dim or getattr(self._kg_rope, "max_len", None) != max_rope_len:
                self._kg_rope = RotatoryPositionalEncoding(head_dim, max_len=max_rope_len)
            rope_local = self._kg_rope.to(hidden_states.device, hidden_states.dtype)
            query_states = rope_local(query_states.transpose(1, 2)).transpose(1, 2)
            key_states   = rope_local(key_states.transpose(1, 2)).transpose(1, 2)

            # -------- Bypass first long prefill under the patch (return EXACTLY 2 outputs) --------
            if not hasattr(self, "_kg_prefill_seen") and q_len > 1:
                self._kg_prefill_seen = True
                Hq = num_heads_q; Hkv = num_heads_kv
                assert Hq % Hkv == 0
                groups = Hq // Hkv
                K_cur = _expand_kv_view(key_states, groups)
                V_cur = _expand_kv_view(value_states, groups)
                attn_mask_total = attention_mask.to(query_states.dtype) if attention_mask is not None else None
                attn = F.scaled_dot_product_attention(
                    query_states, K_cur, V_cur, attn_mask=attn_mask_total, dropout_p=0.0, is_causal=False
                )
                attn = attn.transpose(1, 2).contiguous().view(bsz, q_len, getattr(self.o_proj, "in_features", hidden_size))
                out  = self.o_proj(attn)
                return out, None  # 2-tuple

            # -------- CRITICAL FIX: decode steps -> use ONLY the last token --------
            if q_len > 1:
                query_states = query_states[:, :, -1:, :]
                key_states   = key_states[:,   :, -1:, :]
                value_states = value_states[:, :, -1:, :]
                if attention_mask is not None:
                    attention_mask = attention_mask[..., -1:, :]  # [B,1,1,KV]

            # -------- KeyGraph SDPA (GQA-aware) --------
            attn_output = _sdpa_with_keygraph(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                keygraph_cache=keygraph_cache,
                layer_idx=layer_idx,
                epsilon_var=getattr(self, "epsilon_var", 1e-3),
            )  # [B,H_q,1,D] typically

            # -------- Output projection --------
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, attn_output.shape[1], getattr(self.o_proj, "in_features", hidden_size))
            attn_output = self.o_proj(attn_output)

            # Append ONLY last token to tail (per K/V head)
            k_last = key_states[:, :, -1:, :]
            v_last = value_states[:, :, -1:, :]
            keygraph_cache.append_tail(layer_idx, k_last, v_last)

            # ALWAYS return EXACTLY 2 to match HF when use_cache=False path
            return attn_output, None

        return patched_forward

    num_layers = model.config.num_hidden_layers
    try:
        for i in range(num_layers):
            layer = model.model.layers[i]
            original_forward_methods[i] = layer.self_attn.forward
            layer.self_attn.forward = create_patched_forward(i).__get__(layer.self_attn, type(layer.self_attn))
        yield keygraph_cache
    finally:
        for layer_idx, original_method in original_forward_methods.items():
            layer = model.model.layers[layer_idx]
            layer.self_attn.forward = original_method
