# kg_patch.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Local import
    from kg_cache import LayerKeygraphCache
except Exception:
    # Fallback import path (if used as a package)
    from keygraph.method.kg_cache import LayerKeygraphCache


__all__ = [
    "PatchConfig",
    "KeygraphAttentionPatch",
    "integrate_with_llama_attention",
]


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class PatchConfig:
    # Screening over clusters
    top_clusters: int = 16          # shortlist size for Stage-A
    mass_alpha: float = 1.0         # weight for log|C| mass compensation (add to logits)

    # Representatives-only path and rescue
    use_representatives_only: bool = True  # default path for memory reduction
    enable_rescue: bool = True             # enable variance-probe rescue
    rescue_var_eps: float = 0.08           # variance threshold in probe scores to trigger rescue
    rescue_tokens_per_cluster: int = 16    # #member tokens to expand for a rescued cluster (uses probes first)

    # Fallback
    small_S_exact_fallback: int = 64       # if S <= this and K/V available, run exact attention

    # GQA mapping: optional explicit qh -> kvh map
    gqa_map: Optional[List[int]] = None

    # Numerics
    compute_dtype: torch.dtype = torch.float16
    attn_dropout_p: float = 0.0            # not used at decode, here for completeness
    scale_override: Optional[float] = None # if set, use this instead of 1/sqrt(D)

    # Masks: Stage-A is decode-time friendly (Tq=1); if you train with Tq>1,
    # reps attention ignores attn_mask. Exact and token rescue apply masks.


# -----------------------------------------------------------------------------
# RoPE utilities (inverse-rotation to get un-RoPE'd queries)
# -----------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # [.., D] with D even: (-x_2, x_1)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def _rope_cos_sin(head_dim: int, position_ids: torch.Tensor, base: float, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return cos/sin with shape [B, 1, T, D] to broadcast over heads.
    Matches standard LLaMA/HF convention: inv_freq = base^{-(2i/D)} with i stepping over even dims.
    """
    assert head_dim % 2 == 0, "RoPE head_dim must be even"
    half = head_dim // 2

    # frequencies
    idx = torch.arange(0, half, device=device, dtype=torch.float32)              # [half]
    inv_freq = base ** (-2.0 * idx / float(head_dim))                            # [half]
    pos = position_ids.to(device=device, dtype=torch.float32)                    # [B,T]
    freqs = pos[..., None] * inv_freq                                           # [B,T,half]

    # build [B,1,T,D] by interleaving even/odd
    cos = torch.empty(pos.shape[0], 1, pos.shape[1], head_dim, device=device, dtype=dtype)
    sin = torch.empty_like(cos)
    cos[..., ::2] = torch.cos(freqs)
    cos[..., 1::2] = torch.cos(freqs)
    sin[..., ::2] = torch.sin(freqs)
    sin[..., 1::2] = torch.sin(freqs)

    return cos, sin

def _unrope_queries(Q_bhtd: torch.Tensor, position_ids: torch.Tensor, rope_base: float) -> torch.Tensor:
    """
    Invert the RoPE rotation for queries.
    Q_bhtd: [B, Hq, T, D] (already RoPE-applied).
    position_ids: [B, T].
    Return: un-RoPE'd queries, [B, Hq, T, D].
    """
    cos, sin = _rope_cos_sin(Q_bhtd.shape[-1], position_ids, rope_base, Q_bhtd.device, Q_bhtd.dtype)
    return (Q_bhtd * cos) - (_rotate_half(Q_bhtd) * sin)

def _unrope_queries_tokenwise(Q_bhtd: torch.Tensor,
                              position_ids: torch.Tensor,
                              rope_base: float) -> torch.Tensor:
    """
    Inverse RoPE for queries, per (batch, time) token.
    Q_bhtd:      [B, H, T, D]   (usual attention input)
    position_ids:[B, T]         absolute positions for each token
    rope_base:   float          RoPE base (e.g., 1e6 for LLaMA)
    Returns:     [B, H, T, D] (float32)
    """
    B, H, T, D = Q_bhtd.shape
    assert D % 2 == 0
    device = Q_bhtd.device
    dtype = torch.float32

    x  = Q_bhtd.to(dtype)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    half = D // 2
    idx  = torch.arange(0, half, device=device, dtype=torch.float32)
    invf = rope_base ** (-2.0 * idx / float(D))         # [half]
    pos  = position_ids.to(device=device,
                           dtype=torch.float32).unsqueeze(-1)  # [B,T,1]
    ang  = pos * invf                                   # [B,T,half]
    cos  = ang.cos().unsqueeze(1)                       # [B,1,T,half]
    sin  = ang.sin().unsqueeze(1)

    # inverse rotation (same formula we used in the design notes)
    u1 = x1 * cos + x2 * sin
    u2 = x2 * cos - x1 * sin

    out = torch.empty((B, H, T, D), device=device, dtype=dtype)
    out[..., 0::2] = u1
    out[..., 1::2] = u2
    return out


def _unrope_keys_tokenwise(K_bhsd: torch.Tensor,
                           position_ids: torch.Tensor,
                           rope_base: float) -> torch.Tensor:
    """
    Inverse RoPE for keys, per (batch, time) token.
    K_bhsd:      [B, H, T, D]   (can be T=1 for current step)
    position_ids:[B, T]         absolute positions for those tokens
    rope_base:   float          RoPE base
    Returns:     [B, H, T, D] (float32)
    """
    # note: identical math as queries’ inverse — LLaMA uses symmetric rotation
    return _unrope_queries_tokenwise(K_bhsd, position_ids, rope_base)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _kv_head_for(qh: int, Hkv: int, gqa_map: Optional[List[int]]) -> int:
    if gqa_map is not None:
        kvh = gqa_map[qh]
        if not (0 <= kvh < Hkv):
            raise ValueError(f"gqa_map[{qh}]={kvh} out of range for Hkv={Hkv}")
        return kvh
    return qh % Hkv  # default LLaMA/GQA grouping

def _maybe_scale_logits(logits: torch.Tensor, scale: float) -> torch.Tensor:
    return logits * scale

def _apply_mask_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return logits
    # mask assumed to be additive (0 for allowed, -inf for disallowed) or boolean
    if mask.dtype == torch.bool:
        return logits.masked_fill(~mask, float("-inf"))
    return logits + mask


# -----------------------------------------------------------------------------
# Main patch
# -----------------------------------------------------------------------------

class KeygraphAttentionPatch(nn.Module):
    """
    Representatives-first attention with conditional token rescue.
    - Stage-A: Build query descriptor φ(q) by UnRoPE → per-head ℓ2 → mean across heads → RP → ℓ2.
               Score vs repsPhi (+ mass_alpha*log|C|), shortlist top-C clusters per token.
    - Stage-B: If rescue triggered (probe variance > eps) and layer.K/V available:
               mix token-level attention for rescued clusters with representative attention
               for the rest. Otherwise, use representatives-only.
    """
    def __init__(self, attn_module: nn.Module, layer_cache: LayerKeygraphCache, cfg: PatchConfig):
        super().__init__()
        self.attn = attn_module
        self.cache = layer_cache
        self.cfg = cfg

    def forward(
        self,
        Q_bhtd: torch.Tensor,
        K_bhsd: Optional[torch.Tensor] = None,
        V_bhsd: Optional[torch.Tensor] = None,
        K_cur_bhsd: Optional[torch.Tensor] = None,
        V_cur_bhsd: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,      # <-- make sure this is in the signature
        top_idx: Optional[torch.Tensor] = None,
        rescue_mask_bt: Optional[torch.Tensor] = None,
        rescue_clusters_bt: Optional[torch.Tensor] = None,
    ):
        """
        Minimal wrapper showing how to ensure position_ids exists and
        how to route to mixed or reps-only path.
        """
        layer = self.cache
        cfg   = self.cfg
        B, Hq, Tq, D = Q_bhtd.shape
        device = Q_bhtd.device

        # Ensure we have absolute positions for the Tq queries.
        if position_ids is None:
            # Fallback: derive from layer.pos_idx and the step offset.
            # Past length = number of stored source tokens S (if full KV present) else last pos in layer.pos_idx + 1
            if (K_bhsd is not None) and (K_bhsd.shape[2] > 0):
                past_len = K_bhsd.shape[2]
            else:
                # if store_full_kv=False, layer.pos_idx holds source positions
                past_len = int(layer.pos_idx.max().item()) + 1 if hasattr(layer, "pos_idx") and layer.pos_idx.numel() > 0 else 0
            position_ids = (past_len + torch.arange(Tq, device=device)).view(1, Tq).expand(B, -1)  # [B,Tq]

        # Build shortlist on-the-fly if not provided (exact mode / fallback)
        top_idx = self._ensure_top_idx(Q_bhtd=Q_bhtd, position_ids=position_ids, top_idx=top_idx)


        # Decide path
        use_full_kv = (K_bhsd is not None) and (V_bhsd is not None)
        do_rescue   = (rescue_mask_bt is not None) and (rescue_clusters_bt is not None)

        if do_rescue:
            return self._attend_mixed_with_rescue(
                Q_bhtd, K_bhsd, V_bhsd, K_cur_bhsd, V_cur_bhsd,
                attn_mask, top_idx, rescue_mask_bt, rescue_clusters_bt, position_ids
            )

        # else reps-only path, still done in un-RoPE space for consistency
        q_un = _unrope_queries_tokenwise(Q_bhtd, position_ids=position_ids, rope_base=layer.rope_base)  # [B,Hq,Tq,D] fp32
        # Use the same top_idx; compute logits vs reps only (small inline impl):
        repsK = layer.repsK.to(torch.float32)         # [C,Hkv,D]
        repsV = layer.repsV                           # [C,Hkv,D]
        log_sizes = layer.log_sizes.view(1,1,-1)      # [1,1,C]
        C, Hkv, D = repsK.shape
        gqa_map = getattr(cfg, "gqa_map", None)
        def _kv_head_for(hq: int) -> int:
            if gqa_map is None: return hq % Hkv
            return int(gqa_map[hq])
        scale = cfg.scale_override if getattr(cfg, "scale_override", None) is not None else (1.0 / math.sqrt(D))

        out = torch.empty_like(Q_bhtd)
        for b in range(B):
            for t in range(Tq):
                idx = top_idx[b, t]                              # [topC]
                for hq in range(Hq):
                    kvh = _kv_head_for(hq)
                    q = q_un[b, hq, t].view(1, D)                # [1,D] f32
                    K_bank = repsK[:, kvh, :]                    # [C,D] f32
                    V_bank = repsV[:, kvh, :]                    # [C,D]
                    logits = torch.matmul(q, K_bank.t()) * scale # [1,C]
                    if getattr(cfg, "mass_alpha", 0.0) != 0.0:
                        logits = logits + cfg.mass_alpha * log_sizes.to(logits.dtype)
                    top_logits = self._take_along_last_dim(logits, idx)  # [1,topC]
                    V_sel = V_bank.index_select(0, idx).unsqueeze(0)                 # [1,topC,D]
                    attn = F.softmax(top_logits.to(torch.float32), dim=-1).to(Q_bhtd.dtype)  # [1,topC]
                    out[b, hq, t, :] = torch.matmul(attn, V_sel).squeeze(0)
        return out

    # --- add inside class KeygraphAttentionPatch ---
    def _take_along_last_dim(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Robust gather on last dim. Works with logits shaped [B,C], [B,T,C], or [B,H,T,C].
        idx can be [B,topC] or [B,T,topC]; we broadcast as needed.
        """
        # x: [..., C]
        # idx: [B, topC] or [B,T,topC]
        if idx.dtype != torch.long:
            idx = idx.long()
        # Align batch/time dims: make idx have the same number of dims as x
        # Insert singleton dims right before the last dim until ranks match.
        while idx.dim() < x.dim():
            # Heuristic: add a head dim if x has 4 dims (B,H,T,C) and idx has 3 (B,T,topC)
            if x.dim() == 4 and idx.dim() == 3:
                # idx: [B,T,topC] -> [B,1,T,topC]
                idx = idx.unsqueeze(1)
            else:
                # Generic: prepend a singleton after B if possible, else at front
                idx = idx.unsqueeze(-2)  # [..., 1, topC]
        # Now expand idx to match x except on last dim
        expand_shape = list(x.shape[:-1]) + [idx.shape[-1]]
        idx = idx.expand(*expand_shape)
        return torch.gather(x, dim=-1, index=idx)


    # --- add/replace inside class KeygraphAttentionPatch ---
    def _ensure_top_idx(
        self,
        Q_bhtd: torch.Tensor,           # [B,Hq,Tq,D] queries (RoPE space)
        position_ids: torch.Tensor,     # [B,Tq]
        top_idx: Optional[torch.Tensor] # [B,Tq,topC] or None
    ) -> torch.Tensor:
        if top_idx is not None:
            return top_idx

        layer = self.cache
        cfg   = self.cfg
        device = Q_bhtd.device
        B, Hq, Tq, D = Q_bhtd.shape

        repsK = layer.repsK.to(torch.float32)  # [C,Hkv,D], UN-RoPE
        C, Hkv, _ = repsK.shape
        topC = int(min(getattr(cfg, "topC", 32), C))

        # Map query heads to KV heads (GQA)
        gqa_map = getattr(cfg, "gqa_map", None)
        if gqa_map is None:
            kv_heads = torch.arange(Hq, device=device) % Hkv  # [Hq]
        else:
            kv_heads = torch.as_tensor(gqa_map, device=device)  # [Hq]

        # Un-RoPE queries token-wise to match repsK space
        q_un = _unrope_queries_tokenwise(Q_bhtd, position_ids=position_ids, rope_base=layer.rope_base).to(torch.float32)  # [B,Hq,Tq,D]

        # repsK_sel: [Hq, C, D]
        repsK_sel = repsK.permute(1, 0, 2)[kv_heads]  # [Hq,C,D]

        scale = 1.0 / math.sqrt(D)
        # logits_full: [B,Hq,Tq,C]
        logits_full = torch.einsum('bhtd,hcd->bhtc', q_un, repsK_sel) * scale

        # Optional size/mass bias
        mass_alpha = float(getattr(cfg, "mass_alpha", 0.0))
        if mass_alpha != 0.0 and getattr(layer, "mass", None) is not None:
            mass = layer.mass.to(logits_full.dtype).to(device)
            if mass.dim() == 2:
                mass = mass.mean(dim=1)  # [C]
            logits_full = logits_full + mass_alpha * torch.log(mass.clamp_min(1e-6))[None, None, None, :]

        # Aggregate heads and take topC over C -> idx: [B,Tq,topC]
        agg = logits_full.mean(dim=1)                     # [B,Tq,C]
        _, idx = torch.topk(agg, k=topC, dim=-1, largest=True, sorted=False)  # [B,Tq,topC]
        return idx.contiguous()



    def _gather_rescue_token_bank(
        self,
        rescue_clusters: List[int],
        layer: "LayerKeygraphCache",
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build a token bank from per-cluster probes (no full-KV required).
        Returns: (K_tok [Nt,Hkv,D], V_tok [Nt,Hkv,D], pos_idx_tok [Nt])
        Nt = total selected probe tokens across clusters (skips zero-padded rows)
        """
        assert hasattr(layer, "K_probe") and hasattr(layer, "V_probe") and hasattr(layer, "probe_idx")
        Kp = layer.K_probe   # [C,Hkv,m,D] on CPU pinned
        Vp = layer.V_probe   # [C,Hkv,m,D]
        idx_list = layer.probe_idx  # list[C] -> LongTensor[m_valid]
        Hkv = Kp.shape[1]
        Ks, Vs, Ps = [], [], []

        for c in rescue_clusters:
            # valid probe indices for this cluster
            pick = idx_list[c]
            if pick is None or pick.numel() == 0:
                continue
            m_valid = int(pick.numel())
            Ks.append(Kp[c, :, :m_valid, :].to(device, non_blocking=True))   # [Hkv,mv,D]
            Vs.append(Vp[c, :, :m_valid, :].to(device, non_blocking=True))
            # map to absolute positions via layer.pos_idx
            Ps.append(layer.pos_idx.index_select(0, pick.to(layer.pos_idx.device)).to(device))

        if not Ks:
            return (torch.empty(0, 0, 0, device=device), torch.empty(0, 0, 0, device=device), torch.empty(0, device=device, dtype=torch.long))

        K_cat = torch.cat(Ks, dim=1)   # [Hkv, Nt, D]
        V_cat = torch.cat(Vs, dim=1)   # [Hkv, Nt, D]
        P_cat = torch.cat(Ps, dim=0)   # [Nt]
        # transpose to [Nt,Hkv,D] for easier per-head indexing later
        return K_cat.permute(1,0,2).contiguous(), V_cat.permute(1,0,2).contiguous(), P_cat


    # ------------------------------ Stage-A ------------------------------

    def _make_query_descriptors(
        self, Q_bhtd: torch.Tensor, position_ids: torch.Tensor, rope_base: float, rp: torch.Tensor
    ) -> torch.Tensor:
        """
        Build φ(q) like in build_descriptors_unrope for keys:
        UnRoPE -> per-head ℓ2 -> mean across heads -> RP -> ℓ2
        Returns [B, Tq, r] (fp32 unit-norm).
        """
        B, Hq, Tq, D = Q_bhtd.shape
        # Un-apply RoPE for Q
        Q_un = _unrope_queries(Q_bhtd, position_ids, rope_base=rope_base)  # [B,Hq,Tq,D]
        # Per-head ℓ2
        Q_n = F.normalize(Q_un.to(torch.float32), dim=-1, eps=1e-6)        # [B,Hq,Tq,D]
        # Mean across heads -> [B, Tq, D]
        Q_mean = Q_n.mean(dim=1, keepdim=False)                            # [B,Tq,D]
        # Random projection -> [B,Tq,r]
        phi = torch.einsum("btd,dr->btr", Q_mean, rp.to(Q_mean))           # fp32
        # Unit norm
        phi = F.normalize(phi, dim=-1, eps=1e-6)
        return phi

    def _decide_rescue_with_probes(
        self,
        Q_bhtd: torch.Tensor,           # [B, Hq, Tq, D]  (RoPE-applied)
        top_idx: torch.Tensor,          # [B, Tq, topC]
    ) -> Tuple[torch.Tensor, List[List[List[int]]]]:
        """
        Decide rescue via **attention-space** variance on per-cluster probe K:
        For each shortlisted cluster c, compute Var((K_probe[c] · q)/sqrt(D)) averaged across heads.
        If mean variance > eps → mark cluster c for rescue at (b,t).

        Returns:
          rescue_mask_bt: [B, Tq] bool
          rescue_clusters_bt: nested list of lists of cluster ids to expand
        """
        cfg   = self.cfg
        layer = self.cache
        device = Q_bhtd.device
        B, Hq, Tq, D = Q_bhtd.shape
        _, _, topC = top_idx.shape
        sqrt_d = D ** 0.5

        rescue_mask = torch.zeros(B, Tq, dtype=torch.bool, device=device)
        rescue_clusters: List[List[List[int]]] = [[[] for _ in range(Tq)] for _ in range(B)]

        if (not cfg.enable_rescue) or (getattr(layer, "K_probe", None) is None) or (getattr(layer, "probe_idx", None) is None):
            return rescue_mask, rescue_clusters

        # Shapes:
        #   layer.K_probe: [C, Hkv, m, D]  (pinned CPU by builder)
        #   layer.probe_idx: list[C] of LongTensor[m_valid]
        #   layer.repsK: [C, Hkv, D]
        Kp = layer.K_probe               # may be CPU pinned; we'll .to(device) per cluster slice when needed
        probe_idx_list = layer.probe_idx
        C, Hkv, m_alloc, _ = Kp.shape

        for b in range(B):
            for t in range(Tq):
                any_high = False
                clusters_to_rescue: List[int] = []

                for k in range(topC):
                    c = int(top_idx[b, t, k].item())
                    if c < 0 or c >= C:
                        continue
                    prob_ids = probe_idx_list[c] if (probe_idx_list is not None and c < len(probe_idx_list)) else None
                    m_valid = int(prob_ids.numel()) if (prob_ids is not None) else 0
                    if m_valid < 2:
                        continue

                    # Aggregate variance across mapped heads
                    var_acc = 0.0
                    for hq in range(Hq):
                        kvh = _kv_head_for(hq, Hkv, cfg.gqa_map)
                        # Take only the valid probe rows [m_valid, D]
                        Kc = Kp[c, kvh, :m_valid, :].to(device=device, dtype=Q_bhtd.dtype)
                        q  = Q_bhtd[b, hq, t, :]                       # [D]
                        logits = (Kc @ q) / sqrt_d                     # [m_valid]
                        var_acc = var_acc + torch.var(logits, unbiased=False)
                    var_acc = var_acc / Hq

                    if var_acc.item() > cfg.rescue_var_eps:
                        any_high = True
                        clusters_to_rescue.append(c)

                if any_high:
                    rescue_mask[b, t] = True
                    rescue_clusters[b][t] = clusters_to_rescue

        return rescue_mask, rescue_clusters


    # ------------------------------ Stage-B paths ------------------------------

    # In kg_patch.py

    def _attend_reps_only(
        self,
        Q_bhtd: torch.Tensor,
        K_cur_bhsd: Optional[torch.Tensor],
        V_cur_bhsd: Optional[torch.Tensor],
        top_idx: Optional[torch.Tensor]
    ) -> torch.Tensor:
        layer = self.cache
        cfg = self.cfg
        B, Hq, Tq, D = Q_bhtd.shape
        Hkv = layer.repsK.shape[1]
        C   = layer.num_clusters()

        # UnRoPE query once (shared for all heads)
        Q_un = _unrope_queries(Q_bhtd, position_ids=torch.zeros(B, Tq, device=Q_bhtd.device, dtype=torch.long) if Tq==0 else torch.arange(0, Tq, device=Q_bhtd.device, dtype=torch.long).unsqueeze(0).expand(B,-1),
                            rope_base=layer.rope_base).to(torch.float32)  # [B,Hq,Tq,D]

        # repsK are already un-rotated (Patch #1)
        repsK = layer.repsK.to(torch.float32)   # [C,Hkv,D]
        repsV = layer.repsV                     # [C,Hkv,D]
        log_sizes = layer.log_sizes.view(1, 1, C)

        scale = cfg.scale_override if cfg.scale_override is not None else (1.0 / math.sqrt(D))
        out = torch.empty(B, Hq, Tq, D, device=Q_bhtd.device, dtype=Q_bhtd.dtype)

        has_reps = top_idx is not None
        has_curr = (K_cur_bhsd is not None) and (V_cur_bhsd is not None)

        # UnRoPE current step key if present
        if has_curr:
            # position_ids for current step are given in forward(...); capture via closure arg there
            # here we assume Tq==1 during decode; use provided position_ids from forward call
            pass  # handled in forward where we call this with K_cur_un already if needed

        for hq in range(Hq):
            kvh = _kv_head_for(hq, Hkv, cfg.gqa_map)

            logits_list = []
            Vbanks_list = []

            if has_reps:
                K_bank = repsK[:, kvh, :]  # [C,D] (un-rotated)
                V_bank = repsV[:, kvh, :]  # [C,D]

                rep_logits = torch.matmul(Q_un[:, hq], K_bank.t())  # [B,Tq,C]
                rep_logits = _maybe_scale_logits(rep_logits, scale)
                if cfg.mass_alpha != 0.0:
                    rep_logits = rep_logits + cfg.mass_alpha * log_sizes.to(rep_logits)

                Bsz, Tlen, _ = rep_logits.shape
                idx = top_idx
                top_rep_logits = torch.gather(rep_logits, dim=-1, index=idx)
                V_sel = V_bank.index_select(0, idx.view(-1)).view(Bsz, Tlen, idx.shape[-1], D)
                logits_list.append(top_rep_logits)
                Vbanks_list.append(V_sel)

            if has_curr:
                # K_cur_bhsd: [B, Hkv, 1, D]; we need un-rotated
                # Use position_ids supplied to forward (single-step). We'll pass K_cur_un from there.
                pass

            final_logits = torch.cat(logits_list, dim=-1) if logits_list else None
            final_V = torch.cat(Vbanks_list, dim=2) if Vbanks_list else None

            if final_logits is None:
                out[:, hq] = torch.zeros((B, Tq, D), device=out.device, dtype=out.dtype)
                continue

            attn = F.softmax(final_logits.to(torch.float32), dim=-1).to(Q_bhtd.dtype)
            out[:, hq] = torch.einsum("btc,btcd->btd", attn, final_V)

        return out


    def _attend_mixed_with_rescue(
        self,
        Q_bhtd: torch.Tensor,                       # [B,Hq,Tq,D]  (RoPE-applied queries)
        K_bhsd: Optional[torch.Tensor],             # [B,Hkv,S,D]  (full past keys; can be None)
        V_bhsd: Optional[torch.Tensor],             # [B,Hkv,S,D]  (full past vals; can be None)
        K_cur_bhsd: Optional[torch.Tensor],         # [B,Hkv,1,D]  current-step key (RoPE); optional
        V_cur_bhsd: Optional[torch.Tensor],         # [B,Hkv,1,D]  current-step value; optional
        attn_mask: Optional[torch.Tensor],          # usually None at decode
        top_idx: torch.Tensor,                      # [B,Tq,topC]  shortlist indices into reps
        rescue_mask_bt: torch.Tensor,               # [B,Tq] bool  whether to add token bank
        rescue_clusters_bt: torch.Tensor,           # [B,Tq,R] long clusters to rescue (R per step)
        position_ids: torch.Tensor,                 # [B,Tq] long  absolute positions of queries
    ) -> torch.Tensor:
        """
        Mixed attention over (a) representative centroids (already un-RoPE’d in cache),
        (b) a token bank (full-KV members OR per-cluster probes), and (c) the current token.
        All logits are computed in the **same un-RoPE space**. This function makes explicit
        where q_un is computed and how position_ids are used.

        Returns: context [B,Hq,Tq,D] (same dtype as Q_bhtd)
        """
        layer = self.cache
        cfg   = self.cfg
        device = Q_bhtd.device
        B, Hq, Tq, D = Q_bhtd.shape

        # Representatives (already un-rotated by the cache build step!)
        repsK = layer.repsK.to(torch.float32)          # [C,Hkv,D]
        repsV = layer.repsV                            # [C,Hkv,D]
        log_sizes = layer.log_sizes.view(1, 1, -1)     # [1,1,C]
        C   = repsK.shape[0]
        Hkv = repsK.shape[1]

        # Map query-head -> kv-head for GQA
        gqa_map = getattr(cfg, "gqa_map", None)
        def _kv_head_for(hq: int) -> int:
            if gqa_map is None: return hq % Hkv
            return int(gqa_map[hq])

        # scale for QK^T
        scale = cfg.scale_override if getattr(cfg, "scale_override", None) is not None else (1.0 / math.sqrt(D))

        # Un-RoPE queries ONCE for the whole [B,Hq,Tq,D]
        # (We inverse-rotate per token using the provided absolute position_ids)
        q_un_bhtd = _unrope_queries_tokenwise(Q_bhtd, position_ids=position_ids, rope_base=layer.rope_base)  # float32

        # Prepare output
        out = torch.empty((B, Hq, Tq, D), device=device, dtype=Q_bhtd.dtype)

        # Members (full-KV) if available
        members_by_c = getattr(layer, "members_by_c", None)   # Optional[List[Tensor]]

        # Iterate tokens (decode is typically Tq == 1, but this is generic)
        for b in range(B):
            for t in range(Tq):
                # shortlist representatives for this (b,t)
                idx = top_idx[b, t]                            # [topC]
                # which clusters to rescue as token-bank?
                rescue = bool(rescue_mask_bt[b, t].item())
                rescue_cs = rescue_clusters_bt[b, t] if rescue else torch.empty(0, dtype=torch.long, device=device)

                # Build token bank if requested
                K_tok_bank = V_tok_bank = pos_tok = None
                if rescue and (rescue_cs.numel() > 0):
                    # try full-KV members if we have them; fall back to probe-only
                    if (K_bhsd is not None) and (V_bhsd is not None) and (members_by_c is not None):
                        # de-duplicate indices across rescued clusters
                        uniq = []
                        seen = set()
                        for c in rescue_cs.tolist():
                            if c < len(members_by_c):
                                for s in members_by_c[c].tolist():
                                    if s not in seen:
                                        uniq.append(s); seen.add(s)
                        if len(uniq) > 0:
                            idx_t = torch.tensor(uniq, device=device, dtype=torch.long)   # [Nt]
                            # gather [Nt,Hkv,D]; use layer.pos_idx to get absolute positions
                            K_tok_bank = K_bhsd[b].permute(1,0,2).index_select(0, idx_t)  # [Nt,Hkv,D]
                            V_tok_bank = V_bhsd[b].permute(1,0,2).index_select(0, idx_t)  # [Nt,Hkv,D]
                            pos_tok    = layer.pos_idx.index_select(0, idx_t.to(layer.pos_idx.device)).to(device)  # [Nt]
                    if (K_tok_bank is None) or (V_tok_bank is None):
                        # probe-only path (works when store_full_kv=False)
                        K_tok_bank, V_tok_bank, pos_tok = self._gather_rescue_token_bank(list(rescue_cs.tolist()), layer, device)

                # current-step key/value (RoPE) -> we will un-RoPE per head later if present
                have_cur = (K_cur_bhsd is not None) and (V_cur_bhsd is not None)

                for hq in range(Hq):
                    kvh = _kv_head_for(hq)

                    logits_blocks = []
                    values_blocks = []

                    # ----- (A) Representatives (already UN-RoPE’d) -----
                    K_bank_rep = repsK[:, kvh, :]           # [C,D] (float32, un-rotated)
                    V_bank_rep = repsV[:, kvh, :]           # [C,D] (model dtype)
                    # q_un for this (b,hq,t): [D]
                    q_un = q_un_bhtd[b, hq, t].view(1, D)   # [1,D] float32

                    rep_logits = torch.matmul(q_un, K_bank_rep.t())   # [1,C]
                    rep_logits = rep_logits * scale
                    if getattr(cfg, "mass_alpha", 0.0) != 0.0:
                        rep_logits = rep_logits + cfg.mass_alpha * log_sizes.to(rep_logits.dtype)  # [1,1,C]

                    # shortlist reps by top_idx
                    rep_sel = idx.view(1, -1)                               # [1,topC]
                    rep_logits_top = torch.gather(rep_logits, dim=-1, index=rep_sel)  # [1,topC]
                    V_rep_top = V_bank_rep.index_select(0, idx)             # [topC,D]

                    logits_blocks.append(rep_logits_top)                    # list of [1,?]
                    values_blocks.append(V_rep_top.unsqueeze(0))            # list of [1,?,D]

                    # ----- (B) Token bank (if any) — UN-RoPE with pos_tok -----
                    if (K_tok_bank is not None) and (K_tok_bank.numel() > 0):
                        # head slice
                        K_tok_h = K_tok_bank[:, kvh, :].unsqueeze(0).unsqueeze(0)  # [1,1,Nt,D] (RoPE space)
                        # un-rotate per token by their absolute positions
                        pos_nt = pos_tok.view(1, -1)                                # [1,Nt]
                        K_tok_un = _unrope_keys_tokenwise(K_tok_h, position_ids=pos_nt, rope_base=layer.rope_base)
                        K_tok_un = K_tok_un.squeeze(0).squeeze(0)                   # [Nt,D] float32

                        tok_logits = torch.matmul(q_un, K_tok_un.t()) * scale       # [1,Nt]
                        logits_blocks.append(tok_logits)
                        values_blocks.append(V_tok_bank[:, kvh, :].unsqueeze(0))    # [1,Nt,D]

                    # ----- (C) Current-step token (if provided) — UN-RoPE with position_ids[b,t] -----
                    if have_cur:
                        Kc = K_cur_bhsd[b, kvh:kvh+1, :, :]                         # [1,1,D]
                        pos_bt = position_ids[b:b+1, t:t+1]                         # [1,1]
                        Kc_un = _unrope_keys_tokenwise(Kc, position_ids=pos_bt, rope_base=layer.rope_base)  # [1,1,D]
                        cur_logits = torch.matmul(q_un, Kc_un.view(1, D).t()) * scale  # [1,1]
                        logits_blocks.append(cur_logits)
                        values_blocks.append(V_cur_bhsd[b, kvh:kvh+1, :, :].view(1, 1, D))  # [1,1,D]

                    # ----- (D) Mix and apply softmax -----
                    logits_cat = torch.cat(logits_blocks, dim=-1)      # [1, M]
                    # (attn_mask is usually None at decode; handle if you keep one)
                    attn = F.softmax(logits_cat.to(torch.float32), dim=-1).to(Q_bhtd.dtype)  # [1,M]
                    V_cat = torch.cat(values_blocks, dim=1)            # [1, M, D]
                    out[b, hq, t, :] = torch.matmul(attn, V_cat).squeeze(0)  # [D]

        return out



    # ------------------------------ Exact attention (safe fallback) ------------------------------

    def _attend_exact(
        self,
        Q_bhtd: torch.Tensor,
        K_bhsd: torch.Tensor,
        V_bhsd: torch.Tensor,
        K_cur_bhsd: Optional[torch.Tensor], # NEW
        V_cur_bhsd: Optional[torch.Tensor], # NEW
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Pure exact attention over provided K/V (safe, portable).
        *** NOW INCLUDES CURRENT K/V ***
        """
        
        # *** FIX: Augment Past K/V with Current K/V ***
        K_past = K_bhsd # [B, Hkv, S, D]
        V_past = V_bhsd # [B, Hkv, S, D]
        
        has_curr = K_cur_bhsd is not None and V_cur_bhsd is not None
        if has_curr:
            K_aug = torch.cat([K_past, K_cur_bhsd], dim=2) # [B, Hkv, S+1, D]
            V_aug = torch.cat([V_past, V_cur_bhsd], dim=2)
            
            # Also augment the mask if it exists
            if attn_mask is not None:
                # attn_mask is [B, 1, Tq, S]
                # Add a column of 0 (or True) for the current token
                mask_curr_shape = (B, 1, Tq, 1)
                if attn_mask.dtype == torch.bool:
                    mask_curr = torch.ones(mask_curr_shape, device=attn_mask.device, dtype=torch.bool)
                else:
                    mask_curr = torch.zeros(mask_curr_shape, device=attn_mask.device, dtype=attn_mask.dtype)
                attn_mask = torch.cat([attn_mask, mask_curr], dim=3) # [B, 1, Tq, S+1]
        else:
            K_aug, V_aug = K_past, V_past
        
        
        B, Hq, Tq, D = Q_bhtd.shape
        _, Hkv, S_aug, _ = K_aug.shape
        scale = 1.0 / math.sqrt(D)
        out = torch.empty(B, Hq, Tq, D, device=Q_bhtd.device, dtype=Q_bhtd.dtype)
        
        for hq in range(Hq):
            kvh = _kv_head_for(hq, Hkv, self.cfg.gqa_map)
            K = K_aug[:, kvh, :, :]   # [B,S_aug,D]
            V = V_aug[:, kvh, :, :]
            logits = torch.matmul(Q_bhtd[:, hq], K.transpose(-1, -2))  # [B,Tq,S_aug]
            logits = _maybe_scale_logits(logits, scale)
            if attn_mask is not None:
                logits = _apply_mask_logits(logits, attn_mask.squeeze(1))  # [B,Tq,S_aug]
            attn = F.softmax(logits.to(torch.float32), dim=-1).to(Q_bhtd.dtype)  # [B,Tq,S_aug]
            out[:, hq] = torch.matmul(attn, V)
        return out


# -----------------------------------------------------------------------------
# Integration helper
# -----------------------------------------------------------------------------

def integrate_with_llama_attention(
    llama_attention_module: nn.Module,
    layer_cache: LayerKeygraphCache,
    cfg: PatchConfig,
):
    """
    Returns a forward function you can plug into your model instead of the default attn forward.

    Expected call signature (B,H,T,D layout):
        patched_forward(
            Q_bhtd,            # [B,Hq,Tq,D]  (RoPE-applied queries)
            K_bhsd,            # [B,Hkv,S,D]  (RoPE-applied keys, optional for reps-only)
            V_bhsd,            # [B,Hkv,S,D]
            position_ids,      # [B,Tq]
            attention_mask,    # [B,1,Tq,S]
        ) -> [B,Hq,Tq,D]

    Notes:
      - For **representatives-only** operation, you can pass `K_bhsd=None, V_bhsd=None`.
      - For **rescue** or **exact fallback**, pass valid K/V tensors (only used when needed).
    """
    patch = KeygraphAttentionPatch(llama_attention_module, layer_cache, cfg)

    def patched_forward(
        Q_bhtd: torch.Tensor,
        K_bhsd: Optional[torch.Tensor],
        V_bhsd: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compute current-step K/V from the module (single-step decode assumed)
        hidden_states = getattr(llama_attention_module, "_kg_hidden_states", None)
        K_cur = V_cur = None
        if hidden_states is not None:
            q_proj = llama_attention_module.q_proj  # exists if it's a LLaMA-style block
            Dh = getattr(llama_attention_module, "head_dim", Q_bhtd.shape[-1] // max(1, getattr(llama_attention_module, "num_attention_heads", 1)))
            k_vec = llama_attention_module.k_proj(hidden_states)  # [B,1,Hkv*Dh]
            v_vec = llama_attention_module.v_proj(hidden_states)
            Hkv = k_vec.shape[-1] // Dh
            K_cur = k_vec.view(Q_bhtd.shape[0], 1, Hkv, Dh).permute(0,2,1,3).contiguous()
            V_cur = v_vec.view(Q_bhtd.shape[0], 1, Hkv, Dh).permute(0,2,1,3).contiguous()
        return patch(
            Q_bhtd=Q_bhtd,
            K_bhsd=K_bhsd,
            V_bhsd=V_bhsd,
            K_cur_bhsd=K_cur,
            V_cur_bhsd=V_cur,
            position_ids=position_ids,
            attn_mask=attention_mask,
        )

    return patched_forward
