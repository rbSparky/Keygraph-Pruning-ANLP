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
    # D must be even
    assert head_dim % 2 == 0, "RoPE head_dim must be even"
    half = head_dim // 2
    # inv_freq: [half] on device
    # NOTE: using "2*i/head_dim" exponent matches common RoPE usage
    idx = torch.arange(0, half, device=device, dtype=torch.float32)
    inv_freq = (base ** (-2 * idx / head_dim))  # [half]
    # positions: [B, T, 1]
    pos = position_ids.to(dtype=torch.float32).unsqueeze(-1)
    # freqs: [B, T, half]
    freqs = pos * inv_freq
    # Build [B, T, D] by interleaving (cos, sin)
    cos = torch.zeros((*freqs.shape[:-1], head_dim), device=device, dtype=dtype)
    sin = torch.zeros_like(cos)
    cos[..., ::2] = torch.cos(freqs)
    cos[..., 1::2] = torch.cos(freqs)
    sin[..., ::2] = torch.sin(freqs)
    sin[..., 1::2] = torch.sin(freqs)
    # Add a head axis to broadcast over Hq
    cos = cos.unsqueeze(1)  # [B, 1, T, D]
    sin = sin.unsqueeze(1)  # [B, 1, T, D]
    return cos, sin

def _unrope_queries(Q_bhtd: torch.Tensor, position_ids: torch.Tensor, rope_base: float) -> torch.Tensor:
    """
    Invert the RoPE rotation for queries.
    Q_bhtd: [B, Hq, T, D] (already RoPE-applied).
    position_ids: [B, T].
    Return: un-RoPE'd queries, [B, Hq, T, D].
    """
    B, Hq, T, D = Q_bhtd.shape
    cos, sin = _rope_cos_sin(D, position_ids, rope_base, Q_bhtd.device, Q_bhtd.dtype)
    # Inverse rotation is: x = y*cos - rotate_half(y)*sin
    return (Q_bhtd * cos) - (_rotate_half(Q_bhtd) * sin)


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
        Q_bhtd: torch.Tensor,               # [B, Hq, Tq, D] (RoPE-applied)
        K_bhsd: Optional[torch.Tensor],     # [B, Hkv, S, D] (RoPE-applied)  (optional, used only if exact/rescue with tokens)
        V_bhsd: Optional[torch.Tensor],     # [B, Hkv, S, D]                  (optional)
        position_ids: torch.Tensor,         # [B, Tq]
        attn_mask: Optional[torch.Tensor] = None,  # [B, 1, Tq, S] (only for token paths)
    ) -> torch.Tensor:
        layer = self.cache
        cfg = self.cfg
        B, Hq, Tq, D = Q_bhtd.shape
        Hkv = layer.repsK.shape[1]  # [C, Hkv, D]
        C = layer.num_clusters()
        S = layer.num_tokens()

        # Fast fallback: if S is tiny and we have full K/V, use exact attention.
        if (layer.K is not None) and (layer.V is not None) and (S <= cfg.small_S_exact_fallback) and (K_bhsd is not None) and (V_bhsd is not None):
            return self._attend_exact(Q_bhtd, K_bhsd, V_bhsd, attn_mask)

        # Representatives-only path (default)
        # Stage-A: compute query descriptor φ(q)
        phi_q = self._make_query_descriptors(Q_bhtd, position_ids, rope_base=layer.rope_base, rp=layer.rp_matrix)  # [B, Tq, r]
        # Score vs centroid descriptors
        scores_ct = torch.matmul(phi_q, layer.repsPhi.t())  # [B, Tq, C]
        if cfg.mass_alpha != 0.0:
            scores_ct = scores_ct + cfg.mass_alpha * layer.log_sizes.view(1, 1, C)

        # Shortlist top clusters
        topC = min(cfg.top_clusters, C) if C > 0 else 0
        if topC == 0:
            # No clusters? fall back to zero output
            return torch.zeros(B, Hq, Tq, D, device=Q_bhtd.device, dtype=Q_bhtd.dtype)

        top_scores, top_idx = torch.topk(scores_ct, k=topC, dim=-1)  # [B, Tq, topC]
        # Decide rescue per (b,t) by probe variance among shortlisted clusters
        self._Q_for_rescue = Q_bhtd        # [B, Hq, Tq, D]
        self._K_for_rescue = K_bhsd 
        rescue_mask_bt, rescue_clusters_bt = self._decide_rescue(phi_q, top_idx)

        # If representatives-only or K/V absent => always use reps
        if cfg.use_representatives_only or (layer.K is None) or (layer.V is None) or (K_bhsd is None) or (V_bhsd is None):
            return self._attend_reps_only(Q_bhtd, top_idx)

        # Otherwise, mix token rescue for high-variance clusters
        return self._attend_mixed_with_rescue(Q_bhtd, K_bhsd, V_bhsd, attn_mask, top_idx, rescue_mask_bt, rescue_clusters_bt)

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

    def _decide_rescue(
        self,
        phi_q: torch.Tensor,      # [B, Tq, r] query descriptors (unit-norm fp32)
        top_idx: torch.Tensor,    # [B, Tq, topC] shortlisted cluster ids
    ) -> Tuple[torch.Tensor, List[List[List[int]]]]:
        """
        Decide rescue by measuring dispersion inside each shortlisted cluster.

        Preferred (when available): ATTENTION-SPACE variance
        Var((Q · K_probe) / sqrt(D)) aggregated across heads.
        Fallback: DESCRIPTOR-SPACE cosine variance with cluster probe descriptors.

        Returns:
        rescue_mask_bt: [B, Tq] bool — whether (b,t) needs token rescue
        rescue_clusters_bt: nested list of cluster ids to expand for (b,t)
        """
        cfg   = self.cfg
        layer = self.cache
        device = phi_q.device
        B, Tq, topC = top_idx.shape

        rescue_mask = torch.zeros(B, Tq, dtype=torch.bool, device=device)
        rescue_clusters: List[List[List[int]]] = [[[] for _ in range(Tq)] for _ in range(B)]

        if not cfg.enable_rescue:
            return rescue_mask, rescue_clusters

        # Try attention-space path if Q/K are available
        Q_bhtd = getattr(self, "_Q_for_rescue", None)   # [B, Hq, Tq, D]
        K_bhsd = getattr(self, "_K_for_rescue", None)   # [B, Hkv, S, D] or None
        attn_space_ok = (Q_bhtd is not None) and (K_bhsd is not None)

        if attn_space_ok:
            Bq, Hq, Tq_q, D = Q_bhtd.shape
            _, Hkv, S, _ = K_bhsd.shape
            assert Bq == B and Tq_q == Tq, "Q shape mismatch for rescue"
            sqrt_d = D ** 0.5

            for b in range(B):
                for t in range(Tq):
                    any_high = False
                    clusters_to_rescue: List[int] = []

                    for k in range(topC):
                        c = int(top_idx[b, t, k].item())

                        # Probe token indices for this cluster (CPU by design)
                        if layer.probe_sets is None or layer.probe_sets[c] is None or layer.probe_sets[c].numel() < 2:
                            continue
                        tok_idx = layer.probe_sets[c].to(device=device, dtype=torch.long)  # [P]
                        if tok_idx.numel() < 2:
                            continue

                        # Aggregate variance across Q-heads mapped to KV-heads (GQA-friendly)
                        var_acc = 0.0
                        for hq in range(Hq):
                            kvh = _kv_head_for(hq, Hkv, cfg.gqa_map)
                            q = Q_bhtd[b, hq, t, :]                  # [D]
                            Kp = K_bhsd[b, kvh, tok_idx, :]          # [P, D]
                            logits = (Kp @ q) / sqrt_d               # [P]
                            var_acc = var_acc + torch.var(logits, unbiased=False)
                        var_acc = var_acc / Hq

                        if var_acc.item() > cfg.rescue_var_eps:
                            any_high = True
                            clusters_to_rescue.append(c)

                    if any_high:
                        rescue_mask[b, t] = True
                        rescue_clusters[b][t] = clusters_to_rescue

            return rescue_mask, rescue_clusters

        # -------------------- Fallback: descriptor-space variance --------------------
        if layer.probe_desc is None:
            return rescue_mask, rescue_clusters

        for b in range(B):
            for t in range(Tq):
                phi = phi_q[b, t]  # [r]
                any_high = False
                clusters_to_rescue: List[int] = []
                for k in range(topC):
                    c = int(top_idx[b, t, k].item())
                    pd = layer.probe_desc[c]
                    if pd is None or pd.numel() == 0:
                        continue
                    s = F.normalize(pd.to(device), dim=-1, eps=1e-6) @ phi  # [P]
                    var = torch.var(s, unbiased=False).item()
                    if var > cfg.rescue_var_eps:
                        any_high = True
                        clusters_to_rescue.append(c)
                if any_high:
                    rescue_mask[b, t] = True
                    rescue_clusters[b][t] = clusters_to_rescue

        return rescue_mask, rescue_clusters


    # ------------------------------ Stage-B paths ------------------------------

    def _attend_reps_only(self, Q_bhtd: torch.Tensor, top_idx: torch.Tensor) -> torch.Tensor:
        """
        Representatives-only attention (memory-saving). No attn_mask applied here.
        """
        layer = self.cache
        cfg = self.cfg
        B, Hq, Tq, D = Q_bhtd.shape
        Hkv = layer.repsK.shape[1]
        C = layer.num_clusters()

        # Gather the shortlisted reps once per head as matmul banks
        # We will compute per-head logits against [C, D], but only materialize top clusters for speed.
        # For simplicity and clarity, we compute full-C bank per head (C is usually small vs S).
        repsK = layer.repsK  # [C,Hkv,D]
        repsV = layer.repsV  # [C,Hkv,D]
        log_sizes = layer.log_sizes.view(1, 1, C)  # [1,1,C]

        scale = cfg.scale_override if cfg.scale_override is not None else (1.0 / math.sqrt(D))
        out = torch.empty(B, Hq, Tq, D, device=Q_bhtd.device, dtype=Q_bhtd.dtype)

        for hq in range(Hq):
            kvh = _kv_head_for(hq, Hkv, cfg.gqa_map)
            K_bank = repsK[:, kvh, :]  # [C,D]
            V_bank = repsV[:, kvh, :]  # [C,D]

            # logits: [B,Tq,C]
            logits = torch.matmul(Q_bhtd[:, hq], K_bank.t())  # [B,Tq,C]
            logits = _maybe_scale_logits(logits, scale)
            if cfg.mass_alpha != 0.0:
                logits = logits + cfg.mass_alpha * log_sizes.to(logits)

            # Optionally clip to shortlisted clusters to save compute
            # (we still need probs over only those clusters)
            Bsz, Tlen, _ = logits.shape
            # gather top logits per (b,t)
            top_logits = torch.gather(logits, dim=-1, index=top_idx)  # [B,Tq,topC]
            attn = F.softmax(top_logits.to(torch.float32), dim=-1).to(Q_bhtd.dtype)  # [B,Tq,topC]
            # Gather V for the same clusters and do a batched matmul
            V_sel = V_bank.index_select(0, top_idx.view(-1)).view(Bsz, Tlen, top_idx.shape[-1], D)  # [B,Tq,topC,D]
            out[:, hq] = torch.einsum("btc,btcd->btd", attn, V_sel)
        return out

    def _attend_mixed_with_rescue(
        self,
        Q_bhtd: torch.Tensor,
        K_bhsd: torch.Tensor,
        V_bhsd: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        top_idx: torch.Tensor,
        rescue_mask_bt: torch.Tensor,
        rescue_clusters_bt: List[List[List[int]]],
    ) -> torch.Tensor:
        """
        Mix token-level attention for rescued clusters with representative attention for the rest.
        Applies attn_mask to token logits; mass compensation only to representative logits.
        """
        layer = self.cache
        cfg = self.cfg
        B, Hq, Tq, D = Q_bhtd.shape
        _, Hkv, S, _ = K_bhsd.shape
        C = layer.num_clusters()

        scale = cfg.scale_override if cfg.scale_override is not None else (1.0 / math.sqrt(D))
        repsK = layer.repsK  # [C,Hkv,D]
        repsV = layer.repsV
        log_sizes = layer.log_sizes  # [C]

        out = torch.empty(B, Hq, Tq, D, device=Q_bhtd.device, dtype=Q_bhtd.dtype)

        # Precompute member lists per cluster (CPU), and optional probe lists
        # We'll map to GPU indices only for chosen (b,t).
        labels_cpu = layer.labels.detach().to("cpu")
        members_by_c: Dict[int, torch.Tensor] = {}
        for c in range(C):
            members_by_c[c] = torch.nonzero(labels_cpu == c, as_tuple=False).flatten()

        for b in range(B):
            for t in range(Tq):
                # Determine rescued clusters for this position
                rescue_cs = set(rescue_clusters_bt[b][t]) if bool(rescue_mask_bt[b, t].item()) else set()
                # Build two banks: representative clusters (non-rescued) and token shortlist for rescued ones
                # (1) Representative bank
                rep_clusters = []
                for k in range(top_idx.shape[-1]):
                    c = int(top_idx[b, t, k].item())
                    if c not in rescue_cs:
                        rep_clusters.append(c)
                rep_clusters_tensor = torch.tensor(rep_clusters, device=Q_bhtd.device, dtype=torch.long) if len(rep_clusters) else None

                # (2) Token shortlist for rescued clusters
                token_indices = []
                if len(rescue_cs) > 0:
                    for c in rescue_cs:
                        # Prefer probes first (already highest centroid-cos), else use first members
                        chosen = []
                        if layer.probe_sets is not None and layer.probe_sets[c] is not None and layer.probe_sets[c].numel() > 0:
                            # probe indices are ABSOLUTE token indices
                            ps = layer.probe_sets[c].to(torch.long).tolist()
                            chosen.extend(ps[: cfg.rescue_tokens_per_cluster])
                        # fill from remaining members if needed
                        if len(chosen) < cfg.rescue_tokens_per_cluster:
                            m = members_by_c[c].tolist()
                            # Avoid duplicates
                            for idx in m:
                                if idx not in chosen:
                                    chosen.append(int(idx))
                                if len(chosen) >= cfg.rescue_tokens_per_cluster:
                                    break
                        token_indices.extend(chosen)
                    # De-duplicate and keep order
                    seen = set()
                    token_indices = [i for i in token_indices if (i not in seen) and (not seen.add(i))]
                token_idx_tensor = torch.tensor(token_indices, device=Q_bhtd.device, dtype=torch.long) if len(token_indices) else None

                for hq in range(Hq):
                    kvh = _kv_head_for(hq, Hkv, cfg.gqa_map)
                    q = Q_bhtd[b, hq, t:t+1, :]  # [1,1,D]

                    # Representatives part
                    rep_logits = None
                    rep_V = None
                    if rep_clusters_tensor is not None and rep_clusters_tensor.numel() > 0:
                        K_rep = repsK.index_select(0, rep_clusters_tensor)[:, kvh, :]  # [Cr,D]
                        V_rep = repsV.index_select(0, rep_clusters_tensor)[:, kvh, :]  # [Cr,D]
                        rep_logits = torch.matmul(q, K_rep.t()).squeeze(0)  # [1,Cr] -> [Cr]
                        rep_logits = _maybe_scale_logits(rep_logits, scale)
                        rep_logits = rep_logits + cfg.mass_alpha * log_sizes.index_select(0, rep_clusters_tensor).to(rep_logits)
                        rep_V = V_rep  # [Cr,D]

                    # Token part (rescued clusters)
                    tok_logits = None
                    tok_V = None
                    if token_idx_tensor is not None and token_idx_tensor.numel() > 0:
                        K_tok = K_bhsd[b, kvh, token_idx_tensor, :]  # [Nt,D]
                        V_tok = V_bhsd[b, kvh, token_idx_tensor, :]  # [Nt,D]
                        tok_logits = torch.matmul(q, K_tok.t()).squeeze(0)  # [Nt]
                        tok_logits = _maybe_scale_logits(tok_logits, scale)
                        # Apply attention mask if provided: [B,1,Tq,S] -> slice S by token_idx_tensor
                        if attn_mask is not None:
                            # attn_mask[b,0,t,token_idx] additive
                            mask_slice = attn_mask[b, 0, t, token_idx_tensor]
                            tok_logits = _apply_mask_logits(tok_logits, mask_slice)

                        tok_V = V_tok  # [Nt,D]

                    if rep_logits is None and tok_logits is None:
                        # Nothing selected (corner case) -> zero out
                        out[b, hq, t] = torch.zeros(D, device=q.device, dtype=q.dtype)
                        continue

                    if rep_logits is None:
                        all_logits = tok_logits.unsqueeze(0)  # [1,Nt] for shape uniformity
                        all_V = tok_V.unsqueeze(0)            # [1,Nt,D]
                        # Softmax over tokens
                        attn = F.softmax(all_logits.to(torch.float32), dim=-1).to(q.dtype)  # [1,Nt]
                        out[b, hq, t:t+1] = torch.matmul(attn, all_V)  # [1,1,D]
                        continue

                    if tok_logits is None:
                        all_logits = rep_logits.unsqueeze(0)  # [1,Cr]
                        all_V = rep_V.unsqueeze(0)            # [1,Cr,D]
                        attn = F.softmax(all_logits.to(torch.float32), dim=-1).to(q.dtype)
                        out[b, hq, t:t+1] = torch.matmul(attn, all_V)
                        continue

                    # Mix representatives (with mass) + tokens (no mass)
                    all_logits = torch.cat([rep_logits, tok_logits], dim=-1).unsqueeze(0)  # [1,Cr+Nt]
                    all_V = torch.cat([rep_V, tok_V], dim=0).unsqueeze(0)                  # [1,Cr+Nt,D]
                    attn = F.softmax(all_logits.to(torch.float32), dim=-1).to(q.dtype)
                    out[b, hq, t:t+1] = torch.matmul(attn, all_V)

        return out

    # ------------------------------ Exact attention (safe fallback) ------------------------------

    def _attend_exact(
        self,
        Q_bhtd: torch.Tensor,
        K_bhsd: torch.Tensor,
        V_bhsd: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Pure exact attention over provided K/V (safe, portable).
        """
        B, Hq, Tq, D = Q_bhtd.shape
        _, Hkv, S, _ = K_bhsd.shape
        # Simple GQA: map each qh to kvh
        scale = 1.0 / math.sqrt(D)
        out = torch.empty(B, Hq, Tq, D, device=Q_bhtd.device, dtype=Q_bhtd.dtype)
        for hq in range(Hq):
            kvh = _kv_head_for(hq, Hkv, self.cfg.gqa_map)
            K = K_bhsd[:, kvh, :, :]   # [B,S,D]
            V = V_bhsd[:, kvh, :, :]
            logits = torch.matmul(Q_bhtd[:, hq], K.transpose(-1, -2))  # [B,Tq,S]
            logits = _maybe_scale_logits(logits, scale)
            if attn_mask is not None:
                logits = _apply_mask_logits(logits, attn_mask.squeeze(1))  # [B,Tq,S]
            attn = F.softmax(logits.to(torch.float32), dim=-1).to(Q_bhtd.dtype)  # [B,Tq,S]
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
        return patch(
            Q_bhtd=Q_bhtd,
            K_bhsd=K_bhsd,
            V_bhsd=V_bhsd,
            position_ids=position_ids,
            attn_mask=attention_mask,
        )

    return patched_forward
