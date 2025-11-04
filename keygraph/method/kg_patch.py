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
        K_bhsd: Optional[torch.Tensor],     # [B, Hkv, S, D] (PAST keys, optional)
        V_bhsd: Optional[torch.Tensor],     # [B, Hkv, S, D] (PAST values, optional)
        K_cur_bhsd: Optional[torch.Tensor], # [B, Hkv, 1, D] (NEW: CURRENT key)
        V_cur_bhsd: Optional[torch.Tensor], # [B, Hkv, 1, D] (NEW: CURRENT value)
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
        # *** FIX: Pass current K/V to exact fallback ***
        if (layer.K is not None) and (layer.V is not None) and (S <= cfg.small_S_exact_fallback) and (K_bhsd is not None) and (V_bhsd is not None):
            return self._attend_exact(Q_bhtd, K_bhsd, V_bhsd, K_cur_bhsd, V_cur_bhsd, attn_mask)

        # Stage-A: compute query descriptor φ(q) (decode-friendly)
        phi_q = self._make_query_descriptors(Q_bhtd, position_ids, rope_base=layer.rope_base, rp=layer.rp_matrix)  # [B, Tq, r]

        # Score vs centroid descriptors
        scores_ct = torch.matmul(phi_q, layer.repsPhi.t())  # [B, Tq, C]
        if cfg.mass_alpha != 0.0:
            scores_ct = scores_ct + cfg.mass_alpha * layer.log_sizes.view(1, 1, C)

        # Shortlist top clusters
        topC = min(cfg.top_clusters, C) if C > 0 else 0
        if topC == 0 and K_cur_bhsd is None: # No past and no present
            return torch.zeros(B, Hq, Tq, D, device=Q_bhtd.device, dtype=Q_bhtd.dtype)

        top_scores, top_idx = (None, None)
        if topC > 0:
            top_scores, top_idx = torch.topk(scores_ct, k=topC, dim=-1)  # [B, Tq, topC]

        # Decide rescue per (b,t) by probe variance among shortlisted clusters
        rescue_mask_bt, rescue_clusters_bt = self._decide_rescue_with_probes(Q_bhtd, top_idx)

        # If representatives-only or K/V absent => always use reps
        # *** FIX: Pass current K/V to reps_only ***
        if cfg.use_representatives_only or (layer.K is None) or (layer.V is None) or (K_bhsd is None) or (V_bhsd is None):
            return self._attend_reps_only(Q_bhtd, K_cur_bhsd, V_cur_bhsd, top_idx)

        # Otherwise, mix token rescue for high-variance clusters
        # *** FIX: Pass current K/V to mixed_with_rescue ***
        return self._attend_mixed_with_rescue(
            Q_bhtd, K_bhsd, V_bhsd, K_cur_bhsd, V_cur_bhsd,
            attn_mask, top_idx, rescue_mask_bt, rescue_clusters_bt
        )

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
        """
        Representatives-only attention (memory-saving), NOW including current token.
        """
        layer = self.cache
        cfg = self.cfg
        B, Hq, Tq, D = Q_bhtd.shape
        Hkv = layer.repsK.shape[1]
        C = layer.num_clusters()

        repsK = layer.repsK  # [C,Hkv,D]
        repsV = layer.repsV  # [C,Hkv,D]
        log_sizes = layer.log_sizes.view(1, 1, C)  # [1,1,C]

        scale = cfg.scale_override if cfg.scale_override is not None else (1.0 / math.sqrt(D))
        out = torch.empty(B, Hq, Tq, D, device=Q_bhtd.device, dtype=Q_bhtd.dtype)
        
        has_reps = top_idx is not None
        has_curr = K_cur_bhsd is not None and V_cur_bhsd is not None

        for hq in range(Hq):
            kvh = _kv_head_for(hq, Hkv, cfg.gqa_map)
            
            all_logits = []
            all_V_banks = []
            
            # 1. Past Representatives
            if has_reps:
                K_bank = repsK[:, kvh, :]  # [C,D]
                V_bank = repsV[:, kvh, :]  # [C,D]
                
                rep_logits = torch.matmul(Q_bhtd[:, hq], K_bank.t())  # [B,Tq,C]
                rep_logits = _maybe_scale_logits(rep_logits, scale)
                if cfg.mass_alpha != 0.0:
                    rep_logits = rep_logits + cfg.mass_alpha * log_sizes.to(rep_logits)

                Bsz, Tlen, _ = rep_logits.shape
                top_rep_logits = torch.gather(rep_logits, dim=-1, index=top_idx) # [B,Tq,topC]
                
                V_sel = V_bank.index_select(0, top_idx.view(-1)).view(Bsz, Tlen, top_idx.shape[-1], D)  # [B,Tq,topC,D]
                
                all_logits.append(top_rep_logits)
                all_V_banks.append(V_sel)

            # 2. Current Token
            if has_curr:
                K_cur_h = K_cur_bhsd[:, kvh, :, :] # [B, 1, D]
                V_cur_h = V_cur_bhsd[:, kvh, :, :] # [B, 1, D]

                # Q is [B,Tq,D], K_cur_h is [B,1,D] -> [B,Tq,1]
                cur_logits = torch.matmul(Q_bhtd[:, hq], K_cur_h.transpose(-1, -2))
                cur_logits = _maybe_scale_logits(cur_logits, scale)
                
                V_cur_btd = V_cur_h.unsqueeze(1).expand(-1, Tq, -1, -1) # [B,Tq,1,D]
                
                all_logits.append(cur_logits)
                all_V_banks.append(V_cur_btd)

            # 3. Combine and Softmax
            if not all_logits:
                out[:, hq] = torch.zeros((B, Tq, D), device=out.device, dtype=out.dtype)
                continue

            # Concat logits [B, Tq, topC + 1] and V banks [B, Tq, topC + 1, D]
            final_logits = torch.cat(all_logits, dim=-1)
            final_V = torch.cat(all_V_banks, dim=2)

            attn = F.softmax(final_logits.to(torch.float32), dim=-1).to(Q_bhtd.dtype)
            out[:, hq] = torch.einsum("btc,btcd->btd", attn, final_V)
            
        return out

    def _attend_mixed_with_rescue(
        self,
        Q_bhtd: torch.Tensor,
        K_bhsd: torch.Tensor,
        V_bhsd: torch.Tensor,
        K_cur_bhsd: Optional[torch.Tensor], # NEW
        V_cur_bhsd: Optional[torch.Tensor], # NEW
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

        # Precompute membership lists per cluster (indices are 0..S_eff-1)
        labels_cpu = layer.labels.detach().to("cpu")
        members_by_c: Dict[int, torch.Tensor] = {}
        for c in range(C):
            members_by_c[c] = torch.nonzero(labels_cpu == c, as_tuple=False).flatten()

        # probe_idx_list = getattr(layer, "probe_idx", None) # No longer needed for this logic

        for b in range(B):
            for t in range(Tq):
                rescue_cs = set(rescue_clusters_bt[b][t]) if bool(rescue_mask_bt[b, t].item()) else set()

                # (1) Representative clusters = shortlisted minus rescued
                rep_clusters = [int(top_idx[b, t, k].item()) for k in range(top_idx.shape[-1]) if int(top_idx[b, t, k].item()) not in rescue_cs]
                rep_clusters_tensor = torch.tensor(rep_clusters, device=Q_bhtd.device, dtype=torch.long) if len(rep_clusters) else None

                # (2) Token shortlist for rescued clusters
                token_indices: List[int] = []
                if len(rescue_cs) > 0:
                    for c in rescue_cs:
                        # *** FIX: Expand to ALL members as per proposal, not just 'rescue_tokens_per_cluster' ***
                        all_members_c = members_by_c[c].tolist()
                        token_indices.extend([int(idx) for idx in all_members_c])

                    # de-duplicate while preserving order
                    seen = set()
                    token_indices = [i for i in token_indices if (i not in seen and not seen.add(i))]

                token_idx_tensor = torch.tensor(token_indices, device=Q_bhtd.device, dtype=torch.long) if len(token_indices) else None

                for hq in range(Hq):
                    kvh = _kv_head_for(hq, Hkv, cfg.gqa_map)
                    q = Q_bhtd[b, hq, t:t+1, :]  # [1,1,D]

                    all_logits_list = []
                    all_V_list = []

                    # (A) Representatives part (Past)
                    if rep_clusters_tensor is not None and rep_clusters_tensor.numel() > 0:
                        K_rep = repsK.index_select(0, rep_clusters_tensor)[:, kvh, :]  # [Cr,D]
                        V_rep = repsV.index_select(0, rep_clusters_tensor)[:, kvh, :]  # [Cr,D]
                        rep_logits = torch.matmul(q, K_rep.t()).squeeze(0)              # [Cr]
                        rep_logits = _maybe_scale_logits(rep_logits, scale)
                        rep_logits = rep_logits + cfg.mass_alpha * log_sizes.index_select(0, rep_clusters_tensor).to(rep_logits)
                        
                        all_logits_list.append(rep_logits)
                        all_V_list.append(V_rep)

                    # (B) Token part (Rescued Past)
                    if token_idx_tensor is not None and token_idx_tensor.numel() > 0:
                        K_tok = K_bhsd[b, kvh, token_idx_tensor, :]  # [Nt,D]
                        V_tok = V_bhsd[b, kvh, token_idx_tensor, :]  # [Nt,D]
                        tok_logits = torch.matmul(q, K_tok.t()).squeeze(0)  # [Nt]
                        tok_logits = _maybe_scale_logits(tok_logits, scale)
                        if attn_mask is not None:
                            mask_slice = attn_mask[b, 0, t, token_idx_tensor]
                            tok_logits = _apply_mask_logits(tok_logits, mask_slice)
                        
                        all_logits_list.append(tok_logits)
                        all_V_list.append(V_tok)

                    # (C) Current Token part (Present)
                    has_curr = K_cur_bhsd is not None and V_cur_bhsd is not None
                    if has_curr:
                        # K_cur_bhsd is [B, Hkv, 1, D]
                        K_cur_h = K_cur_bhsd[b, kvh] # [1, D]
                        V_cur_h = V_cur_bhsd[b, kvh] # [1, D]
                        
                        # q is [1,1,D], K_cur_h is [1,D]
                        cur_logits = torch.matmul(q.squeeze(0), K_cur_h.t()) # [1, 1]
                        cur_logits = _maybe_scale_logits(cur_logits, scale).squeeze(-1) # [1]
                        
                        all_logits_list.append(cur_logits)
                        all_V_list.append(V_cur_h)


                    if not all_logits_list:
                        out[b, hq, t] = torch.zeros(D, device=q.device, dtype=q.dtype)
                        continue

                    # Mix all parts
                    all_logits = torch.cat(all_logits_list, dim=-1).unsqueeze(0)  # [1, Cr+Nt+1]
                    all_V = torch.cat(all_V_list, dim=0).unsqueeze(0)              # [1, Cr+Nt+1, D]
                    attn = F.softmax(all_logits.to(torch.float32), dim=-1).to(q.dtype)
                    out[b, hq, t:t+1] = torch.matmul(attn, all_V)

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
        return patch(
            Q_bhtd=Q_bhtd,
            K_bhsd=K_bhsd,
            V_bhsd=V_bhsd,
            position_ids=position_ids,
            attn_mask=attention_mask,
        )

    return patched_forward
