from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from keygraph.method.keygraph_core import (
    build_descriptors_unrope,    # [H,S,D] + pos -> Phi[S,r], rp_matrix[D,r]
    build_knn_and_clusters,      # Phi -> {labels, ...} (ANN/exact + CC)
    aggregate_reps_from_labels,  # (K,V,labels) -> repsK[C,H,D], repsV[C,H,D], sizes[C]
)

__all__ = ["KeygraphCacheConfig", "LayerKeygraphCache", "KeygraphCache"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class KeygraphCacheConfig:
    # Descriptor / graph
    rp_dim: int = 64                     # random projection dimension r
    tau: float = 0.25                    # cosine threshold for edges
    ann_k: int = 32                      # neighbors per node during build
    mutual: bool = True                  # use mutual-kNN (else OR)
    ann_backend: str = "faiss"           # "faiss" | "torch_ivf" | "exact"

    # Probe sets
    probe_per_cluster: int = 16          # how many probe members to keep per cluster
    probe_on_cpu: bool = True            # store probes (idx & desc) on CPU

    # Online assignment during append_tokens
    tokens_assign_tau: float = 0.10      # min centroid-cosine to reuse an existing cluster
    assign_mode: str = "centroid"        # "centroid" | "probe_ann" (centroid is default)

    # Housekeeping
    keep_descriptors: bool = False       # store full Phi[S,r] in layer (debug/analysis)
    store_full_kv: bool = False          # keep full K/V (debug); False for memory saving
    dtype: torch.dtype = torch.float16
    device: Optional[torch.device] = None


# ---------------------------------------------------------------------------
# Layer container
# ---------------------------------------------------------------------------

@dataclass
class LayerKeygraphCache:
    """
    Per-layer cache with mean representatives (K*, V*), descriptor centroids, and metadata.
    Shapes follow LLaMA-style: K/V given to us as [H, S, D] for a single layer.
    """
    # Mandatory
    labels: torch.Tensor            # [S] int64, cluster id per token (0..C-1)
    repsK: torch.Tensor             # [C, H, D] mean key per cluster/head (same dtype as inputs)
    repsV: torch.Tensor             # [C, H, D] mean value per cluster/head (same dtype as inputs)
    log_sizes: torch.Tensor         # [C] log(|C|)
    repsPhi: torch.Tensor           # [C, r] unit-norm centroid in descriptor space (fp32)
    rp_matrix: torch.Tensor         # [D, r] fp32, column-normalized RP used to build Phi
    rope_base: float                # base used for RoPE (needed by the patch)
    pos_idx: torch.Tensor           # [S] absolute positions for tokens (0..S-1)

    # Optional descriptors (helpful for analyses / probe refresh)
    descriptors: Optional[torch.Tensor]  # [S, r] fp32 or None

    # Probe sets per cluster
    probe_sets: Optional[List[torch.Tensor]] = None   # len=C; int64 indices (CPU by default)
    probe_desc: Optional[List[torch.Tensor]] = None   # len=C; [|P[c]|, r] fp32 (CPU by default)

    # Optional: keep full K/V (debug/inspection; not needed for inference)
    K: Optional[torch.Tensor] = None      # [H, S, D]
    V: Optional[torch.Tensor] = None      # [H, S, D]

    # Config (for convenience)
    cfg: Optional[KeygraphCacheConfig] = None

    # ---- small helpers ----
    def num_tokens(self) -> int:
        return int(self.labels.numel())

    def num_clusters(self) -> int:
        return int(self.repsPhi.shape[0])

    def to(self, device=None, dtype: Optional[torch.dtype] = None) -> "LayerKeygraphCache":
        """
        Move tensors to a device / dtype. Probe sets default to CPU; keep them there
        unless the user explicitly wants to move them.
        """
        device = device or (self.cfg.device if self.cfg and self.cfg.device is not None else None)
        # Scalar / lists
        if device is not None:
            self.labels = self.labels.to(device)
            self.repsK = self.repsK.to(device=device, dtype=dtype or self.repsK.dtype)
            self.repsV = self.repsV.to(device=device, dtype=dtype or self.repsV.dtype)
            self.log_sizes = self.log_sizes.to(device)
            self.repsPhi = self.repsPhi.to(device)   # keep fp32
            self.rp_matrix = self.rp_matrix.to(device)  # fp32
            self.pos_idx = self.pos_idx.to(device)
            if self.descriptors is not None:
                self.descriptors = self.descriptors.to(device)
            if self.K is not None:
                self.K = self.K.to(device=device, dtype=dtype or self.K.dtype)
            if self.V is not None:
                self.V = self.V.to(device=device, dtype=dtype or self.V.dtype)
        return self


# ---------------------------------------------------------------------------
# Builder / updater
# ---------------------------------------------------------------------------

class KeygraphCache:
    """
    Build and maintain per-layer KeyGraph caches.
    Usage (prefill):
        cache = KeygraphCache(cfg)
        layer = cache.build_layer(K[H,S,D], V[H,S,D], pos_idx[S], rope_base, rp_matrix=None)

    Usage (decode append):
        layer = cache.append_tokens(layer, K_new[H,s_new,D], V_new[H,s_new,D], pos_new[s_new])
    """

    def __init__(self, cfg: KeygraphCacheConfig):
        self.cfg = cfg

    # -------------------------- public API --------------------------

    @torch.no_grad()
    def build_layer(
        self,
        K: torch.Tensor,                 # [H, S, D]
        V: torch.Tensor,                 # [H, S, D]
        pos_idx: torch.Tensor,           # [S]
        rope_base: float,
        rp_matrix: Optional[torch.Tensor] = None,  # [D, r]
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> LayerKeygraphCache:
        """
        One-shot build from prefill keys/values. Produces cluster labels, representatives,
        centroid descriptors, probe sets, and stores the RP used.
        """
        self._validate_kv(K, V)
        H, S, D = K.shape
        assert pos_idx.shape == (S,), f"pos_idx must be [S], got {tuple(pos_idx.shape)}"

        device = device or (self.cfg.device if self.cfg.device is not None else K.device)
        dtype = dtype or self.cfg.dtype

        Phi, rp_matrix = build_descriptors_unrope(
            keys_per_head=K, pos_idx=pos_idx,
            rp_matrix=rp_matrix, r=int(self.cfg.rp_dim),
            base=float(rope_base),
        )  

        # 2) kNN + CC
        ann = self._make_ann_spec(self.cfg.ann_backend)
        knn = build_knn_and_clusters(
            phi=Phi,
            tau=float(self.cfg.tau),
            k=int(min(self.cfg.ann_k, max(0, S - 1))),
            mutual=bool(self.cfg.mutual),
            ann=ann,
        )
        labels = knn["labels"].to(torch.long)  # [S]
        assert labels.numel() == S

        # 3) Representatives (MEANS) + sizes
        repsK, repsV, sizes = aggregate_reps_from_labels(K, V, labels)
        # Reps are [H,C,D] from core; transpose to [C,H,D] for cache ergonomics
        repsK = repsK.permute(1, 0, 2).contiguous()
        repsV = repsV.permute(1, 0, 2).contiguous()
        log_sizes = sizes.to(torch.float32).clamp_min_(1).log_()  # [C]

        # 4) Centroid descriptors repsPhi[C,r]
        repsPhi = self._centroids_from_labels(Phi, labels)  # fp32, normalized

        # 5) Probes (indices & desc) - CPU by default
        probe_sets, probe_desc = self._make_probe_sets(Phi, labels, repsPhi)

        # 6) Pack the layer
        layer = LayerKeygraphCache(
            labels=labels.to(device),
            repsK=repsK.to(device=device, dtype=dtype),
            repsV=repsV.to(device=device, dtype=dtype),
            log_sizes=log_sizes.to(device),
            repsPhi=repsPhi.to(device),
            rp_matrix=rp_matrix.to(device),
            rope_base=float(rope_base),
            pos_idx=pos_idx.to(device),
            descriptors=Phi.to(device) if self.cfg.keep_descriptors else None,
            probe_sets=probe_sets,
            probe_desc=probe_desc,
            K=K if self.cfg.store_full_kv else None,
            V=V if self.cfg.store_full_kv else None,
            cfg=self.cfg,
        )
        return layer

    @torch.no_grad()
    def append_tokens(
        self,
        layer: LayerKeygraphCache,
        K_new: torch.Tensor,           # [H, s, D]
        V_new: torch.Tensor,           # [H, s, D]
        pos_new: torch.Tensor,         # [s]
        assign_mode: Optional[str] = None,  
    ) -> LayerKeygraphCache:
        """
        Online update: assign new tokens to clusters; update means and centroid descriptors.
        Keeps probe sets fresh via simple replacement policy (optional).
        """
        self._validate_kv(K_new, V_new)
        cfg = layer.cfg or self.cfg
        assign_mode = assign_mode or cfg.assign_mode
        H, s_new, D = K_new.shape
        assert pos_new.shape == (s_new,)

        device = layer.repsK.device

        # 1) Build descriptors for the NEW tokens only (fp32, same RP)
        Phi_new, _ = build_descriptors_unrope(
            keys_per_head=K_new,
            pos_idx=pos_new,
            rp_matrix=layer.rp_matrix,   # reuse same projection
            r=int(cfg.rp_dim),
            base=float(layer.rope_base),
        )  # [s, r] fp32

        # 2) Assign each new token to an existing cluster or create a new one
        if layer.num_clusters() > 0:
            if assign_mode == "centroid":
                assign, create_mask = self._assign_via_centroids(layer, Phi_new)
            elif assign_mode == "probe_ann":
                assign, create_mask = self._assign_via_probes(layer, Phi_new)
            else:
                raise ValueError(f"Unknown assign_mode: {assign_mode}")
        else:
            # first tokens bootstrap C new singleton clusters
            assign = torch.arange(s_new, device=Phi_new.device, dtype=torch.long)
            create_mask = torch.ones(s_new, dtype=torch.bool, device=Phi_new.device)

        # Create ids for the 'new cluster' tokens, if any
        C_old = layer.num_clusters()
        if create_mask.any():
            n_create = int(create_mask.sum().item())
            new_ids = torch.arange(C_old, C_old + n_create, device=assign.device, dtype=torch.long)
            assign[create_mask] = new_ids
            # Expand layer tensors to C_new
            layer = self._expand_layer_for_new_clusters(layer, n_create)

        # 3) Append labels for these s tokens
        layer.labels = torch.cat([layer.labels, assign.to(layer.labels.device)], dim=0)

        # 4) Update counts and log_sizes
        C = layer.num_clusters()
        counts_new = torch.zeros((C,), dtype=torch.float32, device=device)
        counts_new.index_add_(0, assign, torch.ones_like(assign, dtype=torch.float32, device=device))
        n_old = layer.log_sizes.exp()      # [C]
        n_tot = n_old + counts_new
        layer.log_sizes = n_tot.clamp_min_(1).log_()

        # 5) Update repsK/repsV running means
        # compute sum over new tokens per cluster/head via 2D index_add
        K2d = K_new.permute(1, 0, 2).reshape(s_new, -1)   # [s, H*D]
        V2d = V_new.permute(1, 0, 2).reshape(s_new, -1)   # [s, H*D]
        sumK2d = torch.zeros((C, H * D), device=device, dtype=layer.repsK.dtype)
        sumV2d = torch.zeros((C, H * D), device=device, dtype=layer.repsV.dtype)
        sumK2d.index_add_(0, assign, K2d.to(sumK2d.dtype))
        sumV2d.index_add_(0, assign, V2d.to(sumV2d.dtype))
        sumK = sumK2d.view(C, H, D)
        sumV = sumV2d.view(C, H, D)

        # new_mean = (n_old*old_mean + sum_new) / n_tot
        n_old_resh = n_old.view(C, 1, 1).to(dtype=layer.repsK.dtype)
        n_tot_resh = n_tot.clamp_min_(1).view(C, 1, 1).to(dtype=layer.repsK.dtype)
        layer.repsK = (n_old_resh * layer.repsK + sumK) / n_tot_resh
        layer.repsV = (n_old_resh * layer.repsV + sumV) / n_tot_resh

        # 6) Update centroid descriptors repsPhi (fp32, normalized)
        repsPhi_new = self._update_centroids_running(layer.repsPhi, layer.log_sizes, Phi_new, assign)
        layer.repsPhi = repsPhi_new

        # 7) Update pos_idx (append)
        layer.pos_idx = torch.cat([layer.pos_idx, pos_new.to(layer.pos_idx.device)], dim=0)

        # 8) (Optional) keep full K/V and/or descriptors
        if layer.K is not None:
            layer.K = torch.cat([layer.K, K_new.to(layer.K)], dim=1)
        if layer.V is not None:
            layer.V = torch.cat([layer.V, V_new.to(layer.V)], dim=1)
        if layer.descriptors is not None:  # keep full Φ
            layer.descriptors = torch.cat([layer.descriptors, Phi_new.to(layer.descriptors)], dim=0)

        # 9) Maintain probe sets (simple replacement: keep top-cos vs centroid)
        if layer.probe_sets is not None:
            self._online_probe_refresh(layer, Phi_new, assign)

        return layer

    # -------------------------- internals --------------------------

    @staticmethod
    def _validate_kv(K: torch.Tensor, V: torch.Tensor):
        assert K.ndim == 3 and V.ndim == 3, f"K/V must be [H,S,D], got {tuple(K.shape)} / {tuple(V.shape)}"
        assert K.shape == V.shape, f"K and V shapes must match; got {tuple(K.shape)} vs {tuple(V.shape)}"

    @staticmethod
    def _make_ann_spec(backend: str):
        backend = (backend or "exact").lower()
        if backend == "faiss":
            return {"method": "faiss_ivf_flat", "params": {"nlist": 128, "nprobe": 8}}
        if backend == "torch_ivf":
            return {"method": "torch_ivf", "params": {"nlist": 128, "nprobe": 8}}
        if backend == "exact":
            return None
        raise ValueError(f"Unknown ann_backend: {backend}")

    @staticmethod
    def _centroids_from_labels(Phi: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Phi: [S,r] (fp32). Return normalized centroid per cluster: [C,r].
        Phi_n = F.normalize(Phi.to(torch.float32), dim=1, eps=1e-6)
        C = int(labels.max().item()) + 1 if labels.numel() else 0
        sums = torch.zeros((C, Phi_n.shape[1]), dtype=torch.float32, device=Phi_n.device)
        cnts = torch.zeros((C,), dtype=torch.float32, device=Phi_n.device)
        sums.index_add_(0, labels, Phi_n)
        cnts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
        cents = sums / cnts.clamp_min_(1).unsqueeze(1)
        return F.normalize(cents, dim=1, eps=1e-6)

    def _make_probe_sets(
        self, Phi: torch.Tensor, labels: torch.Tensor, repsPhi: torch.Tensor
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        """
        Choose up to probe_per_cluster members with highest centroid-cosine.
        Store indices (int64) and descriptors (fp32, normalized). By default: on CPU.
        """
        if self.cfg.probe_per_cluster <= 0:
            return None, None

        device_idx = torch.device("cpu") if self.cfg.probe_on_cpu else labels.device
        device_desc = torch.device("cpu") if self.cfg.probe_on_cpu else Phi.device

        Phi_n = F.normalize(Phi.to(torch.float32), dim=1, eps=1e-6)
        C = repsPhi.shape[0]
        probes_idx: List[torch.Tensor] = []
        probes_desc: List[torch.Tensor] = []
        for c in range(C):
            idx_c = torch.nonzero(labels == c, as_tuple=False).flatten()
            if idx_c.numel() == 0:
                probes_idx.append(torch.empty(0, dtype=torch.long, device=device_idx))
                probes_desc.append(torch.empty(0, Phi_n.shape[1], dtype=torch.float32, device=device_desc))
                continue
            v = Phi_n.index_select(0, idx_c)              # [n_c, r]
            scores = v @ repsPhi[c]                       # [n_c]
            k = min(self.cfg.probe_per_cluster, v.shape[0])
            topk = torch.topk(scores, k=k, largest=True).indices
            idx_sel = idx_c.index_select(0, topk)
            desc_sel = v.index_select(0, topk)
            probes_idx.append(idx_sel.to(device_idx, non_blocking=True))
            probes_desc.append(desc_sel.to(device_desc, non_blocking=True))
        return probes_idx, probes_desc

    def _assign_via_centroids(
        self, layer: LayerKeygraphCache, Phi_new: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign to the nearest centroid in descriptor space.
        Returns (assign[C_id per token], create_mask[bool per token]).
        """
        Phi_n = F.normalize(Phi_new.to(torch.float32), dim=1, eps=1e-6)           # [s, r]
        C = layer.num_clusters()
        Phi_c = F.normalize(layer.repsPhi.to(Phi_n), dim=1, eps=1e-6)             # [C, r]
        sims = Phi_n @ Phi_c.t()                                                  # [s, C]
        max_sim, argmax = sims.max(dim=1)
        threshold = float((layer.cfg or self.cfg).tokens_assign_tau)
        create_mask = max_sim < threshold
        return argmax.to(torch.long), create_mask

    def _assign_via_probes(
        self, layer: LayerKeygraphCache, Phi_new: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        (Optional) Assign by scoring against concatenated probe descriptors.
        If no probes exist, falls back to centroid assignment.
        """
        if not layer.probe_desc or sum(x.numel() for x in layer.probe_desc) == 0:
            return self._assign_via_centroids(layer, Phi_new)

        # Stack probe descriptors to [P, r] and keep a book-keeping array of cluster ids
        probe_desc = [p.to(Phi_new.device) for p in layer.probe_desc]
        cluster_ids = []
        desc_cat = []
        for c, d in enumerate(probe_desc):
            if d.numel():
                desc_cat.append(d)
                cluster_ids.append(torch.full((d.shape[0],), c, dtype=torch.long, device=Phi_new.device))
        desc_cat = torch.cat(desc_cat, dim=0)  # [P, r]
        cluster_ids = torch.cat(cluster_ids, dim=0)  # [P]

        # Score new descriptors against probes
        Phi_n = F.normalize(Phi_new.to(torch.float32), dim=1, eps=1e-6)  # [s, r]
        desc_n = F.normalize(desc_cat, dim=1, eps=1e-6)                  # [P, r]
        scores = Phi_n @ desc_n.t()                                      # [s, P]
        best_probe = scores.argmax(dim=1)                                # [s]
        assign = cluster_ids.index_select(0, best_probe)                 # [s]

        # Create if too far from the assigned centroid (safety)
        _, create_mask = self._assign_via_centroids(layer, Phi_new)
        return assign, create_mask

    @staticmethod
    def _expand_layer_for_new_clusters(layer: LayerKeygraphCache, n_create: int) -> LayerKeygraphCache:
        """
        Append new empty rows for repsK/repsV/repsPhi/log_sizes and (optionally) probe lists.
        """
        if n_create <= 0:
            return layer
        C_old, H, D = layer.repsK.shape
        C_new = C_old + n_create
        device = layer.repsK.device

        zeros_rep = torch.zeros((n_create, H, D), dtype=layer.repsK.dtype, device=device)
        layer.repsK = torch.cat([layer.repsK, zeros_rep.clone()], dim=0)
        layer.repsV = torch.cat([layer.repsV, zeros_rep.clone()], dim=0)

        r = layer.repsPhi.shape[1]
        layer.repsPhi = torch.cat(
            [layer.repsPhi, F.normalize(torch.randn(n_create, r, device=device), dim=1, eps=1e-6)],
            dim=0,
        )
        layer.log_sizes = torch.cat([layer.log_sizes,
                             torch.full((n_create,), float("-inf"), device=device)], dim=0)

        if layer.probe_sets is not None:
            for _ in range(n_create):
                layer.probe_sets.append(torch.empty(0, dtype=torch.long, device=torch.device("cpu")))
                layer.probe_desc.append(torch.empty(0, r, dtype=torch.float32, device=torch.device("cpu")))
        return layer

    @staticmethod
    def _update_centroids_running(
        repsPhi: torch.Tensor, log_sizes: torch.Tensor, Phi_new: torch.Tensor, assign: torch.Tensor
    ) -> torch.Tensor:
        """
        Running mean update in descriptor space. Keep fp32 and renormalize.
        """
        C, r = repsPhi.shape
        Phi_n = F.normalize(Phi_new.to(torch.float32), dim=1, eps=1e-6)  # [s, r]

        # Sum per cluster
        sums = torch.zeros((C, r), dtype=torch.float32, device=repsPhi.device)
        sums.index_add_(0, assign.to(repsPhi.device), Phi_n.to(repsPhi.device))

        n_old = log_sizes.exp()                    # [C] on same device as repsPhi/log_sizes
        n_tot = n_old + torch.bincount(assign.to(n_old.device), minlength=C).to(n_old)
        cents = (n_old.unsqueeze(1) * repsPhi + sums) / n_tot.clamp_min_(1).unsqueeze(1)
        return F.normalize(cents, dim=1, eps=1e-6)

    def _online_probe_refresh(self, layer: LayerKeygraphCache, Phi_new: torch.Tensor, assign: torch.Tensor):
        """
        Lightweight maintenance: for each new token, possibly promote it into the cluster's probe set
        if it scores higher against the centroid than the current worst probe.
        """
        if layer.probe_sets is None:
            return
        # Normalize once
        Phi_n = F.normalize(Phi_new.to(torch.float32), dim=1, eps=1e-6)

        # Iterate new tokens (s is tiny at decode)
        for i in range(Phi_n.shape[0]):
            c = int(assign[i].item())
            # Ensure lists exist (CPU by default)
            if layer.probe_sets[c].device.type != "cpu":
                # keep indices on CPU
                layer.probe_sets[c] = layer.probe_sets[c].cpu()
            if layer.probe_desc[c].device.type != "cpu":
                layer.probe_desc[c] = layer.probe_desc[c].cpu()

            cand_desc = Phi_n[i:i+1].cpu()  # [1, r]
            cand_score = float(cand_desc @ layer.repsPhi[c].cpu())

            # If we have fewer than target probes, just append
            if layer.probe_sets[c].numel() < (layer.cfg or self.cfg).probe_per_cluster:
                # global index of this token = current total - remaining new + i
                abs_idx = layer.num_tokens() - (Phi_n.shape[0] - i)
                layer.probe_sets[c] = torch.cat(
                    [layer.probe_sets[c], torch.tensor([abs_idx], dtype=torch.long)]
                )
                layer.probe_desc[c] = torch.cat([layer.probe_desc[c], cand_desc], dim=0)
                continue

            # Otherwise, replace worst if candidate is better
            if layer.probe_desc[c].numel() == 0:
                continue
            scores = (layer.probe_desc[c] @ layer.repsPhi[c].cpu())
            worst_score, worst_idx = torch.min(scores, dim=0)
            if cand_score > float(worst_score):
                abs_idx = layer.num_tokens() - (Phi_n.shape[0] - i)
                # replace entry worst_idx
                idx_list = layer.probe_sets[c]
                desc_list = layer.probe_desc[c]
                # (fix from previous bug: DO NOT clobber desc_list)
                idx_list[worst_idx] = abs_idx
                desc_list[worst_idx:worst_idx+1] = cand_desc
                layer.probe_sets[c] = idx_list
                layer.probe_desc[c] = desc_list
        return
