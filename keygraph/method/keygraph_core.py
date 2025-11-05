from __future__ import annotations
import math
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn.functional as F

# ---------- Optional FAISS (GPU preferred, CPU fallback) ----------
_FAISS_OK = False
_FAISS_GPU_OK = False
try:
    import faiss  # type: ignore
    _FAISS_OK = True
    try:
        import faiss.contrib.torch_utils  # noqa: F401
    except Exception:
        pass
    try:
        _ = faiss.StandardGpuResources()  # will fail if no GPU faiss
        _FAISS_GPU_OK = True
    except Exception:
        _FAISS_GPU_OK = False
except Exception:
    _FAISS_OK = False
    _FAISS_GPU_OK = False


# ============================================================================
# 1)  Descriptor builder (UnRoPE -> per-head normalization -> mean across heads -> RP -> ℓ2)
# ============================================================================

def build_descriptors_unrope(keys_per_head: torch.Tensor,
                             pos_idx: torch.Tensor,
                             rp_matrix: Optional[torch.Tensor] = None,
                             r: int = 32,
                             base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    keys_per_head: [H, S, D] (RoPE-applied keys)
    pos_idx:       [S]       absolute positions (0..S-1)
    rp_matrix:     [D, r]    optional fixed RP (fp32, column-normalized)
    r:             projection dim
    base:          RoPE base

    Returns:
      phi:       [S, r]  (fp32, unit-norm)
      rp_matrix: [D, r]  (fp32)
    """
    assert keys_per_head.dim() == 3, "keys_per_head must be [H,S,D]"
    H, S, D = keys_per_head.shape
    assert D % 2 == 0, "head_dim must be even for RoPE"

    device = keys_per_head.device

    k = keys_per_head.to(torch.float32)        # [H,S,D]
    x1 = k[..., 0::2]                          # [H,S,D/2]
    x2 = k[..., 1::2]                          # [H,S,D/2]

    # RoPE angles
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, device=device, dtype=torch.float32) / D))  # [D/2]
    angles   = torch.einsum("s,j->sj", pos_idx.to(device=device, dtype=torch.float32), inv_freq) # [S,D/2]
    cos = angles.cos().unsqueeze(0)            # [1,S,D/2]
    sin = angles.sin().unsqueeze(0)            # [1,S,D/2]

    # Inverse rotation
    u1 = x1 * cos + x2 * sin
    u2 = x2 * cos - x1 * sin

    unrot = torch.empty((H, S, D), device=device, dtype=torch.float32)
    unrot[..., 0::2] = u1
    unrot[..., 1::2] = u2

    # Head-invariant mean
    norms = unrot.norm(dim=-1, keepdim=True).clamp_min_(1e-6)
    mean_keys = (unrot / norms).mean(dim=0)    # [S,D], fp32

    # Random projection
    if rp_matrix is None:
        rp_matrix = torch.randn(D, r, device=device, dtype=torch.float32)
        rp_matrix = F.normalize(rp_matrix, p=2, dim=0)  # column-normalize
    else:
        if rp_matrix.dtype != torch.float32:
            rp_matrix = rp_matrix.to(torch.float32)

    phi = mean_keys @ rp_matrix                 # [S,r], fp32
    phi = F.normalize(phi, dim=1, eps=1e-6)
    return phi, rp_matrix


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_n = F.normalize(a, p=2, dim=-1)
    b_n = F.normalize(b, p=2, dim=-1)
    return a_n @ b_n.transpose(-2, -1)


# ---- NEW: UnRoPE keys helper (mirror of _unrope_queries) ----
def unrope_keys(K_hsd: torch.Tensor, pos_idx: torch.Tensor, rope_base: float) -> torch.Tensor:
    """
    K_hsd:   [H, S, D] RoPE-applied keys (per head)
    pos_idx: [S] absolute positions for those keys
    Return:  [H, S, D] un-rotated keys (float32)
    """
    assert K_hsd.dim() == 3 and K_hsd.shape[2] % 2 == 0
    H, S, D = K_hsd.shape
    device = K_hsd.device
    dtype  = torch.float32

    x = K_hsd.to(dtype)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    half = D // 2
    idx  = torch.arange(0, half, device=device, dtype=torch.float32)
    invf = rope_base ** (-2.0 * idx / float(D))                       # [half]
    pos  = pos_idx.to(device=device, dtype=torch.float32)             # [S]
    ang  = torch.einsum("s,d->sd", pos, invf)                         # [S,half]
    cos  = ang.cos().unsqueeze(0)                                     # [1,S,half]
    sin  = ang.sin().unsqueeze(0)

    # Inverse rotation for keys (same as queries’ inverse)
    u1 = x1 * cos + x2 * sin
    u2 = x2 * cos - x1 * sin

    out = torch.empty((H, S, D), device=device, dtype=dtype)
    out[..., 0::2] = u1
    out[..., 1::2] = u2
    return out


# ============================================================================
# 2)  kNN (FAISS-GPU/CPU) + τ sparsification + mutual/OR + sparse CC via union-find
# ============================================================================

class _UF:
    """Tiny union-find on CPU for sparse connected components."""
    __slots__ = ("parent", "rank")
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]
    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
    def labels(self) -> torch.Tensor:
        for i in range(len(self.parent)):
            self.parent[i] = self.find(i)
        # reindex to 0..C-1
        uniq = {}
        cur = 0
        out = torch.empty(len(self.parent), dtype=torch.long)
        for i, p in enumerate(self.parent):
            if p not in uniq:
                uniq[p] = cur; cur += 1
            out[i] = uniq[p]
        return out


@torch.no_grad()
def _faiss_gpu_flat_knn(phi_u: torch.Tensor, k: int, gpu_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    FAISS-GPU FlatIP kNN on unit-norm phi_u [N,r] (fp32). Returns (idx, sim) as CUDA tensors.
    """
    assert _FAISS_OK and _FAISS_GPU_OK, "FAISS-GPU not available"
    N, r = phi_u.shape
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatIP(res, r, faiss.GpuIndexFlatConfig())  # cosine via IP on unit-norm
    # With torch_utils imported, FAISS accepts torch tensors directly.
    index.add(phi_u)                          # database
    sims, inds = index.search(phi_u, k + 1)   # include self
    inds = torch.as_tensor(inds, device=phi_u.device, dtype=torch.long)
    sims = torch.as_tensor(sims, device=phi_u.device, dtype=torch.float32)
    # Drop the self column (assumed first)
    inds = inds[:, 1:(k + 1)]
    sims = sims[:, 1:(k + 1)]
    return inds, sims, "faiss_gpu_flat"


@torch.no_grad()
def _faiss_gpu_ivf_flat_knn(phi_u: torch.Tensor, k: int, nlist: int = 128, nprobe: int = 8, gpu_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    FAISS-GPU IVF-Flat with IP (cosine on unit-norm).
    Adapt nlist/nprobe for small N; fall back to Flat for tiny N to avoid slow/undertrained IVF.
    """
    assert _FAISS_OK and _FAISS_GPU_OK, "FAISS-GPU not available"
    N, r = phi_u.shape

    # ---- Small-N: Flat is faster & better-trained
    if N < 2048 or N < 40 * nlist:
        return _faiss_gpu_flat_knn(phi_u, k=int(min(k, N - 1)), gpu_id=gpu_id)

    # ---- Adapt nlist / nprobe to dataset size
    nlist_adapt = max(32, min(nlist, int(N // 32)))  # ~32 pts per list
    nprobe_adapt = max(4, min(nprobe, nlist_adapt))

    # Train CPU IVF index
    quant = faiss.IndexFlatIP(r)
    ivf = faiss.IndexIVFFlat(quant, r, int(nlist_adapt), faiss.METRIC_INNER_PRODUCT)
    phi_cpu = phi_u.detach().to("cpu").contiguous().numpy()
    ivf.train(phi_cpu)
    ivf.add(phi_cpu)

    # Move to GPU
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuClonerOptions()
    cfg.useFloat16 = True
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, ivf, cfg)
    gpu_index.nprobe = int(nprobe_adapt)

    sims, inds = gpu_index.search(phi_cpu, k + 1)
    inds = torch.from_numpy(inds).to(phi_u.device, dtype=torch.long)[:, 1:(k + 1)]
    sims = torch.from_numpy(sims).to(phi_u.device, dtype=torch.float32)[:, 1:(k + 1)]
    return inds, sims, f"faiss_gpu_ivf_flat(nlist={nlist_adapt},nprobe={nprobe_adapt})"



@torch.no_grad()
def _torch_topk_chunked(phi_u: torch.Tensor, k: int, max_chunk_mb: int = 512) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Exact kNN without SxS materialization: process queries by blocks.
    """
    N, r = phi_u.shape
    device = phi_u.device
    bytes_per = 4  # fp32
    max_bytes = max_chunk_mb * (1 << 20)
    # choose block rows M so that we don't exceed memory with (M x N x r) transient
    M = max(256, min(N, max_bytes // max(N * r * bytes_per, 1)))
    if device.type == "cuda":
        M = max(M, 1024)

    idx_all = torch.empty((N, k), dtype=torch.long, device=device)
    sim_all = torch.empty((N, k), dtype=torch.float32, device=device)

    for s in range(0, N, M):
        e = min(N, s + M)
        blk = phi_u[s:e]           # [M,r]
        sims = blk @ phi_u.t()     # [M,N]
        # mask diagonal rows
        row = torch.arange(0, e - s, device=device)
        col = torch.arange(s, e, device=device)
        sims[row, col] = -1e9

        topk = torch.topk(sims, k=k, dim=1, largest=True)
        idx_all[s:e] = topk.indices
        sim_all[s:e] = topk.values
        del sims, topk, blk
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return idx_all, sim_all, f"torch_exact_chunked_{device.type}"


@torch.no_grad()
def build_knn_and_clusters(
    phi: torch.Tensor,          # [S, r] (any dtype; will be normalized to fp32)
    tau: float = 0.8,
    k: int = 16,
    mutual: bool = True,
    ann: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    """
    kNN (FAISS-GPU preferred) + τ filtering + mutual/OR + sparse CC via union-find.
    No dense SxS adjacency is ever built.

    Returns:
      {
        "neighbors_idx": LongTensor [S, k]     (entries <tau are set to -1)
        "neighbors_sim": FloatTensor [S, k]    (entries <tau are -inf)
        "labels":        LongTensor [S]        (0..C-1)
        "meta":          dict                  (backend_used, mutual, tau, k)
      }
    """
    assert phi.dim() == 2, f"phi must be [S, r], got {phi.shape}"
    S, r = phi.shape
    device = phi.device
    if S == 0 or k <= 0:
        empty_i = torch.empty((S, 0), dtype=torch.long, device=device)
        empty_s = torch.empty((S, 0), dtype=torch.float32, device=device)
        return {"neighbors_idx": empty_i, "neighbors_sim": empty_s,
                "labels": torch.zeros((S,), dtype=torch.long, device=device),
                "meta": {"backend_used": "none", "mutual": bool(mutual), "tau": float(tau), "k": int(k)}}

    # Unit-normalize once (cosine)
    phi_u = F.normalize(phi.to(torch.float32), dim=1, eps=1e-6).contiguous()

    # Select backend
    method = (ann or {}).get("method", "faiss_gpu_flat").lower()
    params = (ann or {}).get("params", {}) or {}
    backend_used = "unknown"

    I = None; Sval = None
    try:
        if method in ("faiss_gpu_flat", "faiss_flat_gpu"):
            I, Sval, backend_used = _faiss_gpu_flat_knn(phi_u, k=int(min(k, S-1)), gpu_id=int(params.get("gpu_id", 0)))
        elif method in ("faiss_gpu_ivf_flat", "faiss_ivf_flat_gpu", "faiss_ivf_flat"):
            I, Sval, backend_used = _faiss_gpu_ivf_flat_knn(
                phi_u, k=int(min(k, S-1)),
                nlist=int(params.get("nlist", 128)), nprobe=int(params.get("nprobe", 8)),
                gpu_id=int(params.get("gpu_id", 0)),
            )
        elif method in ("faiss_cpu", "faiss"):
            # CPU FlatIP
            import faiss  # re-import to ensure present
            index = faiss.IndexFlatIP(r)
            index.add(phi_u.cpu().numpy())
            sims, inds = index.search(phi_u.cpu().numpy(), int(min(k, S-1)) + 1)
            inds = torch.from_numpy(inds[:, 1:]).to(device, dtype=torch.long)
            sims = torch.from_numpy(sims[:, 1:]).to(device, dtype=torch.float32)
            I, Sval, backend_used = inds, sims, "faiss_cpu_flat"
        else:
            # exact chunked
            I, Sval, backend_used = _torch_topk_chunked(phi_u.to(device), k=int(min(k, S-1)))
    except Exception:
        # Robust fallback: exact chunked on current device
        I, Sval, backend_used = _torch_topk_chunked(phi_u.to(device), k=int(min(k, S-1)))

    # τ filtering (keep shape; mark invalid)
    if tau is not None and tau != float("-inf"):
        valid = Sval >= float(tau)
        neg_one = torch.full_like(I, -1)
        neg_inf = torch.full_like(Sval, float("-inf"))
        I = torch.where(valid, I, neg_one)
        Sval = torch.where(valid, Sval, neg_inf)

    # Build sparse edges on CPU from neighbor lists
    I_cpu = I.detach().to("cpu")
    S_cpu = Sval.detach().to("cpu")
    # Per-node valid neighbor list
    nbr_lists: List[List[int]] = []
    for i in range(S):
        row = I_cpu[i]
        m = (row >= 0).nonzero(as_tuple=False).flatten()
        if m.numel() == 0:
            nbr_lists.append([])
        else:
            nbr_lists.append(row.index_select(0, m).tolist())

    if mutual:
        nbr_sets = [set(lst) for lst in nbr_lists]
        edges = []
        for i in range(S):
            li = nbr_lists[i]
            if not li: continue
            si = nbr_sets[i]
            for j in li:
                if i in nbr_sets[j]:
                    a, b = (i, j) if i < j else (j, i)
                    edges.append((a, b))
        # unique
        edges = list(set(edges))
    else:
        edges = []
        for i in range(S):
            for j in nbr_lists[i]:
                a, b = (i, j) if i < j else (j, i)
                edges.append((a, b))
        edges = list(set(edges))

    # Connected components via union-find (CPU, cheap)
    uf = _UF(S)
    for a, b in edges:
        uf.union(a, b)
    labels_cpu = uf.labels()
    labels = labels_cpu.to(device=device, dtype=torch.long)

    return {
        "neighbors_idx": I,        # [S,k]
        "neighbors_sim": Sval,     # [S,k]
        "labels": labels,          # [S]
        "meta": {"backend_used": backend_used, "mutual": bool(mutual), "tau": float(tau), "k": int(k)},
    }


# ============================================================================
# 3)  Representatives 
# ============================================================================

@torch.no_grad()
def aggregate_reps_from_labels(
    K: torch.Tensor,        # [H, S, Dk]
    V: torch.Tensor,        # [H, S, Dv]
    labels: torch.Tensor,   # [S] long, 0..C-1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Segmented mean per cluster per head (fp32 accumulators).
    Returns:
      K_star: [H, C, Dk]   (dtype=K.dtype)
      V_star: [H, C, Dv]   (dtype=V.dtype)
      sizes:  [C]          (dtype=K.dtype; used for log|C|)
    """
    assert K.dim() == 3 and V.dim() == 3, f"Expected K,V [H,S,D], got {K.shape} / {V.shape}"
    H, S, Dk = K.shape
    _, _, Dv = V.shape
    device = K.device
    if S == 0:
        return (K.new_zeros((H, 0, Dk)),
                V.new_zeros((H, 0, Dv)),
                K.new_zeros((0,), dtype=K.dtype))

    labels = labels.to(device=device, dtype=torch.long)
    C = int(labels.max().item()) + 1 if labels.numel() > 0 else 0
    if C == 0:
        return (K.new_zeros((H, 0, Dk)),
                V.new_zeros((H, 0, Dv)),
                K.new_zeros((0,), dtype=K.dtype))

    counts_long = torch.bincount(labels, minlength=C)        # [C] int64
    sizes = counts_long.to(device=device, dtype=K.dtype)     # [C] for downstream log|C|

    K_acc = torch.zeros((H, C, Dk), device=device, dtype=torch.float32)
    V_acc = torch.zeros((H, C, Dv), device=device, dtype=torch.float32)
    for h in range(H):
        K_acc[h].index_add_(0, labels, K[h].to(torch.float32))
        V_acc[h].index_add_(0, labels, V[h].to(torch.float32))

    denom = counts_long.clamp_min_(1).to(device=device, dtype=torch.float32).view(1, C, 1)
    K_star = (K_acc / denom).to(dtype=K.dtype)
    V_star = (V_acc / denom).to(dtype=V.dtype)
    return K_star, V_star, sizes
