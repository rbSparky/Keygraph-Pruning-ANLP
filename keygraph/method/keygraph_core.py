import torch
import numpy as np
import torch.nn.functional as F
from keygraph.method.ivf import _torch_ivf_neighbors, _faiss_ivf_flat_neighbors

def build_descriptors_unrope(keys_per_head, pos_idx, rp_matrix=None, r=32, base=10000.0):
    """
    Build head-invariant descriptors per position using inverse RoPE, *without* view_as_complex.
    Works with bf16/fp16/fp32.

    Args:
        keys_per_head: [H, S, D]  (already RoPE-rotated keys per head)
        pos_idx:       [S]        (0..S-1)
        rp_matrix:     [D, r] or None
        r:             int        (projection dim)
        base:          float      (RoPE base; must match attention)

    Returns:
        phi:        [S, r]   (float32)
        rp_matrix:  [D, r]   (float32, column-normalized)
    """
    assert keys_per_head.dim() == 3, "keys_per_head must be [H,S,D]"
    H, S, D = keys_per_head.shape
    assert D % 2 == 0, "head_dim must be even for RoPE"

    device = keys_per_head.device

    # ---- Work in fp32 for trig; supports bf16/fp16 inputs ----
    k = keys_per_head.to(torch.float32)      # [H,S,D]
    x1 = k[..., 0::2]                        # [H,S,D/2]
    x2 = k[..., 1::2]                        # [H,S,D/2]

    # ---- Build RoPE angles (same formula as your RotatoryPositionalEncoding) ----
    # inv_freq[j] = base^(-(2j)/D) == 1 / base^{(2j)/D}, but your PE uses arange(0, D, 2)/D
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, device=device, dtype=torch.float32) / D))  # [D/2]
    # angles[s,j] = pos_idx[s] * inv_freq[j]
    angles = torch.einsum("s,j->sj", pos_idx.to(device=device, dtype=torch.float32), inv_freq)   # [S, D/2]
    cos = angles.cos().unsqueeze(0)     # [1,S,D/2]
    sin = angles.sin().unsqueeze(0)     # [1,S,D/2]

    # ---- Inverse rotation (undo RoPE):
    # if forward was:  x' = [x1*cos - x2*sin, x2*cos + x1*sin]
    # then inverse is: u1 =  x1'*cos + x2'*sin
    #                  u2 =  x2'*cos - x1'*sin
    u1 = x1 * cos + x2 * sin            # [H,S,D/2]
    u2 = x2 * cos - x1 * sin            # [H,S,D/2]

    # Re-interleave to [H,S,D]
    unrot = torch.empty((H, S, D), device=device, dtype=torch.float32)
    unrot[..., 0::2] = u1
    unrot[..., 1::2] = u2

    # ---- Head-invariant mean ----
    mean_keys = unrot.mean(dim=0)       # [S,D], fp32

    # ---- Random projection to r dims (column-normalized) ----
    if rp_matrix is None:
        rp_matrix = torch.randn(D, r, device=device, dtype=torch.float32)
        rp_matrix = F.normalize(rp_matrix, p=2, dim=0)
    else:
        # Ensure fp32 for the matmul (and column-normalized once)
        if rp_matrix.dtype != torch.float32:
            rp_matrix = rp_matrix.to(torch.float32)
        # (Optional) you can re-normalize if you want strict invariance:
        # rp_matrix = F.normalize(rp_matrix, p=2, dim=0)

    phi = mean_keys @ rp_matrix         # [S,r], fp32
    return phi, rp_matrix

def cosine_similarity(a,b):
    """Compute cosine similarity between two tensors."""
    a_norm =torch.nn.functional.normalize(a,p =2,dim =-1)
    b_norm =torch.nn.functional.normalize(b,p =2,dim =-1)
    return torch.matmul(a_norm,b_norm.transpose(-2,-1))


@torch.no_grad()
def build_knn_and_clusters(
    phi: torch.Tensor,
    tau: float = 0.8,
    k: int = 16,
    mutual: bool = True,
    ann: dict | None = None,
):
    """
    kNN + connected components on GPU.
    - Default: exact (cosine) via full S = phi @ phi.T
    - If ann = {"method": "torch_ivf", "params": {...}}, use Torch-IVF candidate path.

    Returns dict:
      {
        "neighbors_idx": LongTensor [N, k_eff]  (invalid slots = -1)
        "neighbors_sim": Tensor     [N, k_eff]  (invalid slots = -inf)
        "labels":        LongTensor [N]         (0..C-1)
      }
    """
    assert phi.dim() == 2, f"phi must be [N, r], got {phi.shape}"
    device, dt = phi.device, phi.dtype
    N = phi.size(0)

    k_eff = max(0, min(k, max(0, N - 1)))
    if N == 0:
        return {
            "neighbors_idx": torch.empty(0, k_eff, dtype=torch.long, device=device),
            "neighbors_sim": torch.empty(0, k_eff, dtype=dt, device=device),
            "labels":        torch.empty(0, dtype=torch.long, device=device),
        }
    if k_eff == 0:
        neighbors_idx = torch.full((N, 0), -1, dtype=torch.long, device=device)
        neighbors_sim = torch.full((N, 0), float("-inf"), dtype=dt, device=device)
    else:
        method = (ann or {}).get("method", "exact")
        if method == "faiss_ivf_flat":
            params = (ann or {}).get("params", {}) or {}
            try:
                neighbors_idx, neighbors_sim = _faiss_ivf_flat_neighbors(phi, tau=tau, k=k, params=params)
            except Exception as e:
                # Graceful fallback: try Torch-IVF if available, else exact
                # print or log once if you prefer
                if (ann or {}).get("fallback", "torch_ivf") == "exact":
                    phi_norm = F.normalize(phi, p=2, dim=1)
                    S = phi_norm @ phi_norm.T
                    S.fill_diagonal_(-float("inf"))
                    neighbors_sim, neighbors_idx = torch.topk(S, k=max(0, min(k, phi.shape[0]-1)), dim=1)
                    if tau is not None and tau > float("-inf"):
                        valid = neighbors_sim >= float(tau)
                        neighbors_idx = torch.where(valid, neighbors_idx, torch.full_like(neighbors_idx, -1))
                        neg_inf = torch.tensor(float("-inf"), device=phi.device, dtype=phi.dtype)
                        neighbors_sim = torch.where(valid, neighbors_sim, torch.full_like(neighbors_sim, neg_inf))
                else:
                    # Torch-IVF fallback if you kept it in this file
                    neighbors_idx, neighbors_sim = _torch_ivf_neighbors(phi, tau=tau, k=k, params=(ann or {}).get("params", {}))
        elif method == "torch_ivf":
            params = (ann or {}).get("params", {}) or {}
            neighbors_idx, neighbors_sim = _torch_ivf_neighbors(phi, tau=tau, k=k, params=params)
        else:
            # --- exact dense cosine path (Phase B behavior) ---
            phi_norm = F.normalize(phi, p=2, dim=1)
            S = phi_norm @ phi_norm.T     # [N, N]
            S.fill_diagonal_(-float("inf"))
            neighbors_sim, neighbors_idx = torch.topk(S, k=k_eff, dim=1)  # [N, k_eff]
            if tau is not None and tau > float("-inf"):
                valid = neighbors_sim >= float(tau)
                neighbors_idx = torch.where(valid, neighbors_idx, torch.full_like(neighbors_idx, -1))
                neg_inf = torch.tensor(float("-inf"), device=device, dtype=dt)
                neighbors_sim = torch.where(valid, neighbors_sim, torch.full_like(neighbors_sim, neg_inf))
            del S

    # --- Build adjacency A from neighbors (ignoring -1), then mutual/OR ---
    if k_eff == 0:
        M = torch.zeros((N, N), dtype=torch.bool, device=device)
    else:
        A = torch.zeros((N, N), dtype=torch.bool, device=device)
        rows = torch.arange(N, device=device).unsqueeze(1).expand_as(neighbors_idx)  # [N, k_eff]
        mask_valid = neighbors_idx >= 0
        if mask_valid.any():
            A[rows[mask_valid], neighbors_idx[mask_valid]] = True
        M = (A & A.T) if mutual else (A | A.T)
        del A

    # --- Connected components via label propagation (GPU) ---
    labels = torch.arange(N, device=device, dtype=torch.long)
    if M.any():
        labels_f = labels.to(torch.float32)
        INF = torch.tensor(float("inf"), device=device, dtype=torch.float32)
        for _ in range(32):
            neigh_labels = labels_f.unsqueeze(0).expand(N, N).clone()
            neigh_labels = neigh_labels.masked_fill(~M, INF)
            neigh_min = torch.minimum(neigh_labels.min(dim=1).values, labels_f)
            changed = ~torch.isclose(neigh_min, labels_f)
            labels_f = neigh_min
            if not changed.any():
                break
        uniq, inv = torch.unique(labels_f.to(torch.long), return_inverse=True)
        labels = inv  # [N], 0..C-1

    return {
        "neighbors_idx": neighbors_idx,
        "neighbors_sim": neighbors_sim,
        "labels":        labels,
    }



def find_connected_components(n,edges):
    """
    Find connected components using Union-Find.

    Args:
        n: Number of nodes
        edges: List of neighbors for each node

    Returns:
        components: List of components (each component is a list of node indices)
    """

    parent =list(range(n))

    def find(x):
        if parent[x]!=x:
            parent[x]=find(parent[x])
        return parent[x]

    def union(x,y):
        px,py =find(x),find(y)
        if px !=py:
            parent[px]=py


    for i in range(n):
        for nbr in edges[i]:
            union(i,nbr)


    components ={}
    for i in range(n):
        root =find(i)
        if root not in components:
            components[root]=[]
        components[root].append(i)


    return list(components.values())

@torch.no_grad()
def aggregate_reps_from_labels(
    K: torch.Tensor,  # [H, N, Dk]
    V: torch.Tensor,  # [H, N, Dv]
    labels: torch.Tensor,  # [N] long, 0..C-1
):
    """
    GPU segmented mean using per-head fp32 index_add_ (stable & exact up to dtype).
    Returns:
        K_star: [H, C, Dk]  (same dtype as K)
        V_star: [H, C, Dv]  (same dtype as V)
        cluster_sizes: [C]  (same dtype as K/V for downstream log|C|)
    """
    assert K.dim() == 3 and V.dim() == 3, f"Expected K,V [H,N,D], got {K.shape} {V.shape}"
    H, N, Dk = K.shape
    _, _, Dv = V.shape
    device, k_dtype, v_dtype = K.device, K.dtype, V.dtype

    if N == 0:
        return (K.new_zeros((H, 0, Dk)),
                V.new_zeros((H, 0, Dv)),
                K.new_zeros((0,), dtype=k_dtype))

    labels = labels.to(device=device, dtype=torch.long)
    C = int(labels.max().item()) + 1 if labels.numel() > 0 else 0
    if C == 0:
        return (K.new_zeros((H, 0, Dk)),
                V.new_zeros((H, 0, Dv)),
                K.new_zeros((0,), dtype=k_dtype))

    counts_long = torch.bincount(labels, minlength=C)  # [C], int64
    cluster_sizes = counts_long.to(device=device, dtype=k_dtype)

    # fp32 accumulators
    K_star_acc = torch.zeros((H, C, Dk), device=device, dtype=torch.float32)
    V_star_acc = torch.zeros((H, C, Dv), device=device, dtype=torch.float32)

    # Per-head 2-D index_add_ is rock-solid and fast (H is small)
    for h in range(H):
        K_star_acc[h].index_add_(0, labels, K[h].to(torch.float32))  # (C,Dk) += (N,Dk)
        V_star_acc[h].index_add_(0, labels, V[h].to(torch.float32))  # (C,Dv) += (N,Dv)

    denom = counts_long.clamp_min(1).to(device=device, dtype=torch.float32).view(1, C, 1)
    K_star = (K_star_acc / denom).to(dtype=k_dtype)
    V_star = (V_star_acc / denom).to(dtype=v_dtype)
    return K_star, V_star, cluster_sizes
