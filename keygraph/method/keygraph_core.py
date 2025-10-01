import torch
import numpy as np
import torch.nn.functional as F
from keygraph.method.ivf import _torch_ivf_neighbors, _faiss_ivf_flat_neighbors

def build_descriptors_unrope(keys_per_head,pos_idx,rp_matrix =None,r =32):
    """
    Build head-invariant descriptors per position using correct inverse RoPE.

    Args:
        keys_per_head: Tensor of shape [num_heads, seq_len, head_dim]
        pos_idx: Position indices of shape [seq_len]
        rp_matrix: Precomputed random projection matrix
        r: Dimension of random projection

    Returns:
        phi: Descriptors of shape [seq_len, r]
    """
    num_heads,seq_len,head_dim =keys_per_head.shape




    i =torch.arange(1,head_dim //2 +1,dtype =torch.float32,device =keys_per_head.device)
    theta =10000 **(-2 *(i -1)/head_dim)


    keys_reshaped =keys_per_head.view(num_heads,seq_len,head_dim //2,2)


    keys_complex =torch.view_as_complex(keys_reshaped)



    angles =-pos_idx.unsqueeze(1)*theta.unsqueeze(0)


    rotation_vector =torch.polar(torch.ones_like(angles),angles)


    rotation_vector =rotation_vector.unsqueeze(0).unsqueeze(-1)
    rotation_vector =rotation_vector.expand(num_heads,-1,-1,-1)
    rotation_vector =rotation_vector.squeeze(-1)


    unrope_keys_complex =keys_complex *rotation_vector


    unrope_keys_real =torch.view_as_real(unrope_keys_complex)
    unrope_keys =unrope_keys_real.view(num_heads,seq_len,head_dim)


    unrope_keys =torch.nn.functional.normalize(unrope_keys,p =2,dim =-1)


    mean_keys =torch.mean(unrope_keys,dim =0)


    if rp_matrix is None:

        rp_matrix =torch.randn(head_dim,r,device =mean_keys.device)

        rp_matrix =torch.nn.functional.normalize(rp_matrix,p =2,dim =0)


    phi =torch.matmul(mean_keys,rp_matrix)

    return phi,rp_matrix


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
