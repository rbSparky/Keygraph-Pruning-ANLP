import torch
import numpy as np
import torch.nn.functional as F


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


def build_knn_and_clusters(phi,tau =0.8,k =16,mutual =True):
    """
    Build kNN graph and clusters using PyTorch operations.

    Args:
        phi: Descriptors of shape [seq_len, r]
        tau: Cosine similarity threshold
        k: Number of neighbors
        mutual: Whether to use mutual kNN

    Returns:
        neighbors: List of neighbors for each node
        clusters: List of clusters (connected components)
    """
    seq_len,r =phi.shape


    similarities =cosine_similarity(phi,phi)


    diag_indices =torch.arange(seq_len,device =phi.device)
    similarities[diag_indices,diag_indices]=-1.0


    topk_similarities,topk_indices =torch.topk(similarities,k =min(k +1,seq_len),dim =1)


    neighbors =[]
    for i in range(seq_len):

        valid_neighbors =[]
        for j in range(topk_indices[i].shape[0]):
            neighbor_idx =topk_indices[i,j].item()
            sim =topk_similarities[i,j].item()

            if neighbor_idx !=i and sim >=tau:

                    valid_neighbors.append(neighbor_idx)
        neighbors.append(valid_neighbors)


    if mutual:
        mutual_neighbors =[]
        for i in range(seq_len):
            mutual_nbrs =[]
            for nbr in neighbors[i]:
                if i in neighbors[nbr]:
                    mutual_nbrs.append(nbr)
            mutual_neighbors.append(mutual_nbrs)
        neighbors =mutual_neighbors


    clusters =find_connected_components(seq_len,neighbors)

    return neighbors,clusters

@torch.no_grad()
def build_knn_and_clusters(
    phi: torch.Tensor,
    tau: float = 0.8,
    k: int = 16,
    mutual: bool = True,
):
    """
    GPU kNN (cosine) + connected components (label propagation).

    Args:
        phi: [N, r] descriptors (any dtype supported by matmul/topk). Will be L2-normalized along dim=1.
        tau: cosine threshold. Neighbors with sim < tau are discarded.
        k:   retain top-k neighbors per node (after masking the diagonal).
        mutual: if True, keep only mutual kNN edges; else symmetrize with OR.

    Returns:
        {
          'neighbors_idx': LongTensor [N, k]  (invalid slots = -1)
          'neighbors_sim': Tensor     [N, k]  (invalid slots = -inf)
          'labels':        LongTensor [N]     (cluster label for each node, 0..C-1)
        }
    """
    assert phi.dim() == 2, f"phi must be [N, r], got {phi.shape}"
    device, dt = phi.device, phi.dtype
    N = phi.size(0)
    if N == 0:
        return {
            "neighbors_idx": torch.empty(0, k, dtype=torch.long, device=device),
            "neighbors_sim": torch.empty(0, k, dtype=dt, device=device),
            "labels":        torch.empty(0, dtype=torch.long, device=device),
        }

    # 1) Cosine sim on device
    phi = F.normalize(phi, p=2, dim=1)
    # Full sim matrix (we free it ASAP after topk)
    S = phi @ phi.T  # [N, N]
    S.fill_diagonal_(-float("inf"))

    k_eff = max(0, min(k, N - 1))
    if k_eff == 0:
        neighbors_idx = torch.full((N, 0), -1, dtype=torch.long, device=device)
        neighbors_sim = torch.full((N, 0), float("-inf"), dtype=dt, device=device)
    else:
        neighbors_sim, neighbors_idx = torch.topk(S, k=k_eff, dim=1)  # [N, k]
        # threshold
        valid = neighbors_sim >= tau
        neighbors_idx = torch.where(valid, neighbors_idx, torch.full_like(neighbors_idx, -1))
        neg_inf = torch.tensor(float("-inf"), device=device, dtype=dt)
        neighbors_sim = torch.where(valid, neighbors_sim, torch.full_like(neighbors_sim, neg_inf))

    # Build boolean adjacency A from neighbors (ignoring -1 slots)
    # We will discard S now to free memory.
    del S

    if k_eff == 0:
        A = torch.zeros((N, N), dtype=torch.bool, device=device)
    else:
        A = torch.zeros((N, N), dtype=torch.bool, device=device)
        row = torch.arange(N, device=device).unsqueeze(1).expand_as(neighbors_idx)  # [N, k]
        mask_valid = neighbors_idx >= 0
        row = row[mask_valid]
        col = neighbors_idx[mask_valid]
        if row.numel() > 0:
            A[row, col] = True

    # Mutual or OR symmetrization
    M = (A & A.T) if mutual else (A | A.T)
    del A

    # 2) Connected components via label propagation (GPU)
    # labels initialized to own indices
    labels = torch.arange(N, device=device, dtype=torch.long)

    # If there are no edges, every node is its own component
    if not M.any():
        # Re-label to 0..N-1 to keep the contract stable
        # (already true because labels == arange(N))
        return {
            "neighbors_idx": neighbors_idx,
            "neighbors_sim": neighbors_sim,
            "labels":        labels,  # 0..N-1
        }

    # Iterative min-label relaxation: at each step, each node takes min label among its neighbors ∪ {itself}.
    # Implemented with dense masks; memory OK up to ~4k tokens.
    # To keep types simple, we compute mins in float32 then cast back.
    labels_f = labels.to(torch.float32)

    # Precompute a big 'inf' sentinel
    INF = torch.tensor(float("inf"), device=device, dtype=torch.float32)

    # Cap iterations; usually converges in < 10
    for _ in range(32):
        # neighbor label matrix: [N, N], with inf where no edge
        neigh_labels = labels_f.unsqueeze(0).expand(N, N).clone()
        neigh_labels = neigh_labels.masked_fill(~M, INF)
        neigh_min = torch.minimum(neigh_labels.min(dim=1).values, labels_f)  # include self
        changed = ~torch.isclose(neigh_min, labels_f)
        labels_f = neigh_min
        if not changed.any():
            break

    # Compact labels to 0..C-1
    # (Stable compaction so small ints map to small ints)
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



def aggregate_reps(K: torch.Tensor,
                   V: torch.Tensor,
                   clusters,
                   per_head: bool = True):
    """
    Aggregate keys/values for each cluster to form representatives.

    Args:
        K: Keys of shape [num_heads, seq_len, head_dim_k] (if per_head) or [seq_len, head_dim_k]
        V: Values of shape [num_heads, seq_len, head_dim_v] (if per_head) or [seq_len, head_dim_v]
        clusters: List[List[int]] cluster -> member indices
        per_head: Expect [H, N, D] inputs and return [H, C, D] outputs if True

    Returns:
        K_star: Tensor of shape [H, C, Dk] if per_head else [C, Dk]
        V_star: Tensor of shape [H, C, Dv] if per_head else [C, Dv]
        cluster_sizes: Tensor of shape [C] (float dtype, same as K/V)
    """
    if per_head:
        H, N, Dk = K.shape
        _, _, Dv = V.shape
        device, dtype = K.device, K.dtype
        C = len(clusters)
        if C == 0:
            return (K.new_zeros((H, 0, Dk)),
                    V.new_zeros((H, 0, Dv)),
                    K.new_zeros((0,), dtype=dtype))

        K_star_list = []
        V_star_list = []
        sizes = []

        for cluster in clusters:
            idx = torch.as_tensor(cluster, device=device, dtype=torch.long)
            # [H, |cluster|, D]
            Kc = K.index_select(1, idx)
            Vc = V.index_select(1, idx)
            # mean over members -> [H, D]
            Kmu = Kc.mean(dim=1)
            Vmu = Vc.mean(dim=1)
            K_star_list.append(Kmu)
            V_star_list.append(Vmu)
            sizes.append(idx.numel())

        K_star = torch.stack(K_star_list, dim=1)     # [H, C, Dk]
        V_star = torch.stack(V_star_list, dim=1)     # [H, C, Dv]
        cluster_sizes = torch.tensor(sizes, device=device, dtype=dtype)  # float dtype for log()
        return K_star, V_star, cluster_sizes

    # Non-per-head path 
    N, Dk = K.shape
    _, Dv = V.shape
    device, dtype = K.device, K.dtype
    C = len(clusters)
    if C == 0:
        return (K.new_zeros((0, Dk)),
                V.new_zeros((0, Dv)),
                K.new_zeros((0,), dtype=dtype))
    K_star_list = []
    V_star_list = []
    sizes = []
    for cluster in clusters:
        idx = torch.as_tensor(cluster, device=device, dtype=torch.long)
        Kmu = K.index_select(0, idx).mean(dim=0)
        Vmu = V.index_select(0, idx).mean(dim=0)
        K_star_list.append(Kmu)
        V_star_list.append(Vmu)
        sizes.append(idx.numel())
    K_star = torch.stack(K_star_list, dim=0)         # [C, Dk]
    V_star = torch.stack(V_star_list, dim=0)         # [C, Dv]
    cluster_sizes = torch.tensor(sizes, device=device, dtype=dtype)
    return K_star, V_star, cluster_sizes