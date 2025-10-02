import torch
import numpy as np


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


def build_knn_and_clusters(phi,tau =0.8,k =16,mutual =True,bucket_stride =512):
    """
    Build kNN graph and clusters using PyTorch operations.

    Args:
        phi: Descriptors of shape [seq_len, r]
        tau: Cosine similarity threshold
        k: Number of neighbors
        mutual: Whether to use mutual kNN
        bucket_stride: Position bucket size

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

                if bucket_stride >0:
                    if abs(i -neighbor_idx)<bucket_stride:
                        valid_neighbors.append(neighbor_idx)
                else:
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


def aggregate_reps(K,V,clusters,per_head =True):
    """
    Aggregate keys/values for each cluster to form representatives.

    Args:
        K: Keys of shape [num_heads, seq_len, head_dim_k] or [seq_len, head_dim_k]
        V: Values of shape [num_heads, seq_len, head_dim_v] or [seq_len, head_dim_v]
        clusters: List of clusters
        per_head: Whether to process per head or aggregate across all positions

    Returns:
        K_star: Representative keys for each cluster
        V_star: Representative values for each cluster
        cluster_sizes: Size of each cluster
    """
    if per_head:
        num_heads,seq_len,head_dim_k =K.shape
        _,_,head_dim_v =V.shape
    else:
        seq_len,head_dim_k =K.shape
        _,head_dim_v =V.shape

    K_star =[]
    V_star =[]
    cluster_sizes =[]

    for cluster in clusters:
        cluster_size =len(cluster)
        cluster_sizes.append(cluster_size)

        if per_head:

            cluster_K =K[:,cluster,:]
            cluster_V =V[:,cluster,:]


            mean_K =torch.mean(cluster_K,dim =1)
            mean_V =torch.mean(cluster_V,dim =1)
        else:

            cluster_K =K[cluster,:]
            cluster_V =V[cluster,:]


            mean_K =torch.mean(cluster_K,dim =0)
            mean_V =torch.mean(cluster_V,dim =0)

        K_star.append(mean_K)
        V_star.append(mean_V)

    return K_star,V_star,cluster_sizes