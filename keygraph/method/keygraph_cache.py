import torch
from keygraph.method.keygraph_core import build_descriptors_unrope,build_knn_and_clusters,aggregate_reps_from_labels


@torch.no_grad()
def _build_members_padded(labels: torch.Tensor):
    """
    From labels [N] -> (members_padded [C, max_m], sizes [C])
    members_padded[c, :sizes[c]] are valid indices; the rest = -1.
    """
    device = labels.device
    N = labels.numel()
    if N == 0:
        return (torch.empty(0, 0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device))
    C = int(labels.max().item()) + 1
    sizes = torch.bincount(labels, minlength=C)
    max_m = int(sizes.max().item())
    members_padded = torch.full((C, max_m), -1, dtype=torch.long, device=device)
    # Fill rows (small Python loop at prefill time is fine)
    for c in range(C):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if idx.numel():
            members_padded[c, :idx.numel()] = idx
    return members_padded, sizes

@torch.no_grad()
def _build_probe_idx(members_padded: torch.Tensor, sizes: torch.Tensor, P: int):
    """
    For each cluster c, sample up to P distinct member indices.
    Returns probe_idx [C, P] with -1 padding where cluster smaller than P.
    """
    device = members_padded.device
    C, max_m = members_padded.shape
    if C == 0 or P <= 0:
        return torch.empty(C, 0, dtype=torch.long, device=device)
    probe_idx = torch.full((C, P), -1, dtype=torch.long, device=device)
    for c in range(C):
        m = int(sizes[c].item())
        if m == 0:
            continue
        take = min(P, m)
        perm = torch.randperm(m, device=device)[:take]
        probe_idx[c, :take] = members_padded[c, perm]
    return probe_idx


class KeyGraphCache:
    """
    Cache for KeyGraph representatives, cluster sizes, and member lists.
    Built once after prefill.
    """

    def __init__(self,model,tokenizer,prompt,r_dim =32,tau =0.8,knn_k =16,
    rescue =True,rescue_probe_size =6,upper_layers_only =False):
        self.model =model
        self.tokenizer =tokenizer
        self.r_dim =r_dim
        self.tau =tau
        self.knn_k =knn_k
        self.rescue =rescue
        self.rescue_probe_size =rescue_probe_size
        self.upper_layers_only =upper_layers_only


        self._build_cache(prompt)

    def _build_cache(self,prompt,ann=False):
        """
        Build the KeyGraph cache from the prompt.
        """

        inputs =self.tokenizer(
        prompt,
        return_tensors ="pt",
        padding =True,
        truncation =True).to(self.model.device)


        config =self.model.config
        num_layers =config.num_hidden_layers
        num_heads =config.num_attention_heads
        head_dim =config.hidden_size //num_heads


        if self.upper_layers_only:

            start_layer =(2 *num_layers)//3
            layers_to_process =list(range(start_layer,num_layers))
        else:
            layers_to_process =list(range(num_layers))


        with torch.no_grad():
            outputs =self.model(
            **inputs,
            output_attentions =False,
            output_hidden_states =False,
            use_cache =True,
            return_dict =True)


        past_key_values =outputs.past_key_values


        self.layer_reps ={}
        self.rp_matrices ={}
        self.total_clusters =0
        self.total_positions =0


        for layer_idx in layers_to_process:
            k,v =past_key_values[layer_idx]


            k =k.squeeze(0)
            v =v.squeeze(0)

            seq_len =k.shape[1]
            self.total_positions +=seq_len
            
            with torch.no_grad():
                pos_idx = torch.arange(seq_len, device=k.device)
                phi, rp_matrix = build_descriptors_unrope(k, pos_idx, r=self.r_dim)
                self.rp_matrices[layer_idx] = rp_matrix

                if ann:
                    knn = build_knn_and_clusters(phi, tau=self.tau, k=self.knn_k, mutual=True, ann={"method": "torch_ivf",
                    "params": {
                        "nlist": None,          
                        "nprobe": 16,           
                        "kmeans_iters": 4,
                        "train_subset": 4096,
                        "seed": 42,
                        "block_max_dots": 200_000
                    }})
                else:
                    knn = build_knn_and_clusters(phi, tau=self.tau, k=self.knn_k, mutual=True)
                
                labels: torch.Tensor = knn["labels"]  # [N]

                num_clusters = int(labels.max().item()) + 1 if labels.numel() else 0
                self.total_clusters += num_clusters

                # Representatives + sizes
                K_star, V_star, cluster_sizes = aggregate_reps_from_labels(k, v, labels)

                # Padded membership + probes (for variance-probe rescue)
                members_padded, sizes_long = _build_members_padded(labels)      # [C, max_m], [C]
                probe_idx = _build_probe_idx(members_padded, sizes_long, P=self.rescue_probe_size)

                self.layer_reps[layer_idx] = {
                    "K_star": K_star,                      # [H, C, Dk]
                    "V_star": V_star,                      # [H, C, Dv]
                    "cluster_sizes": cluster_sizes,        # [C] (float dtype)
                    "original_k": k,                       # [H, S, Dk]
                    "original_v": v,                       # [H, S, Dv]
                    "labels": labels,                      # [S]
                    "members_padded": members_padded,      # [C, max_m] (long, -1 padded)
                    "sizes_long": sizes_long,              # [C] (long)
                    "probe_idx": probe_idx,                # [C, P] (long, -1 padded)
                }


        self.full_kv_layers =[i for i in range(num_layers)if i not in layers_to_process]
        if self.full_kv_layers:
            self.full_past_key_values =tuple(past_key_values[i]for i in self.full_kv_layers)
        else:
            self.full_past_key_values =None

    def get_layer_representatives(self,layer_idx):
        """Get representatives for a specific layer."""
        if layer_idx in self.layer_reps:
            return self.layer_reps[layer_idx]
        return None

    def get_compression_ratio(self):
        """Get the estimated compression ratio."""
        if self.total_positions ==0:
            return 0
        return self.total_clusters /self.total_positions

    def get_kv_bytes_saved(self):
        """Estimate KV cache bytes saved."""


        config =self.model.config
        head_dim =config.hidden_size //config.num_attention_heads
        kv_size_per_token =2 *config.num_attention_heads *head_dim *4

        original_size =self.total_positions *kv_size_per_token
        compressed_size =self.total_clusters *kv_size_per_token

        return original_size -compressed_size