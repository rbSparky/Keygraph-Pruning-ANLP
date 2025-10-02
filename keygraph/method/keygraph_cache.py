import torch
from keygraph.method.keygraph_core import (
    build_descriptors_unrope,
    build_knn_and_clusters,
    aggregate_reps_from_labels,
)


@torch.no_grad()
def _build_members_padded(labels: torch.Tensor):
    """
    From labels [N] -> (members_padded [C, max_m], sizes [C])
    members_padded[c, :sizes[c]] are valid indices; the rest = -1.
    """
    device = labels.device
    N = labels.numel()
    if N == 0:
        return (
            torch.empty(0, 0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    C = int(labels.max().item()) + 1
    sizes = torch.bincount(labels, minlength=C)
    max_m = int(sizes.max().item())
    members_padded = torch.full((C, max_m), -1, dtype=torch.long, device=device)
    # Python loop only runs once at prefill -> acceptable
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
    Built once after prefill. Originals are offloaded to CPU; GPU keeps reps (+ small tail).
    """

    def __init__(
        self,
        model,
        tokenizer,
        prompt,
        r_dim: int = 32,
        tau: float = 0.8,
        knn_k: int = 16,
        rescue: bool = True,
        rescue_probe_size: int = 6,
        upper_layers_only: bool = False,
        tail_max_len: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.r_dim = r_dim
        self.tau = tau
        self.knn_k = knn_k
        self.rescue = rescue
        self.rescue_probe_size = rescue_probe_size
        self.upper_layers_only = upper_layers_only
        self.tail_max_len = tail_max_len

        # Per-layer tiny tail buffers (GPU resident)
        self.tail = {}  # layer_idx -> {"K": [B,H,T,D], "V": [B,H,T,D]}

        self._build_cache(prompt)

    def _build_cache(self, prompt, ann: bool = False):
        """
        Build the KeyGraph cache from the prompt (prefill).
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        config = self.model.config
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads

        if self.upper_layers_only:
            start_layer = (2 * num_layers) // 3
            layers_to_process = list(range(start_layer, num_layers))
        else:
            layers_to_process = list(range(num_layers))

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=True,
                return_dict=True,
            )

        past_key_values = outputs.past_key_values

        self.layer_reps = {}
        self.rp_matrices = {}
        self.total_clusters = 0
        self.total_positions = 0

        for layer_idx in layers_to_process:
            # HF returns (k, v) with shape [B, H, S, D]
            k, v = past_key_values[layer_idx]
            k = k.squeeze(0)  # [H, S, D]
            v = v.squeeze(0)  # [H, S, D]
            seq_len = k.shape[1]
            self.total_positions += seq_len

            with torch.no_grad():
                # Build descriptors (UnRoPE etc.)
                pos_idx = torch.arange(seq_len, device=k.device)
                phi, rp_matrix = build_descriptors_unrope(k, pos_idx, r=self.r_dim)
                self.rp_matrices[layer_idx] = rp_matrix

                if ann:
                    knn = build_knn_and_clusters(
                        phi,
                        tau=self.tau,
                        k=self.knn_k,
                        mutual=True,
                        ann={
                            "method": "torch_ivf",
                            "params": {
                                "nlist": None,
                                "nprobe": 16,
                                "kmeans_iters": 4,
                                "train_subset": 4096,
                                "seed": 42,
                                "block_max_dots": 200_000,
                            },
                        },
                    )
                else:
                    knn = build_knn_and_clusters(
                        phi, tau=self.tau, k=self.knn_k, mutual=True
                    )

                labels: torch.Tensor = knn["labels"]  # [S]
                num_clusters = int(labels.max().item()) + 1 if labels.numel() else 0
                self.total_clusters += num_clusters

                # Representatives + sizes
                K_star, V_star, cluster_sizes = aggregate_reps_from_labels(k, v, labels)

                # Padded membership + probes (for variance-probe rescue)
                members_padded, sizes_long = _build_members_padded(labels)  # [C,max_m],[C]
                probe_idx = _build_probe_idx(
                    members_padded, sizes_long, P=self.rescue_probe_size
                )

                # Offload originals to CPU to save VRAM; keep reps on GPU
                original_k_cpu = k.to("cpu", non_blocking=True).contiguous()
                original_v_cpu = v.to("cpu", non_blocking=True).contiguous()
                # free GPU copies of k,v (locals will go out of scope)

                self.layer_reps[layer_idx] = {
                    "K_star": K_star.contiguous(),  # [H, C, D]
                    "V_star": V_star.contiguous(),  # [H, C, D]
                    "cluster_sizes": cluster_sizes,  # [C] (float)
                    "original_k_cpu": original_k_cpu,  # [H, S, D] on CPU
                    "original_v_cpu": original_v_cpu,  # [H, S, D] on CPU
                    "labels": labels,  # [S] on GPU
                    "members_padded": members_padded,  # [C, max_m] (long, -1 padded)
                    "sizes_long": sizes_long,  # [C] (long)
                    "probe_idx": probe_idx,  # [C, P] (long, -1 padded)
                }

            # Initialize empty tail for this layer (GPU)
            self.tail[layer_idx] = {"K": None, "V": None}

        # Layers not processed by KeyGraph -> keep full past_kv
        self.full_kv_layers = [i for i in range(num_layers) if i not in layers_to_process]
        if self.full_kv_layers:
            self.full_past_key_values = tuple(past_key_values[i] for i in self.full_kv_layers)
        else:
            self.full_past_key_values = None

    # -------- Tail buffer (recent tokens) --------

    @torch.no_grad()
    def append_tail(self, layer_idx: int, K_step: torch.Tensor, V_step: torch.Tensor, max_len: int = None):
        """
        Append the most recent step's K/V to a small GPU ring buffer.
        Expects K_step, V_step: [B, H, 1, D] (decode is B=1 typically).
        """
        if max_len is None:
            max_len = self.tail_max_len
        t = self.tail[layer_idx]
        if t["K"] is None:
            t["K"] = K_step
            t["V"] = V_step
        else:
            t["K"] = torch.cat([t["K"], K_step], dim=2)[..., -max_len:, :].contiguous()
            t["V"] = torch.cat([t["V"], V_step], dim=2)[..., -max_len:, :].contiguous()

    @torch.no_grad()
    def get_tail(self, layer_idx: int):
        t = self.tail[layer_idx]
        return (t["K"], t["V"]) if t["K"] is not None else (None, None)

    # -------- CPU originals -> on-demand rescue fetch --------

    @torch.no_grad()
    def gather_original(self, layer_idx: int, member_idx: torch.LongTensor, device, dtype):
        """
        Fetch a subset of originals by absolute token indices (1D LongTensor).
        Returns (K_sel, V_sel) with shape [H, M, D] on 'device' with 'dtype'.
        """
        rep = self.layer_reps[layer_idx]
        srcK = rep["original_k_cpu"]  # [H, S, D] on CPU
        srcV = rep["original_v_cpu"]  # [H, S, D] on CPU
        if member_idx.numel() == 0:
            H, _, Dk = srcK.shape
            return (
                torch.empty(H, 0, Dk, device=device, dtype=dtype),
                torch.empty(H, 0, srcV.shape[-1], device=device, dtype=dtype),
            )
        idx_cpu = member_idx.to(srcK.device, non_blocking=True)
        K_sel = srcK.index_select(1, idx_cpu).to(device=device, dtype=dtype, non_blocking=True).contiguous()
        V_sel = srcV.index_select(1, idx_cpu).to(device=device, dtype=dtype, non_blocking=True).contiguous()
        return K_sel, V_sel

    # -------- Public helpers --------

    def get_layer_representatives(self, layer_idx):
        """Get representatives for a specific layer."""
        return self.layer_reps.get(layer_idx, None)

    def get_compression_ratio(self):
        """Get the estimated compression ratio."""
        if self.total_positions == 0:
            return 0.0
        return float(self.total_clusters) / float(self.total_positions)

    def get_kv_bytes_saved(self):
        """Estimate KV cache bytes saved (ignores tiny tail)."""
        config = self.model.config
        head_dim = config.hidden_size // config.num_attention_heads
        kv_size_per_token = 2 * config.num_attention_heads * head_dim * 4  # fp32-bytes estimate
        original_size = self.total_positions * kv_size_per_token
        compressed_size = self.total_clusters * kv_size_per_token
        return original_size - compressed_size
