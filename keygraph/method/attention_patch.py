import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from keygraph.method.keygraph_cache import KeyGraphCache
import math

# NOTE: The RotatoryPositionalEncoding class remains unchanged.
class RotatoryPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len, base=10000):
        super(RotatoryPositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.base = base
        self._create_rotary_embedding()
    def reset_parameters(self):
        self._create_rotary_embedding()

    def _create_rotary_embedding(self):
        theta = 1.0 / (self.base ** (torch.arange(0, self.embedding_dim, 2)[: (self.embedding_dim // 2)].float() / self.embedding_dim))
        self.register_buffer("theta", theta)
        seq_len = torch.arange(self.max_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum("i,j->ij", seq_len, self.theta).float()
        cache=torch.stack([idx_theta.cos(), idx_theta.sin()], dim=-1)
        self.register_buffer("cache", cache)
    def forward(self, x):
        B, S, H, d_k = x.shape
        x = x.view(B, S, H, d_k // 2, 2)
        rope = self.cache[:S].unsqueeze(0).unsqueeze(2)  
        cos, sin = rope[..., 0], rope[..., 1]
        x1, x2 = x[..., 0], x[..., 1]
        x_rotated = torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return x_rotated.flatten(-2)  

@contextmanager
def keygraph_attention_patch(model, keygraph_cache, rescue_threshold=0.5):
    """
    Context manager to patch attention with KeyGraph representatives.
    """
    original_forward_methods = {}

    def create_patched_forward(layer_idx):
        rope = RotatoryPositionalEncoding(model.config.hidden_size // model.config.num_attention_heads, max_len=5000)

        def patched_forward(self, hidden_states, attention_mask=None, position_ids=None,
                            past_key_value=None, output_attentions=False, use_cache=True):

            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Apply RoPE to current query and key states
            query_states = rope(query_states.transpose(1,2)).transpose(1,2)
            key_states = rope(key_states.transpose(1,2)).transpose(1,2)
            
            K_star_stacked, V_star_stacked, cluster_sizes_tensor = None, None, None
            num_clusters = 0

            layer_rep = keygraph_cache.get_layer_representatives(layer_idx)
            if layer_rep and 'K_star' in layer_rep and layer_rep['K_star']:
                K_star = layer_rep['K_star']
                V_star = layer_rep['V_star']
                cluster_sizes = layer_rep['cluster_sizes']
                num_clusters = K_star.shape[1]

                # The representatives are now the "past_key_value"
                K_star_stacked = K_star.unsqueeze(0).expand(bsz, -1, -1, -1)
                V_star_stacked = V_star.unsqueeze(0).expand(bsz, -1, -1, -1)
                cluster_sizes_tensor = torch.tensor(cluster_sizes, device=query_states.device, dtype=query_states.dtype)

                # Combine representatives with current keys and values
                key_states = torch.cat([K_star_stacked, key_states], dim=2)
                value_states = torch.cat([V_star_stacked, value_states], dim=2)

            # --- Vectorized Attention Calculation ---
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)

            # FIX: Add the log-mass compensation term for the cluster representatives
            if num_clusters > 0 and cluster_sizes_tensor is not None:
                log_mass = torch.log(cluster_sizes_tensor).view(1, 1, 1, num_clusters)
                attn_weights[:, :, :, :num_clusters] += log_mass

            # Apply attention mask
            kv_seq_len = key_states.shape[-2]
            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                     raise ValueError(f"Attention mask shape is incorrect. Expected {(bsz, 1, q_len, kv_seq_len)}, got {attention_mask.size()}")
                attn_weights = attn_weights + attention_mask

            attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            # --- Main Attention Output (Calculated Vectorially) ---
            attn_output = torch.matmul(attn_probs, value_states)

            # --- Rescue Mechanism (Applied per-head if needed) ---
            entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-9), dim=-1)
            needs_rescue_per_head = torch.any(entropy < rescue_threshold, dim=-1) # Shape: (bsz, num_heads)

            if torch.any(needs_rescue_per_head) and layer_rep and 'original_k' in layer_rep:
                original_k = layer_rep['original_k'] # Shape: (num_heads, seq_len, head_dim)
                original_v = layer_rep['original_v'] # Shape: (num_heads, seq_len, head_dim)
                
                # Iterate only through heads that need rescuing
                for h in range(self.num_heads):
                    if not torch.any(needs_rescue_per_head[:, h]):
                        continue

                    # For simplicity, if any token in the batch needs rescue for this head,
                    # we re-compute attention for the whole batch for this head using original KV.
                    # A more optimized approach would be to do this only for specific batch items.
                    
                    # Rebuild K/V cache for this head with original values
                    rescued_k_h = original_k[h].unsqueeze(0).expand(bsz, -1, -1)
                    rescued_v_h = original_v[h].unsqueeze(0).expand(bsz, -1, -1)
                    
                    # Concatenate with current token's K/V
                    current_key_states_h = key_states[:, h, num_clusters:, :] # Get current keys for head h
                    current_value_states_h = value_states[:, h, num_clusters:, :] # Get current values for head h
                    
                    full_rescued_k = torch.cat([rescued_k_h, current_key_states_h], dim=1)
                    full_rescued_v = torch.cat([rescued_v_h, current_value_states_h], dim=1)
                    
                    # Re-compute attention for this head
                    query_states_h = query_states[:, h, :, :]
                    attn_weights_h = torch.matmul(query_states_h, full_rescued_k.transpose(1, 2)) / (self.head_dim ** 0.5)

                    # Note: A rescued attention mask would also be needed here if the original sequence length is different
                    attn_probs_h = F.softmax(attn_weights_h, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    attn_output_h = torch.matmul(attn_probs_h, full_rescued_v)
                    
                    # Overwrite the output for the rescued head
                    attn_output[:, h, :, :] = attn_output_h


            # --- Final Projection ---
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            present_key_value = (key_states, value_states) if use_cache else None
            return attn_output, attn_weights, present_key_value

        return patched_forward

    config = model.config
    num_layers = config.num_hidden_layers
    try:
        for i in range(num_layers):
            layer = model.model.layers[i]
            original_forward_methods[i] = layer.self_attn.forward
            # Create a unique patched forward for each layer
            layer.self_attn.forward = create_patched_forward(i).__get__(layer.self_attn)
        yield keygraph_cache
    finally:
        for layer_idx, original_method in original_forward_methods.items():
            layer = model.model.layers[layer_idx]
            layer.self_attn.forward = original_method