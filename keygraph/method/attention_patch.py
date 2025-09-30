import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from keygraph.method.keygraph_cache import KeyGraphCache
import math

def _expand_attn_mask_for_reps(attn_mask: torch.Tensor, num_reps: int):
    """
    Prepend a zero (no-penalty) block for 'num_reps' positions to the kv dimension
    so that the caller's mask aligns after we prefix representative tokens.
    """
    if attn_mask is None or num_reps == 0:
        return attn_mask
    bsz, one, q_len, kv_len = attn_mask.shape
    zeros = torch.zeros(bsz, 1, q_len, num_reps, device=attn_mask.device, dtype=attn_mask.dtype)
    return torch.cat([zeros, attn_mask], dim=-1)

def _make_rep_bias(cluster_sizes: torch.Tensor, q_len: int) -> torch.Tensor:
    """
    log|C| additive bias for the C rep positions, broadcastable to [B,1,Q,C].
    cluster_sizes: [C] (float dtype, on same device)
    """
    if cluster_sizes.numel() == 0:
        return None
    log_mass = torch.log(cluster_sizes.clamp_min(1.0))  # [C]
    return log_mass.view(1, 1, 1, -1).expand(1, 1, q_len, -1)  # [1,1,Q,C]

def _sdpa_with_keygraph(
    query_states,        # [B,H,Q,D]
    key_states,          # [B,H,S,D]
    value_states,        # [B,H,S,D]
    attention_mask,      # [B,1,Q,S] or None (additive mask)
    keygraph_cache,      # cache object with get_layer_representatives(layer_idx)
    layer_idx,           # int
    epsilon_var: float = 1e-3,
):
    B, H, Q, D = query_states.shape
    S = key_states.shape[2]

    # Default path (no KeyGraph)
    K_full = key_states
    V_full = value_states
    attn_mask_total = attention_mask.to(torch.float32) if attention_mask is not None else None

    layer_rep = keygraph_cache.get_layer_representatives(layer_idx)
    if layer_rep is None or 'K_star' not in layer_rep:
        return F.scaled_dot_product_attention(
            query_states, K_full, V_full, attn_mask=attn_mask_total, dropout_p=0.0, is_causal=False
        )

    # Pull cached tensors (device/dtype align)
    K_star = layer_rep['K_star'].to(key_states.device, key_states.dtype)         # [H,C,D]
    V_star = layer_rep['V_star'].to(value_states.device, value_states.dtype)     # [H,C,D]
    cluster_sizes = layer_rep['cluster_sizes'].to(query_states.device, query_states.dtype)  # [C]

    members_padded = layer_rep['members_padded']    # [C, max_m], long
    sizes_long     = layer_rep['sizes_long']        # [C], long
    probe_idx      = layer_rep['probe_idx']         # [C, P], long
    orig_k         = layer_rep['original_k'].to(key_states.device, key_states.dtype)  # [H,S,D]
    orig_v         = layer_rep['original_v'].to(value_states.device, value_states.dtype)

    C = K_star.shape[1]
    K_rep = K_star.unsqueeze(0).expand(B, -1, -1, -1)   # [B,H,C,D]
    V_rep = V_star.unsqueeze(0).expand(B, -1, -1, -1)   # [B,H,C,D]

    # --- variance-probe (global per layer) ---
    if probe_idx.numel() > 0:
        P   = probe_idx.shape[1]
        d_k = orig_k.shape[-1]
        probe_clamped = torch.clamp(probe_idx, min=0)                         # [C,P]
        idx4 = probe_clamped.view(1, C, P, 1).expand(H, C, P, d_k)            # [H,C,P,D]
        K_probe = orig_k.unsqueeze(1).expand(H, C, S, d_k).gather(2, idx4)    # [H,C,P,D]
        valid  = (probe_idx >= 0).to(query_states.dtype).view(1,1,1,C,P)      # [1,1,1,C,P]
        scores = torch.einsum('bhqd,hcpd->bhqcp', query_states, K_probe) / (D ** 0.5)
        sum_w  = valid.sum(dim=-1).clamp_min(1.0)                             # [1,1,1,C]
        mean   = (scores * valid).sum(dim=-1) / sum_w                         # [B,H,Q,C]
        mean2  = (scores * scores * valid).sum(dim=-1) / sum_w                # [B,H,Q,C]
        var    = (mean2 - mean * mean).relu()                                 # [B,H,Q,C]
        expand_mask_c = (var > epsilon_var).any(dim=(0,1,2))                  # [C] bool
    else:
        expand_mask_c = torch.zeros(C, dtype=torch.bool, device=query_states.device)

    keep_mask_c = ~expand_mask_c
    keep_idx = keep_mask_c.nonzero(as_tuple=True)[0]   # [C_keep]
    exp_idx  = expand_mask_c.nonzero(as_tuple=True)[0] # [C_exp]

    # kept reps
    if keep_idx.numel() > 0:
        K_keep = K_rep.index_select(2, keep_idx)        # [B,H,C_keep,D]
        V_keep = V_rep.index_select(2, keep_idx)
        sizes_keep = cluster_sizes.index_select(0, keep_idx)  # [C_keep]
    else:
        K_keep = K_rep[:, :, :0, :]
        V_keep = V_rep[:, :, :0, :]
        sizes_keep = cluster_sizes[:0]

    # expanded members
    if exp_idx.numel() > 0:
        mem_rows = members_padded.index_select(0, exp_idx)  # [C_exp, max_m]
        m_sizes  = sizes_long.index_select(0, exp_idx)      # [C_exp]
        max_m    = mem_rows.shape[1]
        valid_m  = (torch.arange(max_m, device=mem_rows.device).view(1,-1) < m_sizes.view(-1,1))
        sel      = mem_rows[valid_m]                        # [M_total]
        K_mem = orig_k.unsqueeze(0).expand(B, -1, -1, -1).index_select(2, sel)  # [B,H,M,D]
        V_mem = orig_v.unsqueeze(0).expand(B, -1, -1, -1).index_select(2, sel)
    else:
        K_mem = K_rep[:, :, :0, :]
        V_mem = V_rep[:, :, :0, :]

    # concatenate: kept reps + expanded members + original sequence
    K_full = torch.cat([K_keep, K_mem, key_states], dim=2)  # [B,H,Ktot,D]
    V_full = torch.cat([V_keep, V_mem, value_states], dim=2)
    prefix_len = K_keep.shape[2] + K_mem.shape[2]

    # additive mask: zeros for prefix + caller mask
    if attention_mask is not None:
        zeros_prefix = torch.zeros(B, 1, Q, prefix_len, device=attention_mask.device, dtype=attention_mask.dtype)
        attn_mask_total = torch.cat([zeros_prefix, attention_mask], dim=-1).to(torch.float32)
    else:
        attn_mask_total = None

    # log|C| bias only on kept reps (first C_keep positions of prefix)
    # log|C| bias only on kept reps (first C_keep positions), then pad to FULL KV length
    if sizes_keep.numel() > 0:
        rep_bias = _make_rep_bias(sizes_keep, Q)  # [1,1,Q,C_keep]
        kv_total = K_full.shape[2]                # prefix_len + S
        pad = kv_total - rep_bias.shape[-1]
        if pad > 0:
            rep_bias = torch.cat(
                [rep_bias, torch.zeros(rep_bias.shape[:-1] + (pad,), device=rep_bias.device, dtype=rep_bias.dtype)],
                dim=-1
            )  # [1,1,Q,kv_total]
        attn_mask_total = (attn_mask_total if attn_mask_total is not None else 0.0) + rep_bias.to(torch.float32)

    return F.scaled_dot_product_attention(
        query_states, K_full, V_full, attn_mask=attn_mask_total, dropout_p=0.0, is_causal=False
    )



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

            rope = rope.to(hidden_states.device, hidden_states.dtype)

            # Apply RoPE to current query and key states
            query_states = rope(query_states.transpose(1,2)).transpose(1,2)
            key_states = rope(key_states.transpose(1,2)).transpose(1,2)
                
            # === NEW: KeyGraph SDPA path (single call; replaces old manual attention+rescue) ===
            attn_output = _sdpa_with_keygraph(
                query_states,          # [B,H,Q,D]  (RoPE already applied above)
                key_states,            # [B,H,S,D]
                value_states,          # [B,H,S,D]
                attention_mask,        # [B,1,Q,S] or None (additive)
                keygraph_cache,        # captured from outer scope
                layer_idx,             # int
                epsilon_var=getattr(self, "epsilon_var", 1e-3),
            )

            # SDPA path doesn't return per-token weights; keep API stable:
            attn_weights = None
            present_key_value = (key_states, value_states) if use_cache else None
            
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