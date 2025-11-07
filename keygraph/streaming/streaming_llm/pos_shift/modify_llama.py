import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.utils.checkpoint

import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import Cache, DynamicCache
import types

__all__ = ["enable_llama_pos_shift_attention"]


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def llama_pos_shift_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    past_key_values: Optional[Cache] = None,  # Support new Cache objects
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    # Handle both parameter names for compatibility
    if past_key_values is not None:
        past_key_value = past_key_values
    
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    # FIXED: Handle both Cache objects and tuples
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if isinstance(past_key_value, Cache):
            # For Cache objects, get the length properly
            try:
                cache_len = past_key_value.get_seq_length(self.layer_idx)
                kv_seq_len += cache_len
            except:
                # Cache is empty, no additional length
                pass
        elif isinstance(past_key_value, (tuple, list)) and len(past_key_value) > 0:
            # For tuple format
            kv_seq_len += past_key_value[0].shape[-2]
    
    # Get rotary embeddings - use seq_len as INTEGER (not tensor!)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    
    # Generate position_ids if not provided
    if position_ids is None:
        if cache_position is not None:
            position_ids = cache_position.unsqueeze(0)
        else:
            position_ids = torch.arange(
                kv_seq_len - q_len, kv_seq_len, dtype=torch.long, device=hidden_states.device
            )
            position_ids = position_ids.unsqueeze(0)
    
    ### Shift Pos: query pos is min(cache_size, idx)
    query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
    ###

    # FIXED: Handle both Cache objects and tuples for concatenation
    if past_key_value is not None:
        if isinstance(past_key_value, Cache):
            # For Cache objects, use update method which handles concatenation
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
        elif isinstance(past_key_value, (tuple, list)) and len(past_key_value) > 0:
            # For tuple format, manual concatenation
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

    # Prepare return value for cache
    if use_cache:
        if isinstance(past_key_value, Cache):
            past_key_value_return = past_key_value
        else:
            past_key_value_return = (key_states, value_states)
    else:
        past_key_value_return = None

    ### Shift Pos: key pos is the pos in cache
    key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
    key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)
    ###

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value_return


def enable_llama_pos_shift_attention(model, recent_size=1020):
    """
    Enable position-shifted attention for StreamingLLM.
    
    Args:
        model: The LLaMA model to modify
        recent_size: Size of the recent token window (not used in this version but kept for compatibility)
    """
    # Add layer_idx to each attention module for Cache compatibility
    for idx, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = idx
    
    # Recursively replace forward methods
    def replace_forward(module):
        for name, submodule in module._modules.items():
            if len(list(submodule.children())) > 0:
                replace_forward(submodule)
            
            if isinstance(submodule, LlamaAttention):
                module._modules[name].forward = types.MethodType(
                    llama_pos_shift_attention_forward, module._modules[name]
                )
    
    replace_forward(model)