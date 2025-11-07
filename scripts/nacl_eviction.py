"""
NACL KV Cache Eviction - Fixed GQA/MQA Support
Based on: https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2024-NACL

This implements the actual NACL algorithm with proper GQA/MQA support.

IMPORTANT: NACL eviction should be applied AFTER the full prompt is processed,
not during chunked processing. The model expects continuous KV cache during
the prefill phase.
"""

import torch
import torch.nn.functional as F
import numpy as np
import scipy.special
from typing import Optional, Tuple

# Try to import flash_attn, fallback to manual implementation if not available
try:
    from flash_attn import flash_attn_func
    from flash_attn.flash_attn_interface import _flash_attn_forward
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("Warning: flash_attn not available. Using fallback attention implementation.")


class NACLEviction:
    """
    NACL: A General and Effective KV Cache Eviction Framework
    
    Actual implementation matching the PaddlePaddle code with PyTorch.
    Fixed to properly handle GQA/MQA models.
    """
    
    def __init__(
        self,
        model,
        proxy_tokens_ratio: float = 0.01,
        proxy_token_keep_ratio: float = 0.12,
        random_token_keep_ratio: float = 0.07,
        token_protect_ratio: float = 0.01,
        sink_tokens: int = 256,
        min_eviction_seqlen: int = 2048
    ):
        """
        Initialize NACL eviction policy.
        
        Args:
            model: The language model
            proxy_tokens_ratio: Ratio of tokens used as proxies for scoring
            proxy_token_keep_ratio: Ratio of tokens kept by proxy eviction
            random_token_keep_ratio: Ratio of tokens kept by random eviction
            token_protect_ratio: Ratio of recent tokens to protect
            sink_tokens: Number of initial attention sink tokens
            min_eviction_seqlen: Minimum sequence length to trigger eviction
        """
        self.model = model
        self.proxy_tokens_ratio = proxy_tokens_ratio
        self.proxy_token_keep_ratio = proxy_token_keep_ratio
        self.random_token_keep_ratio = random_token_keep_ratio
        self.token_protect_ratio = token_protect_ratio
        self.sink_tokens = sink_tokens
        self.min_eviction_seqlen = min_eviction_seqlen
        
        self.num_layers = model.config.num_hidden_layers
        self.num_attention_heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads
        
        # Handle GQA/MQA: num_key_value_heads might be different from num_attention_heads
        self.num_kv_heads = getattr(model.config, 'num_key_value_heads', self.num_attention_heads)
        
        print(f"NACL initialized (ACL 2024):")
        print(f"  - proxy_tokens_ratio: {proxy_tokens_ratio} ({proxy_tokens_ratio*100:.1f}%)")
        print(f"  - proxy_token_keep_ratio: {proxy_token_keep_ratio} ({proxy_token_keep_ratio*100:.1f}%)")
        print(f"  - random_token_keep_ratio: {random_token_keep_ratio} ({random_token_keep_ratio*100:.1f}%)")
        print(f"  - sink_tokens: {sink_tokens}")
        print(f"  - num_attention_heads: {self.num_attention_heads}")
        print(f"  - num_kv_heads: {self.num_kv_heads}")
        print(f"  - FlashAttention available: {HAS_FLASH_ATTN}")
    
    def compute_proxy_attention_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention scores using proxy tokens.
        This is the core of NACL's proxy-token eviction strategy.
        
        Args:
            query: Proxy queries [batch, num_kv_heads, num_proxy, head_dim]
            key: All keys [batch, num_kv_heads, seq_len, head_dim]
            value: All values [batch, num_kv_heads, seq_len, head_dim]
            
        Returns:
            Reduced attention scores [batch, num_kv_heads, seq_len]
        """
        batch_size, num_kv_heads, num_proxy, head_dim = query.shape
        seq_len = key.shape[2]
        
        # Always use manual attention for consistency
        # FlashAttention may have compatibility issues with proxy token scoring
        reduced_scores = self._manual_attention_scores(query, key)
        
        return reduced_scores
    
    def _manual_attention_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor
    ) -> torch.Tensor:
        """
        Manual attention score computation.
        
        Args:
            query: [batch, num_kv_heads, num_proxy, head_dim]
            key: [batch, num_kv_heads, seq_len, head_dim]
            
        Returns:
            scores: [batch, num_kv_heads, seq_len]
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # [batch, num_kv_heads, num_proxy, seq_len]
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Average over proxy queries
        reduced_scores = attn_weights.mean(dim=2)
        # [batch, num_kv_heads, seq_len]
        
        return reduced_scores
    
    def evict(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        attention_scores: Optional[Tuple[torch.Tensor, ...]] = None,
        current_length: Optional[int] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        Apply NACL eviction to KV cache.
        
        This is the main eviction function that implements:
        1. Proxy-token based eviction (top-k by attention scores)
        2. Random eviction (probability-weighted sampling)
        3. Protection of sink and recent tokens
        
        Args:
            past_key_values: Tuple of (key, value) pairs for each layer
            attention_scores: Not used (we compute our own)
            current_length: Current sequence length
            
        Returns:
            Evicted KV cache
        """
        if past_key_values is None:
            return None
        
        # Get sequence length from first layer
        # Shape: [batch, num_kv_heads, seq_len, head_dim]
        first_key = past_key_values[0][0]
        seq_len = first_key.shape[2]
        num_kv_heads = first_key.shape[1]
        
        # Only evict if sequence is long enough
        if seq_len <= self.min_eviction_seqlen:
            return past_key_values
        
        # Calculate token budgets
        proxy_tokens = max(1, int(self.proxy_tokens_ratio * seq_len))  # At least 1 proxy token
        recent_protect_tokens = max(0, int(self.token_protect_ratio * seq_len))
        proxy_keep_tokens = int(self.proxy_token_keep_ratio * seq_len)
        random_keep_tokens = int(self.random_token_keep_ratio * seq_len)
        
        kvcache_budget = self.sink_tokens + recent_protect_tokens + proxy_keep_tokens + random_keep_tokens
        
        # Check if eviction is needed
        evict_tokens = seq_len - kvcache_budget
        if evict_tokens <= 0:
            return past_key_values
        
        print(f"NACL evicting {evict_tokens} tokens (keeping {kvcache_budget}/{seq_len} = {100*kvcache_budget/seq_len:.1f}%)")
        print(f"  - Sink tokens: {self.sink_tokens}")
        print(f"  - Recent protect: {recent_protect_tokens}")
        print(f"  - Proxy keep: {proxy_keep_tokens}")
        print(f"  - Random keep: {random_keep_tokens}")
        print(f"  - Proxy tokens: {proxy_tokens}")
        print(f"  - Num KV heads: {num_kv_heads}")
        
        # Start position of recent tokens to protect
        proxy_start_pos = seq_len - recent_protect_tokens if recent_protect_tokens > 0 else seq_len
        
        # Apply eviction layer by layer
        evicted_kv = []
        
        for layer_idx, (keys, values) in enumerate(past_key_values):
            # keys/values shape: [batch, num_kv_heads, seq_len, head_dim]
            device = keys.device
            batch_size = keys.shape[0]
            
            # Verify the number of KV heads matches
            assert keys.shape[1] == num_kv_heads, f"Inconsistent KV heads: {keys.shape[1]} vs {num_kv_heads}"
            
            # Step 1: Select proxy tokens (last N tokens)
            # Make sure we don't try to use more proxy tokens than available
            actual_proxy_tokens = min(proxy_tokens, seq_len)
            proxy_queries = keys[:, :, -actual_proxy_tokens:, :]
            
            print(f"  Layer {layer_idx}: Using {actual_proxy_tokens} proxy tokens from {seq_len} total")
            print(f"    Keys shape: {keys.shape}, Proxy queries shape: {proxy_queries.shape}")
            
            # Step 2: Compute attention scores using proxy tokens
            proxy_scores = self.compute_proxy_attention_scores(
                proxy_queries, keys, values
            )
            # proxy_scores: [batch, num_kv_heads, seq_len]
            
            # Verify shape
            assert proxy_scores.shape == (batch_size, num_kv_heads, seq_len), \
                f"Wrong proxy_scores shape: {proxy_scores.shape}, expected ({batch_size}, {num_kv_heads}, {seq_len})"
            
            # Step 3: Apply eviction per head (head-wise eviction)
            indices_list = []
            
            # Always keep these indices
            sink_keep_idx = np.arange(self.sink_tokens)
            if recent_protect_tokens > 0:
                recent_keep_idx = np.arange(proxy_start_pos, seq_len)
            else:
                recent_keep_idx = np.array([], dtype=np.int64)
            
            # Loop over the KV heads
            for head_idx in range(num_kv_heads):
                # Debug: verify head_idx is valid
                if head_idx >= proxy_scores.shape[1]:
                    raise ValueError(f"head_idx {head_idx} >= proxy_scores.shape[1] {proxy_scores.shape[1]}")
                
                # Get scores for this head (exclude protected tokens)
                head_scores = proxy_scores[0, head_idx, self.sink_tokens:proxy_start_pos].cpu().numpy()
                
                # Handle edge case where no tokens available for eviction
                if len(head_scores) == 0:
                    # Just keep sink and recent tokens
                    head_indices = np.concatenate([sink_keep_idx, recent_keep_idx])
                    indices_list.append(head_indices)
                    continue
                
                # Step 3a: Proxy-based eviction - keep top-k tokens
                actual_proxy_keep = min(proxy_keep_tokens, len(head_scores))
                topk_indices = np.argsort(head_scores)[-actual_proxy_keep:]
                proxy_keep_idx = topk_indices + self.sink_tokens
                
                # Step 3b: Random eviction from remaining tokens
                # Get indices not selected by proxy eviction
                available_indices = np.delete(
                    np.arange(len(head_scores)),
                    topk_indices
                )
                
                if len(available_indices) > 0:
                    available_scores = head_scores[available_indices]
                    
                    # Sample using softmax probabilities
                    if random_keep_tokens > 0:
                        probs = scipy.special.softmax(available_scores)
                        num_to_sample = min(random_keep_tokens, len(available_indices))
                        random_keep_idx = np.random.choice(
                            available_indices,
                            size=num_to_sample,
                            replace=False,
                            p=probs
                        ) + self.sink_tokens
                    else:
                        random_keep_idx = np.array([], dtype=np.int64)
                else:
                    random_keep_idx = np.array([], dtype=np.int64)
                
                # Step 3c: Combine all indices and sort
                all_middle_tokens = np.concatenate([proxy_keep_idx, random_keep_idx])
                all_middle_tokens = np.sort(np.unique(all_middle_tokens))
                
                head_indices = np.concatenate([
                    sink_keep_idx,
                    all_middle_tokens,
                    recent_keep_idx
                ])
                
                # Verify we're within budget
                assert len(head_indices) <= kvcache_budget, \
                    f"Exceeded budget: {len(head_indices)} > {kvcache_budget}"
                
                indices_list.append(head_indices)
            
            # Convert to tensor: [1, num_kv_heads, actual_budget, 1]
            # Ensure all heads have the same number of indices by padding if necessary
            max_indices_len = max(len(indices) for indices in indices_list)
            
            # Pad shorter index arrays to match the longest
            padded_indices = []
            for head_indices in indices_list:
                if len(head_indices) < max_indices_len:
                    # Pad with the last valid index (safest option)
                    pad_value = head_indices[-1] if len(head_indices) > 0 else 0
                    padded = np.pad(
                        head_indices, 
                        (0, max_indices_len - len(head_indices)), 
                        mode='constant', 
                        constant_values=pad_value
                    )
                    padded_indices.append(padded)
                else:
                    padded_indices.append(head_indices)
            
            # Stack into array: [num_kv_heads, actual_budget]
            indices_array = np.stack(padded_indices, axis=0)
            actual_budget = indices_array.shape[1]
            
            # Convert to tensor and add batch and feature dimensions
            indices_tensor = torch.tensor(
                indices_array,
                device=device,
                dtype=torch.long
            ).unsqueeze(0).unsqueeze(-1)  # [1, num_kv_heads, actual_budget, 1]
            
            # Step 4: Apply indices to gather tokens
            # Expand indices to match key/value dimensions
            indices_expanded = indices_tensor.expand(
                batch_size, num_kv_heads, actual_budget, self.head_dim
            )
            
            # Gather keys and values
            keys_evicted = torch.gather(keys, dim=2, index=indices_expanded)
            values_evicted = torch.gather(values, dim=2, index=indices_expanded)
            
            evicted_kv.append((keys_evicted, values_evicted))
        
        return tuple(evicted_kv)


def enable_nacl(model, **kwargs):
    """
    Enable NACL for a model.
    
    Usage:
        nacl = enable_nacl(model, proxy_tokens_ratio=0.01, sink_tokens=256)
        past_key_values = nacl.evict(past_key_values)
    
    Args:
        model: The language model
        **kwargs: NACL parameters (see NACLEviction.__init__)
    
    Returns:
        NACLEviction instance
    """
    return NACLEviction(model=model, **kwargs)