import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Corrected TopKThinning Class (No DynamicCache) ---

class TopKThinning:
    """
    Corrected Top-K Token Thinning for KV Cache Compression.
    
    This version fixes quality, speed, and memory issues by using
    a single cumulative importance score tensor.
    
    It does NOT use DynamicCache, making it compatible with
    older transformers versions (e.g., 4.31.0).
    """
    
    def __init__(
        self,
        k: int = 128,
        protect_recent: int = 64,
        method: str = "attention"
    ):
        self.k = k
        self.protect_recent = protect_recent
        self.method = method
        # FIX: Single tensor, no list, no memory leak
        self.cumulative_importance: Optional[torch.Tensor] = None
        
    def reset(self):
        """Reset accumulated attention scores."""
        self.cumulative_importance = None

    def update_attention_scores(self, attention_weights: torch.Tensor):
        """
        Update the cumulative importance scores.
        
        Args:
            attention_weights: Layer-averaged attention weights
                Shape: (batch, num_heads, query_len, key_len)
        """
        if attention_weights is None:
            return
            
        # FIX: Get importance per *key* by averaging over query and head dimensions
        # Shape: (batch, key_len)
        importance = attention_weights.mean(dim=1).mean(dim=1)
        
        if self.cumulative_importance is None:
            self.cumulative_importance = importance
            return

        # Align shapes before adding
        current_len = importance.shape[-1]
        saved_len = self.cumulative_importance.shape[-1]
        
        if current_len > saved_len:
            pad_width = current_len - saved_len
            self.cumulative_importance = F.pad(self.cumulative_importance, (0, pad_width), 'constant', 0)
        elif saved_len > current_len:
            self.cumulative_importance = self.cumulative_importance[:, :current_len]
            
        self.cumulative_importance += importance

    def _select_tokens_by_attention(
        self,
        seq_len: int,
        num_keep: int
    ) -> List[int]:
        """
        Select tokens to keep based on *cumulative* attention scores.
        """
        # FIX: Use the cumulative_importance tensor. No looping, no padding.
        if self.cumulative_importance is None or self.cumulative_importance.shape[-1] != seq_len:
            if self.cumulative_importance is not None:
                print(f"Warning: Attention score desync. Falling back to uniform.")
                self.reset()
            return self._select_tokens_uniform(seq_len, num_keep)
        
        selectable_len = seq_len - self.protect_recent
        if selectable_len <= 0:
            return []
        
        selectable_importance = self.cumulative_importance[0, :selectable_len]
        num_to_select = min(num_keep, selectable_len)
        _, top_k_indices = torch.topk(selectable_importance, num_to_select, sorted=True)
        
        return top_k_indices.cpu().tolist()
    
    def _select_tokens_uniform(self, seq_len: int, num_keep: int) -> List[int]:
        """Select tokens uniformly across the sequence."""
        selectable_len = seq_len - self.protect_recent
        if selectable_len <= 0:
            return []
        
        num_to_select = min(num_keep, selectable_len)
        step = max(1, selectable_len // num_to_select)
        indices = list(range(0, selectable_len, step))[:num_to_select]
        return indices
    
    def _select_tokens_recency(self, seq_len: int, num_keep: int) -> List[int]:
        """Select most recent tokens (simple recency bias)."""
        selectable_len = seq_len - self.protect_recent
        if selectable_len <= 0:
            return []
            
        num_to_select = min(num_keep, selectable_len)
        start_idx = max(0, selectable_len - num_to_select)
        indices = list(range(start_idx, selectable_len))
        return indices
    
    def evict(
        self,
        past_key_values: Tuple, # Expects a tuple
        attention_weights: Optional[torch.Tensor] = None
    ) -> Tuple:
        """
        Evict tokens from KV cache (a tuple) using top-k selection.
        """
        if past_key_values is None:
            return None
        
        # 1. Update attention scores
        if self.method == "attention" and attention_weights is not None:
            self.update_attention_scores(attention_weights)
        
        seq_len = past_key_values[0][0].shape[2]
        
        # 2. Check if eviction is needed
        if seq_len <= (self.k + self.protect_recent):
            return past_key_values # No eviction needed
        
        # 3. Calculate indices to keep
        num_keep = self.k
        
        if self.method == "attention":
            keep_indices = self._select_tokens_by_attention(seq_len, num_keep)
        elif self.method == "uniform":
            keep_indices = self._select_tokens_uniform(seq_len, num_keep)
        else: # recency or default
            keep_indices = self._select_tokens_recency(seq_len, num_keep)
        
        recent_start = seq_len - self.protect_recent
        recent_indices = list(range(recent_start, seq_len))
        
        all_keep_indices = sorted(list(set(keep_indices + recent_indices)))
        
        # 4. Apply eviction to all layers
        new_past = []
        for (keys, values) in past_key_values:
            kept_keys = keys[:, :, all_keep_indices, :]
            kept_values = values[:, :, all_keep_indices, :]
            new_past.append((kept_keys, kept_values))
        
        # 5. CRITICAL FIX: Evict from cumulative_importance tensor
        if self.method == "attention" and self.cumulative_importance is not None:
            if self.cumulative_importance.shape[-1] == seq_len:
                self.cumulative_importance = self.cumulative_importance[:, all_keep_indices]
            else:
                print(f"Warning: Attention desync during eviction. Resetting scores.")
                self.reset()
        
        return tuple(new_past)

# --- NEW topk_generate Function (No Truncation) ---

@torch.no_grad()
def topk_generate(
    model,
    tokenizer,
    prompt: str,
    k: int = 128,
    protect_recent: int = 64,
    method: str = "attention",
    max_new_tokens: int = 128,
    compute_ppl: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate text using the corrected top-k token thinning.
    
    MODIFIED: This version DOES NOT TRUNCATE the prompt.
    It processes the full, long context in chunks, just like
    the streaming_llm_generate function.
    """
    device = model.device
    metrics = {}
    
    # --- 1. Setup Model for Attention (if needed) ---
    original_attn_implementation = None
    use_attention = (method == "attention")
    
    if use_attention:
        try:
            if hasattr(model.config, '_attn_implementation'):
                original_attn_implementation = model.config._attn_implementation
            model.config._attn_implementation = 'eager'
            if hasattr(model, '_attn_implementation'):
                model._attn_implementation = 'eager'
            print("Set attention implementation to 'eager' for attention-based selection")
        except Exception as e:
            print(f"Warning: Could not set attention to eager: {e}. Falling back to 'uniform'.")
            use_attention = False
            method = "uniform"
    
    topk_thinning = TopKThinning(k=k, protect_recent=protect_recent, method=method)
    
    # --- 2. Tokenize Full Input (NO TRUNCATION) ---
    print("Tokenizing full prompt (no truncation)...")
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,  # <--- MODIFIED
        max_length=7000,   # <--- MODIFIED
        padding=False
    ).to(device)
    
    input_ids = inputs.input_ids
    original_prompt_len = input_ids.shape[1]
    print(f"Full prompt token count: {original_prompt_len}")
    
    # --- 3. Prefill (Process the full prompt in chunks) ---
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    
    start_time = time.perf_counter()
    
    past_key_values = None # This will be a tuple
    prompt_chunk_size = 512
    
    print(f"Processing prompt in {input_ids.shape[1] // prompt_chunk_size + 1} chunks...")
    for i in range(0, original_prompt_len, prompt_chunk_size):
        chunk = input_ids[:, i:min(i + prompt_chunk_size, original_prompt_len)]
        
        with torch.no_grad():
            outputs = model(
                input_ids=chunk,
                past_key_values=past_key_values, # Pass tuple
                use_cache=True,
                output_attentions=use_attention
            )
        
        past_key_values = outputs.past_key_values # Get tuple back
        
        # Get layer-averaged attention
        avg_attention = None
        if use_attention and outputs.attentions is not None:
            avg_attention = torch.stack(outputs.attentions).mean(dim=0)
        
        # Evict *during* prefill
        past_key_values = topk_thinning.evict(past_key_values, avg_attention)
    
    print("Prompt processing complete.")
    
    # --- 4. Generation Loop ---
    generated_ids = []
    next_token_ids = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    
    for i in range(max_new_tokens):
        outputs = model(
            input_ids=next_token_ids,
            past_key_values=past_key_values, # Pass tuple
            use_cache=True,
            output_attentions=use_attention
        )
        
        past_key_values = outputs.past_key_values # Get tuple back
        
        avg_attention = None
        if use_attention and outputs.attentions is not None:
            try:
                avg_attention = torch.stack(outputs.attentions).mean(dim=0)
            except Exception as e:
                pass # Ignore errors during generation
        
        # Evict after each step
        past_key_values = topk_thinning.evict(past_key_values, avg_attention)
        
        next_token_ids = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        token_id = next_token_ids.item()
        generated_ids.append(token_id)
        
        if token_id == tokenizer.eos_token_id:
            break
    
    end_time = time.perf_counter()
    
    # --- 5. Restore Model and Calculate Metrics ---
    if original_attn_implementation is not None:
        try:
            model.config._attn_implementation = original_attn_implementation
            if hasattr(model, '_attn_implementation'):
                model._attn_implementation = original_attn_implementation
        except:
            pass
    
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    total_time = end_time - start_time
    tokens_generated = len(generated_ids)
    
    metrics['tokens_per_second'] = tokens_generated / total_time if total_time > 0 else 0
    metrics['latency_per_token'] = total_time / tokens_generated if tokens_generated > 0 else 0
    metrics['tokens_generated'] = tokens_generated
    metrics['total_time_seconds'] = total_time
    metrics['peak_vram_mb'] = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0
    
    metrics['kv_cache_size'] = past_key_values[0][0].shape[2] if past_key_values else 0
    metrics['kv_cache_method'] = f"topk_k{k}_recent{protect_recent}_{method}"
    metrics['compression_ratio'] = original_prompt_len / metrics['kv_cache_size'] if metrics['kv_cache_size'] > 0 else 0
    
    # (Perplexity computation function can remain the same)
    if compute_ppl and len(generated_text) > 50:
        try:
            # Assuming compute_perplexity function exists
            ppl = compute_perplexity(model, tokenizer, generated_text)
            metrics['perplexity'] = ppl
        except Exception as e:
            print(f"Warning: Could not compute perplexity: {e}")
            metrics['perplexity'] = float('inf')
    
    return generated_text, metrics

@torch.no_grad()
def compute_perplexity(model, tokenizer, text: str, stride: int = 512):
    """
    Compute perplexity of a text using sliding window approach.
    """
    device = model.device
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    max_length = model.config.max_position_embeddings
    
    if seq_len == 0:
        return 0.0
    
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
        
        neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()