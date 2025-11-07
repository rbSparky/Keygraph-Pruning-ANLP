import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Corrected TopKThinning Class (No DynamicCache) ---

class TopKThinning:
    """
    Corrected Top-K Token Thinning for KV Cache Compression.

    Uses a single cumulative importance score tensor and
    stays compatible with older transformers (e.g., 4.31.0).
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
        self.cumulative_importance: Optional[torch.Tensor] = None

    def reset(self):
        """Reset accumulated attention scores."""
        self.cumulative_importance = None

    def update_attention_scores(self, attention_weights: torch.Tensor):
        """
        Update cumulative importance scores from layer-averaged attention.

        Args:
            attention_weights: (batch, num_heads, query_len, key_len)
        """
        if attention_weights is None:
            return

        # Importance per key (avg over heads, then over queries)
        # (batch, key_len)
        importance = attention_weights.mean(dim=1).mean(dim=1)

        if self.cumulative_importance is None:
            self.cumulative_importance = importance
            return

        current_len = importance.shape[-1]
        saved_len = self.cumulative_importance.shape[-1]

        if current_len > saved_len:
            pad_width = current_len - saved_len
            self.cumulative_importance = F.pad(self.cumulative_importance, (0, pad_width), 'constant', 0)
        elif saved_len > current_len:
            self.cumulative_importance = self.cumulative_importance[:, :current_len]

        self.cumulative_importance += importance

    def _select_tokens_by_attention(self, seq_len: int, num_keep: int) -> List[int]:
        """Select tokens by cumulative attention scores."""
        if self.cumulative_importance is None or self.cumulative_importance.shape[-1] != seq_len:
            if self.cumulative_importance is not None:
                print("Warning: Attention score desync. Falling back to uniform.")
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
        """Uniform selection across the sequence (excluding protected tail)."""
        selectable_len = seq_len - self.protect_recent
        if selectable_len <= 0:
            return []
        num_to_select = min(num_keep, selectable_len)
        step = max(1, selectable_len // max(num_to_select, 1))
        indices = list(range(0, selectable_len, step))[:num_to_select]
        return indices

    def _select_tokens_recency(self, seq_len: int, num_keep: int) -> List[int]:
        """Select most recent tokens (excluding protected tail itself)."""
        selectable_len = seq_len - self.protect_recent
        if selectable_len <= 0:
            return []
        num_to_select = min(num_keep, selectable_len)
        start_idx = max(0, selectable_len - num_to_select)
        return list(range(start_idx, selectable_len))

    def evict(
        self,
        past_key_values: Tuple,
        attention_weights: Optional[torch.Tensor] = None
    ) -> Tuple:
        """Evict tokens from KV cache tuple using top-k selection."""
        if past_key_values is None:
            return None

        # Update scores if attention-based
        if self.method == "attention" and attention_weights is not None:
            self.update_attention_scores(attention_weights)

        seq_len = past_key_values[0][0].shape[2]

        # No eviction needed
        if seq_len <= (self.k + self.protect_recent):
            return past_key_values

        # Compute indices to keep
        num_keep = self.k
        if self.method == "attention":
            keep_indices = self._select_tokens_by_attention(seq_len, num_keep)
        elif self.method == "uniform":
            keep_indices = self._select_tokens_uniform(seq_len, num_keep)
        else:
            keep_indices = self._select_tokens_recency(seq_len, num_keep)

        recent_start = seq_len - self.protect_recent
        recent_indices = list(range(recent_start, seq_len))
        all_keep_indices = sorted(list(set(keep_indices + recent_indices)))

        # Apply to all layers
        new_past = []
        for (keys, values) in past_key_values:
            kept_keys = keys[:, :, all_keep_indices, :]
            kept_values = values[:, :, all_keep_indices, :]
            new_past.append((kept_keys, kept_values))

        # Keep cumulative_importance in sync
        if self.method == "attention" and self.cumulative_importance is not None:
            if self.cumulative_importance.shape[-1] == seq_len:
                self.cumulative_importance = self.cumulative_importance[:, all_keep_indices]
            else:
                print("Warning: Attention desync during eviction. Resetting scores.")
                self.reset()

        return tuple(new_past)

# ----------------- CUDA helpers -----------------

def _cuda_reset():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

def _cuda_peak_mb(kind: str = "reserved") -> float:
    if not torch.cuda.is_available():
        return 0.0
    if kind == "allocated":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return torch.cuda.max_memory_reserved() / (1024 ** 2)

# -------- Exact-phase VRAM instrumented generation --------

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
    Generate text using top-k token thinning.

    IMPORTANT CHANGES:
    - FULL prompt forward pass (NO CHUNKING here).
    - Exact VRAM peaks per phase:
        * peak_vram_prefill_mb: peak during prefill only
        * peak_vram_decode_mb: peak during decode only
      (We reset peak counters between phases to isolate them.)
    """
    device = model.device
    metrics: Dict[str, Any] = {}

    # Configure attention impl for accessing attentions if needed
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

    # Tokenize FULL prompt (no trunc / no chunking here)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=7000,
        padding=False
    ).to(device)

    input_ids = inputs.input_ids
    original_prompt_len = input_ids.shape[1]
    print(f"Full prompt token count: {original_prompt_len}")

    # ----------------- PREFILL (exact peak) -----------------
    _cuda_reset()
    prefill_start = time.perf_counter()

    outputs = model(
        input_ids=input_ids,
        past_key_values=None,
        use_cache=True,
        output_attentions=False
    )
    past_key_values = outputs.past_key_values

    # Compress once after prefill (now we have full attention over the context)
    past_key_values = topk_thinning.evict(past_key_values, attention_weights=None)

    prefill_time = time.perf_counter() - prefill_start
    prefill_peak_mb = _cuda_peak_mb("reserved")

    # ----------------- DECODE (exact peak) -----------------
    _cuda_reset()
    decode_start = time.perf_counter()

    generated_ids: List[int] = []
    next_token_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=next_token_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True
        )
        past_key_values = outputs.past_key_values

        if outputs.attentions is not None:
            try:
                avg_attention = torch.stack(outputs.attentions).mean(dim=0)
                topk_thinning.update_attention_scores(avg_attention)
            except Exception:
                pass

        # Evict after each decode step
        past_key_values = topk_thinning.evict(past_key_values, attention_weights=None)

        next_token_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token_id = int(next_token_ids.item())
        generated_ids.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break

    decode_time = time.perf_counter() - decode_start
    decode_peak_mb = _cuda_peak_mb("reserved")

    # ----------------- Finalize -----------------
    # Restore attention implementation if changed
    if original_attn_implementation is not None:
        try:
            model.config._attn_implementation = original_attn_implementation
            if hasattr(model, '_attn_implementation'):
                model._attn_implementation = original_attn_implementation
        except Exception:
            pass

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Metrics
    tokens_generated = len(generated_ids)
    kv_cache_size = past_key_values[0][0].shape[2] if past_key_values else 0

    metrics['prefill_tokens'] = original_prompt_len
    metrics['tokens_generated'] = tokens_generated

    # Decode-only speed/latency
    metrics['tokens_per_second'] = (tokens_generated / decode_time) if decode_time > 0 else 0.0
    metrics['latency_per_token'] = (decode_time / tokens_generated) if tokens_generated > 0 else 0.0

    # Phase times
    metrics['prefill_time_seconds'] = prefill_time
    metrics['decode_time_seconds'] = decode_time

    # Exact phase peaks
    metrics['peak_vram_prefill_mb'] = prefill_peak_mb
    metrics['peak_vram_decode_mb'] = decode_peak_mb
    metrics['peak_vram_total_mb'] = max(prefill_peak_mb, decode_peak_mb)

    metrics['kv_cache_size'] = kv_cache_size
    metrics['kv_cache_method'] = f"topk_k{k}_recent{protect_recent}_{method}"
    metrics['compression_ratio'] = (original_prompt_len / kv_cache_size) if kv_cache_size > 0 else 0.0

    if compute_ppl and len(generated_text) > 50:
        try:
            ppl = compute_perplexity(model, tokenizer, generated_text)
            metrics['perplexity'] = ppl
        except Exception as e:
            print(f"Warning: Could not compute perplexity: {e}")
            metrics['perplexity'] = float('inf')

    return generated_text, metrics

@torch.no_grad()
def compute_perplexity(model, tokenizer, text: str, stride: int = 512):
    """
    Compute perplexity of a text using a sliding window approach.
    """
    device = model.device
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    max_length = getattr(model.config, "max_position_embeddings", 2048)

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

        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()
