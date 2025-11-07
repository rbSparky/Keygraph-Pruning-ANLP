import torch
import time
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from keygraph.eval_metrics import evaluate_prediction
from keygraph.dataset.data import NarrativeQAAdapter
from keygraph.logging_utils import log_metrics_to_csv
from nacl_eviction import NACLEviction
from keygraph.models import load_model_and_tokenizer
from tqdm import tqdm
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from typing import Tuple, Optional, Callable

@torch.no_grad()
def compute_perplexity(model, tokenizer, text, stride=512):
    """
    Computes the perplexity of a model on a given text.
    Uses a sliding window approach for long texts.
    """
    device = model.device
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    
    max_length = model.config.max_position_embeddings
    
    nlls = []
    prev_end_loc = 0

    if seq_len == 0:
        return 0.0

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

@torch.no_grad()
def nacl_generate(model, tokenizer, prompt, max_new_tokens, nacl_eviction, args):
    """
    Generates text using NACL KV cache eviction.
    
    MODIFIED: This version is now aware of the model's 2048 limit.
    It loads the full prompt, logs its real length, and then
    truncates it to the 2048-token limit, just like the 'Full' baseline.
    """
    device = model.device
    metrics = {}
    
    # --- MODIFICATION START ---
    # 1. Tokenize the prompt WITHOUT TRUNCATION to see its real length
    print("Tokenizing full prompt (no truncation)...")
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False,
        max_length=None,
        padding=False
    ).to(device)
    
    input_ids = inputs.input_ids
    prompt_tokens = input_ids.shape[1]
    print(f"Full prompt token count: {prompt_tokens}")

    # 2. Check against model's hard limit and truncate
    max_pos = model.config.max_position_embeddings
    if prompt_tokens > max_pos:
        print(f"WARNING: Prompt ({prompt_tokens} tokens) exceeds model's max position embeddings ({max_pos}).")
        # Truncate to fit the model's absolute limit
        input_ids = input_ids[:, -max_pos:]
        print(f"Truncated to {input_ids.shape[1]} tokens to fit model max length.")
    
    original_cache_size = input_ids.shape[1] # This will be 2048 for long docs
    # --- MODIFICATION END ---
    
    
    # Performance measurement setup
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    start_time = time.perf_counter()

    # ===== PHASE 1: ENCODE FULL (TRUNCATED) PROMPT =====
    past_key_values = None
    prompt_chunk_size = args.chunk_size
    
    print(f"Processing prompt with NACL (length: {input_ids.shape[1]} tokens)...")
    
    # Process prompt in chunks WITHOUT eviction
    for i in range(0, input_ids.shape[1], prompt_chunk_size):
        chunk = input_ids[:, i:min(i + prompt_chunk_size, input_ids.shape[1])]
        
        with torch.no_grad():
            outputs = model(
                input_ids=chunk,
                past_key_values=past_key_values,
                use_cache=True
            )
        
        # Update past_key_values WITHOUT eviction during prefill
        past_key_values = outputs.past_key_values
        
    # Convert DynamicCache to tuple for NACL eviction
    if hasattr(past_key_values, 'to_legacy_cache'):
        past_key_values_tuple = past_key_values.to_legacy_cache()
    else:
        past_key_values_tuple = past_key_values # Assume it's already a tuple
    
    print(f"Prefill complete. Full KV cache size: {past_key_values_tuple[0][0].shape[2]} tokens")
    
    # ===== PHASE 2: APPLY NACL EVICTION TO FULL CACHE =====
    print("Applying NACL eviction to full KV cache...")
    evicted_cache_tuple = nacl_eviction.evict(
        past_key_values=past_key_values_tuple,
        current_length=input_ids.shape[1]
    )
    
    if evicted_cache_tuple is not None:
        evicted_cache_size = evicted_cache_tuple[0][0].shape[2]
        print(f"After eviction, KV cache size: {evicted_cache_size} tokens")
        
        # Convert tuple back to DynamicCache for model compatibility
        if hasattr(DynamicCache, 'from_legacy_cache'):
             past_key_values = DynamicCache.from_legacy_cache(evicted_cache_tuple)
        else:
             past_key_values = evicted_cache_tuple # Fallback
    else:
        print("Warning: eviction returned None!")
        evicted_cache_size = 0
        past_key_values = None # Reset cache

    # ===== PHASE 3: GENERATION WITH COMPRESSED CACHE =====
    generated_ids = []
    
    # Handle edge case where cache is empty
    if past_key_values is None:
        print("Error: KV Cache is None after eviction. Aborting generation.")
        return "", {}
        
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True
            )
        
        current_seq_len = original_cache_size + step + 1
        
        # Convert to tuple, evict, convert back
        if hasattr(outputs.past_key_values, 'to_legacy_cache'):
            pkv_tuple = outputs.past_key_values.to_legacy_cache()
        else:
            pkv_tuple = outputs.past_key_values
            
        # During generation, optionally continue eviction
        if args.evict_during_generation:
            evicted_tuple = nacl_eviction.evict(
                past_key_values=pkv_tuple,
                current_length=current_seq_len
            )
        else:
            # If not evicting, just pass the cache through
            evicted_tuple = pkv_tuple

        # Convert back to DynamicCache
        if hasattr(DynamicCache, 'from_legacy_cache'):
            past_key_values = DynamicCache.from_legacy_cache(evicted_tuple)
        else:
            past_key_values = evicted_tuple
        
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())

        if pred_token_idx.item() == tokenizer.eos_token_id:
            break

    end_time = time.perf_counter()
    
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Calculate metrics
    total_time = end_time - start_time
    tokens_generated = len(generated_ids)
    
    metrics['tokens_per_second'] = tokens_generated / total_time if total_time > 0 else 0
    metrics['latency_per_token'] = total_time / tokens_generated if tokens_generated > 0 else 0
    metrics['tokens_generated'] = tokens_generated
    metrics['total_time'] = total_time
    
    if torch.cuda.is_available():
        metrics['peak_vram_mb'] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        metrics['peak_vram_mb'] = 0.0
    
    # Calculate compression ratio based on the final evicted cache size
    metrics['compression_ratio'] = evicted_cache_size / original_cache_size if original_cache_size > 0 else 1.0
    metrics['cache_size'] = evicted_cache_size # The size after the *initial* eviction

    return generated_text, metrics

def main(args):
    # Load model and tokenizer
    print(f"Loading model from {args.model_dir}...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_dir, 
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Initialize NACL eviction policy
    print(f"Initializing NACL...")
    
    nacl_eviction = NACLEviction(
        model=model,
        proxy_tokens_ratio=args.proxy_tokens_ratio,
        proxy_token_keep_ratio=args.proxy_token_keep_ratio,
        random_token_keep_ratio=args.random_token_keep_ratio,
        token_protect_ratio=args.token_protect_ratio,
        sink_tokens=args.sink_tokens,
        min_eviction_seqlen=args.min_eviction_seqlen
    )

    # Load dataset (NarrativeQA)
    dataset_adapter = NarrativeQAAdapter(args.dataset_dir)
    samples = dataset_adapter.get_samples("test", args.num_samples)

    # Setup output directory
    output_dir = f"runs/nacl_narrativeqa"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "results.csv")

    all_metrics = []  # Store all metrics for averaging

    # Process each sample
    for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
        print(f"\n{'='*60}")
        print(f"Processing sample {i + 1}/{len(samples)}")
        print(f"{'='*60}")
        
        prompt = dataset_adapter.format_prompt(sample)
        print(f"Prompt length: {len(tokenizer.encode(prompt))} tokens")
        print(f"Question: {sample['question'][:100]}...")
        
        # Generate with NACL
        generated_text, metrics = nacl_generate(
            model, tokenizer, prompt, args.max_new_tokens, nacl_eviction, args
        )
        
        # NarrativeQA has multiple valid answers
        ground_truth = sample["answers"]
        
        # Calculate perplexity on GENERATED text
        if args.compute_ppl and generated_text:
            print("Calculating perplexity on generated text...")
            ppl = compute_perplexity(model, tokenizer, generated_text)
            metrics['perplexity'] = ppl
            print(f"Perplexity: {ppl:.2f}")

        # Evaluate generation quality (QA task)
        eval_scores = evaluate_prediction("qa", generated_text, ground_truth)
        metrics.update(eval_scores)
        
        # Add metadata
        metrics["sample_id"] = sample.get("id", f"narrativeqa_{i}")
        metrics["baseline"] = "nacl"
        metrics["model"] = args.model_dir
        metrics["max_new_tokens"] = args.max_new_tokens
        metrics["proxy_tokens_ratio"] = args.proxy_tokens_ratio
        metrics["proxy_token_keep_ratio"] = args.proxy_token_keep_ratio
        metrics["random_token_keep_ratio"] = args.random_token_keep_ratio
        metrics["sink_tokens"] = args.sink_tokens
        
        # Store for averaging
        all_metrics.append(metrics.copy())
        
        # Log to CSV
        log_metrics_to_csv(csv_file, metrics)
        
        # Print summary
        print(f"\n--- Results ---")
        print(f"Tokens/sec: {metrics['tokens_per_second']:.2f}")
        print(f"Latency/token: {metrics.get('latency_per_token', 0):.4f}s")
        print(f"Peak VRAM: {metrics['peak_vram_mb']:.2f} MB")
        print(f"Compression ratio: {metrics['compression_ratio']:.2f}x")
        print(f"Cache size: {metrics['cache_size']} tokens")
        if 'exact_match' in metrics:
            print(f"Exact Match: {metrics['exact_match']:.4f}")
        if 'f1' in metrics:
            print(f"F1: {metrics['f1']:.4f}")
        print(f"\nGenerated answer: {generated_text[:200]}...")
        print(f"Ground truth answers: {ground_truth[:2]}")  # Show first 2 answers
        print(f"{'='*60}\n")

    # Calculate and print average metrics
    if all_metrics:
        print("\n" + "=" * 80)
        print("AVERAGE METRICS ACROSS ALL SAMPLES")
        print("=" * 80)
        
        metrics_to_average = [
            'tokens_per_second', 'latency_per_token', 'peak_vram_mb',
            'compression_ratio', 'cache_size', 'tokens_generated',
            'exact_match', 'f1'
        ]
        if args.compute_ppl:
            metrics_to_average.append('perplexity')
        
        avg_metrics = {}
        for metric_name in metrics_to_average:
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            if values:
                avg_metrics[metric_name] = sum(values) / len(values)
        
        # Print averages
        print(f"Average Tokens/sec: {avg_metrics.get('tokens_per_second', 0):.2f}")
        print(f"Average Latency/token: {avg_metrics.get('latency_per_token', 0):.4f}s")
        print(f"Average Peak VRAM: {avg_metrics.get('peak_vram_mb', 0):.2f} MB")
        print(f"Average Compression Ratio: {avg_metrics.get('compression_ratio', 0):.2f}x")
        print(f"Average Cache Size: {avg_metrics.get('cache_size', 0):.2f} tokens")
        print(f"Average Tokens Generated: {avg_metrics.get('tokens_generated', 0):.2f}")
        print(f"Average Exact Match: {avg_metrics.get('exact_match', 0):.4f}")
        print(f"Average F1: {avg_metrics.get('f1', 0):.4f}")
        
        if args.compute_ppl:
            print(f"Average Perplexity: {avg_metrics.get('perplexity', 0):.2f}")
        
        # Save average metrics
        avg_csv_file = os.path.join(output_dir, "average_results.csv")
        avg_metrics["baseline"] = "nacl"
        avg_metrics["model"] = args.model_dir
        avg_metrics["num_samples"] = len(samples)
        avg_metrics["proxy_tokens_ratio"] = args.proxy_tokens_ratio
        avg_metrics["proxy_token_keep_ratio"] = args.proxy_token_keep_ratio
        avg_metrics["random_token_keep_ratio"] = args.random_token_keep_ratio
        avg_metrics["sink_tokens"] = args.sink_tokens
        log_metrics_to_csv(avg_csv_file, avg_metrics)
        
        print(f"\nDetailed results saved to: {csv_file}")
        print(f"Average metrics saved to: {avg_csv_file}")
        print("=" * 80)

    print(f"\nAll processing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NACL on NarrativeQA")
    
    # Model and dataset
    parser.add_argument("--model_dir", required=True, help="Path to model directory")
    parser.add_argument("--dataset_dir", required=True, help="Path to NarrativeQA dataset directory")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to process")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max new tokens to generate")
    
    # NACL-specific parameters (matching actual implementation)
    parser.add_argument("--proxy_tokens_ratio", type=float, default=0.01,
                        help="Ratio of proxy tokens (default: 0.01 = 1%%)")
    parser.add_argument("--proxy_token_keep_ratio", type=float, default=0.12,
                        help="Ratio kept by proxy eviction (default: 0.12 = 12%%)")
    parser.add_argument("--random_token_keep_ratio", type=float, default=0.07,
                        help="Ratio kept by random eviction (default: 0.07 = 7%%)")
    parser.add_argument("--token_protect_ratio", type=float, default=0.01,
                        help="Ratio of protected tokens (default: 0.01 = 1%%)")
    parser.add_argument("--sink_tokens", type=int, default=256,
                        help="Number of sink tokens to always keep (default: 256)")
    parser.add_argument("--min_eviction_seqlen", type=int, default=2048,
                        help="Minimum sequence length before eviction (default: 2048)")
    parser.add_argument("--evict_during_generation", action="store_true",
                        help="Continue eviction during generation phase")
    
    # Processing parameters
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Chunk size for processing long prompts")
    parser.add_argument("--compute_ppl", action="store_true",
                        help="Enable perplexity calculation")
    
    args = parser.parse_args()
    main(args)