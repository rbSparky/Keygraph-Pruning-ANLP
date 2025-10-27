import torch
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from keygraph.eval_metrics import evaluate_prediction
from keygraph.dataset.data import NarrativeQAAdapter
from keygraph.logging_utils import log_metrics_to_csv
from keygraph.streaming.streaming_llm.enable_streaming_llm import enable_streaming_llm
from keygraph.models import load_model_and_tokenizer
from tqdm import tqdm


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

def streaming_llm_generate(model, tokenizer, prompt, max_new_tokens, kv_cache):
    """
    Generates text using StreamingLLM by processing the prompt in chunks to avoid OOM.
    """
    device = model.device
    metrics = {}
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    start_time = time.perf_counter()

    # Initialize with None - let the model create the cache structure
    past_key_values = None
    prompt_chunk_size = 512
    
    # Process prompt in chunks
    num_chunks = (input_ids.shape[1] + prompt_chunk_size - 1) // prompt_chunk_size
    
    for i in range(0, input_ids.shape[1], prompt_chunk_size):
        chunk = input_ids[:, i:i + prompt_chunk_size]
        
        with torch.no_grad():
            # For the first chunk, past_key_values will be None
            # For subsequent chunks, we pass the cached values
            outputs = model(
                input_ids=chunk, 
                past_key_values=past_key_values, 
                use_cache=True
            )
        
        # Apply streaming KV cache management after the model forward pass
        # This keeps only the recent tokens and attention sinks
        past_key_values = outputs.past_key_values
        
        # Only apply kv_cache eviction starting from the second chunk
        # This ensures the cache is properly initialized first
        if past_key_values is not None and i > 0:
            past_key_values = kv_cache(past_key_values)

    # After processing all prompt chunks, apply kv_cache one more time
    if past_key_values is not None:
        past_key_values = kv_cache(past_key_values)

    # Generate new tokens
    generated_ids = []
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=pred_token_idx, 
                past_key_values=past_key_values, 
                use_cache=True
            )
        
        past_key_values = kv_cache(outputs.past_key_values)
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        
        if pred_token_idx.item() == tokenizer.eos_token_id:
            break

    end_time = time.perf_counter()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Calculate metrics
    total_time = end_time - start_time
    tokens_generated = len(generated_ids)
    
    metrics['latency_per_token'] = total_time / tokens_generated if tokens_generated > 0 else 0
    metrics['tokens_per_second'] = tokens_generated / total_time if total_time > 0 else 0
    metrics['tokens_generated'] = tokens_generated
    metrics['peak_vram_mb'] = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

    return generated_text, metrics


def main(args):
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_dir, "cuda" if torch.cuda.is_available() else "cpu")

    # Enable StreamingLLM
    print(f"Enabling StreamingLLM with start_size={args.start_size} and recent_size={args.recent_size}")
    kv_cache = enable_streaming_llm(
        model, start_size=args.start_size, recent_size=args.recent_size
    )

    # Load NarrativeQA dataset
    dataset_adapter = NarrativeQAAdapter(args.dataset_dir)
    samples = dataset_adapter.get_samples("test", args.num_samples)

    output_dir = "runs/baseline_streamingllm_narrativeqa"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "results.csv")

    # Store all metrics for averaging
    all_metrics = []

    for i, sample in enumerate(tqdm(samples, desc="Processing Samples")):
        print(f"\n--- Processing sample {i + 1}/{len(samples)} ---")
        prompt = dataset_adapter.format_prompt(sample)
        
        # Generate text
        generated_text, metrics = streaming_llm_generate(
            model, tokenizer, prompt, args.max_new_tokens, kv_cache
        )
        
        # Calculate perplexity on generated text (optional)
        if args.compute_ppl:
            print("Calculating perplexity on generated text...")
            ppl = compute_perplexity(model, tokenizer, generated_text)
            metrics['perplexity'] = ppl
            print(f"Perplexity: {ppl:.2f}")
        
        # Evaluate against ground truth answers
        # NarrativeQA has multiple valid answers
        ground_truth_answers = sample["answers"]
        eval_scores = evaluate_prediction("qa", generated_text, ground_truth_answers)
        metrics.update(eval_scores)
        
        # Store metrics
        metrics["sample_id"] = sample.get("id", f"narrativeqa_{i}")
        metrics["baseline"] = "streamingllm"
        metrics["model"] = args.model_dir
        metrics["max_new_tokens"] = args.max_new_tokens
        metrics["streaming_start_size"] = args.start_size
        metrics["streaming_recent_size"] = args.recent_size
        
        # Store for averaging
        all_metrics.append(metrics.copy())
        
        log_metrics_to_csv(csv_file, metrics)
        
        # Print per-sample results
        print(f"Tokens/sec: {metrics['tokens_per_second']:.2f}")
        print(f"Latency/token: {metrics['latency_per_token']:.4f}s")
        print(f"Peak VRAM: {metrics['peak_vram_mb']:.2f} MB")
        if 'exact_match' in metrics: print(f"Exact Match: {metrics['exact_match']:.4f}")
        if 'f1' in metrics: print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Question: {sample['question'][:100]}...")
        print(f"Generated answer: {generated_text[:100]}...")

    # Calculate and print average metrics
    print("\n" + "="*80)
    print("AVERAGE METRICS ACROSS ALL SAMPLES")
    print("="*80)
    
    metrics_to_average = [
        'tokens_per_second', 'latency_per_token', 'peak_vram_mb',
        'exact_match', 'f1', 'tokens_generated'
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
    print(f"Average Exact Match: {avg_metrics.get('exact_match', 0):.4f}")
    print(f"Average F1 Score: {avg_metrics.get('f1', 0):.4f}")
    print(f"Average Tokens Generated: {avg_metrics.get('tokens_generated', 0):.2f}")
    if args.compute_ppl:
        print(f"Average Perplexity: {avg_metrics.get('perplexity', 0):.2f}")
    
    # Save average metrics
    avg_csv_file = os.path.join(output_dir, "average_results.csv")
    avg_metrics["baseline"] = "streamingllm"
    avg_metrics["model"] = args.model_dir
    avg_metrics["num_samples"] = len(samples)
    avg_metrics["streaming_start_size"] = args.start_size
    avg_metrics["streaming_recent_size"] = args.recent_size
    log_metrics_to_csv(avg_csv_file, avg_metrics)
    
    print(f"\nDetailed results saved to: {csv_file}")
    print(f"Average metrics saved to: {avg_csv_file}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run StreamingLLM on NarrativeQA")
    parser.add_argument("--model_dir", required=True, help="Path to model directory")
    parser.add_argument("--dataset_dir", required=True, help="Path or name of NarrativeQA dataset (e.g., 'narrativeqa')")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max new tokens for answer generation")
    parser.add_argument("--start_size", type=int, default=4, help="Number of attention sinks")
    parser.add_argument("--recent_size", type=int, default=1020, help="Size of the recent token window")
    parser.add_argument("--compute_ppl", action="store_true", help="Enable perplexity calculation")
    args = parser.parse_args()
    main(args)