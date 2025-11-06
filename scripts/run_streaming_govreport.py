import torch
import time
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import your existing data adapter and evaluation functions
from keygraph.eval_metrics import evaluate_prediction
from keygraph.dataset.data import GovReportAdapter
from keygraph.logging_utils import log_metrics_to_csv
# Import from the StreamingLLM library
from keygraph.streaming.streaming_llm.enable_streaming_llm import enable_streaming_llm
from keygraph.models import load_model_and_tokenizer
from tqdm import tqdm
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    # Use the model's configured context window size
    max_length = model.config.max_position_embeddings
    nlls = []  # Negative log-likelihoods
    prev_end_loc = 0
    
    if seq_len == 0:
        return 0.0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # Mask tokens that are not being predicted
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()



@torch.no_grad()
def streaming_llm_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    kv_cache: Callable  # This is the function from enable_streaming_llm
) -> Tuple[str, dict]:
    """
    Generates text using StreamingLLM.
    MODIFIED: This version DOES NOT TRUNCATE the prompt.
    It processes the full, long context in chunks.
    """
    device = model.device
    metrics = {}
    
    # 1. Tokenize the prompt WITHOUT TRUNCATION
    print("Tokenizing full prompt (no truncation)...")
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=False,  # <--- MODIFIED
        max_length=None,   # <--- MODIFIED
        padding=False
    ).to(device)
    
    input_ids = inputs.input_ids
    print(f"Full prompt token count: {input_ids.shape[1]}")

    # 2. Reset VRAM and start timer
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    
    start_time = time.perf_counter()
    
    # 3. Process prompt in chunks (Prefill)
    past_key_values = None
    prompt_chunk_size = 512  # Process 512 tokens at a time
    
    print(f"Processing prompt in {input_ids.shape[1] // prompt_chunk_size + 1} chunks...")
    for i in range(0, input_ids.shape[1], prompt_chunk_size):
        chunk = input_ids[:, i:min(i + prompt_chunk_size, input_ids.shape[1])]
        
        with torch.no_grad():
            outputs = model(
                input_ids=chunk, 
                past_key_values=past_key_values, 
                use_cache=True
            )
        
        # Apply StreamingLLM eviction *after* each chunk
        past_key_values = kv_cache(outputs.past_key_values)
    
    print("Prompt processing complete.")
    
    # 4. Generation loop
    generated_ids = []
    # Get the next token prediction from the last chunk's output
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=pred_token_idx, 
                past_key_values=past_key_values, 
                use_cache=True
            )
        
        # Apply StreamingLLM eviction *after* each new token
        past_key_values = kv_cache(outputs.past_key_values)
        
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        token_id = pred_token_idx.item()
        generated_ids.append(token_id)
        
        if token_id == tokenizer.eos_token_id:
            break
    
    end_time = time.perf_counter()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # 5. Calculate metrics
    total_time = end_time - start_time
    tokens_generated = len(generated_ids)
    
    metrics['latency_per_token'] = total_time / tokens_generated if tokens_generated > 0 else 0
    metrics['tokens_per_second'] = tokens_generated / total_time if total_time > 0 else 0
    metrics['tokens_generated'] = tokens_generated
    metrics['peak_vram_mb'] = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0
    
    return generated_text, metrics

def main(args):
    # Clear CUDA cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_dir, 
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Enable StreamingLLM
    print(f"Enabling StreamingLLM with start_size={args.start_size} and recent_size={args.recent_size}")
    kv_cache = enable_streaming_llm(
        model, 
        start_size=args.start_size, 
        recent_size=args.recent_size
    )
    
    # Load dataset
    dataset_adapter = GovReportAdapter(args.dataset_dir)
    samples = dataset_adapter.get_samples("test", args.num_samples)  # Changed to "test"
    
    # Setup output
    output_dir = "runs/baseline_streamingllm_govreport"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "results.csv")
    
    # Process samples
    for i, sample in enumerate(tqdm(samples, desc="Processing Samples")):
        try:
            print(f"\n--- Processing sample {i + 1}/{len(samples)} ---")
            prompt = dataset_adapter.format_prompt(sample)
            
            # Generate text
            generated_text, metrics = streaming_llm_generate(
                model, tokenizer, prompt, args.max_new_tokens, kv_cache
            )
            
            # Compute perplexity if requested
            if args.compute_ppl and len(generated_text) > 50:
                print("Calculating perplexity on generated text...")
                try:
                    ppl = compute_perplexity(model, tokenizer, generated_text)
                    metrics['perplexity'] = ppl
                    print(f"Perplexity: {ppl:.2f}")
                except Exception as e:
                    print(f"Warning: Could not compute perplexity: {e}")
                    metrics['perplexity'] = float('inf')
            
            # Evaluate against ground truth
            ground_truth = sample["summary"]
            eval_scores = evaluate_prediction("summarization", generated_text, ground_truth)
            metrics.update(eval_scores)
            
            # Add metadata
            metrics["sample_id"] = sample.get("id", f"govreport_{i}")
            metrics["baseline"] = "streamingllm"
            metrics["model"] = args.model_dir
            metrics["max_new_tokens"] = args.max_new_tokens
            metrics["streaming_start_size"] = args.start_size
            metrics["streaming_recent_size"] = args.recent_size
            
            # Log to CSV
            log_metrics_to_csv(csv_file, metrics)
            
            # Print results
            print(f"Tokens/sec: {metrics['tokens_per_second']:.2f}")
            print(f"Latency/token: {metrics['latency_per_token']:.4f}s")
            print(f"Peak VRAM: {metrics['peak_vram_mb']:.2f} MB")
            if 'rougeL' in metrics:
                print(f"ROUGE-L: {metrics['rougeL']:.4f}")
            if 'f1' in metrics:
                print(f"F1 Score: {metrics['f1']:.4f}")
            if 'exact_match' in metrics:
                print(f"Exact Match: {metrics['exact_match']:.4f}")
            print(f"Generated text: {generated_text[:100]}...")
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"\n✓ Results saved to: {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run StreamingLLM on GovReport")
    parser.add_argument("--model_dir", required=True, help="Path to model directory")
    parser.add_argument("--dataset_dir", required=True, help="Path to GovReport dataset directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--start_size", type=int, default=4, help="Number of attention sinks")
    parser.add_argument("--recent_size", type=int, default=1020, help="Size of the recent token window")
    parser.add_argument("--compute_ppl", action="store_true", help="Enable perplexity calculation")
    
    args = parser.parse_args()
    main(args)