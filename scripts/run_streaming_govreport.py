import torch
import time
import argparse
import os
import sys
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
# Import your existing data adaptier and evaluation functions
# Make sure the paths are correct based on your project structure
from keygraph.eval_metrics import evaluate_prediction
from keygraph.dataset.data import GovReportAdapter
from keygraph.logging_utils import log_metrics_to_csv

# Import from the StreamingLLM library
from keygraph.streaming.streaming_llm.enable_streaming_llm import enable_streaming_llm
from keygraph.models import load_model_and_tokenizer # Using your model loader
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
    
    # Use the model's configured context window size
    max_length = model.config.max_position_embeddings
    
    nlls = [] # Negative log-likelihoods
    prev_end_loc = 0

    if seq_len == 0:
        return 0.0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 # Mask tokens that are not being predicted

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

    # --- Performance Measurement Setup ---
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    start_time = time.perf_counter()

    past_key_values = None
    
    # --- THIS IS THE FIX: PROCESS PROMPT IN CHUNKS ---
    # Instead of one large forward pass, this loop processes the prompt in smaller pieces.
    # The largest attention matrix ever created is now based on the chunk size, not the full prompt length.
    prompt_chunk_size = 512 
    for i in range(0, input_ids.shape[1], prompt_chunk_size):
        chunk = input_ids[:, i:i + prompt_chunk_size]
        with torch.no_grad():
            outputs = model(input_ids=chunk, past_key_values=past_key_values, use_cache=True)
        
        # After each chunk, the st
        #streaming logic is applied to keep the KV cache size fixed.
        past_key_values = kv_cache(outputs.past_key_values)
    # --- END OF FIX ---

    # Now, generate new tokens one by one with the warmed-up cache
    generated_ids = []
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values, use_cache=True)
        
        past_key_values = kv_cache(outputs.past_key_values)
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())

        if pred_token_idx == tokenizer.eos_token_id:
            break

    end_time = time.perf_counter()
    
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Calculate metrics
    total_time = end_time - start_time
    tokens_generated = len(generated_ids)
    
    metrics['tokens_per_second'] = tokens_generated / total_time if total_time > 0 else 0
    metrics['tokens_generated'] = tokens_generated
    
    if torch.cuda.is_available():
        metrics['peak_vram_mb'] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        metrics['peak_vram_mb'] = 0.0

    return generated_text, metrics


def main(args):
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_dir, "cuda" if torch.cuda.is_available() else "cpu")

    # Enable StreamingLLM
    print(f"Enabling StreamingLLM with start_size={args.start_size} and recent_size={args.recent_size}")
    kv_cache = enable_streaming_llm(
        model, start_size=args.start_size, recent_size=args.recent_size
    )

    dataset_adapter = GovReportAdapter(args.dataset_dir)
    samples = dataset_adapter.get_samples("validation", args.num_samples)

    output_dir = "runs/baseline_streamingllm_govreport"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "results.csv")

    for i, sample in enumerate(samples):
        print(f"\nProcessing sample {i + 1}/{len(samples)}")
        prompt = dataset_adapter.format_prompt(sample)
        
        generated_text, metrics = streaming_llm_generate(
            model, tokenizer, prompt, args.max_new_tokens, kv_cache
        )
        
        ground_truth = sample["summary"]
        
        # --- NEW: CALCULATE PERPLEXITY (if enabled) ---
        if args.compute_ppl:
            print("Calculating perplexity on ground truth summary...")
            ppl = compute_perplexity(model, tokenizer, ground_truth)
            metrics['perplexity'] = ppl
            print(f"Perplexity: {ppl:.2f}")
        # --------------------------------------------

        eval_scores = evaluate_prediction("summarization", generated_text, ground_truth)
        metrics.update(eval_scores)
        
        metrics["sample_id"] = sample.get("id", f"govreport_{i}")
        metrics["baseline"] = "streamingllm"
        metrics["model"] = args.model_dir
        metrics["max_new_tokens"] = args.max_new_tokens
        metrics["streaming_start_size"] = args.start_size
        metrics["streaming_recent_size"] = args.recent_size
        log_metrics_to_csv(csv_file, metrics)
        
        print(f"Tokens/sec: {metrics['tokens_per_second']:.2f}")
        print(f"Peak VRAM: {metrics['peak_vram_mb']:.2f} MB")
        if 'rougeL' in metrics: print(f"ROUGE-L: {metrics['rougeL']:.4f}")
        print(f"Generated text: {generated_text[:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run StreamingLLM on GovReport")
    parser.add_argument("--model_dir", required=True, help="Path to model directory")
    parser.add_argument("--dataset_dir", required=True, help="Path to GovReport dataset directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--start_size", type=int, default=4, help="Number of attention sinks")
    parser.add_argument("--recent_size", type=int, default=1020, help="Size of the recent token window")
    # --- NEW: ARGUMENT FOR PERPLEXITY ---
    parser.add_argument("--compute_ppl", action="store_true", help="Enable perplexity calculation")
    args = parser.parse_args()
    main(args)
