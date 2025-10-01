import torch
import time
import argparse
import os
import sys
from tqdm import tqdm

# Add the parent directory to the path to find your 'keygraph' module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your existing utilities
from keygraph.eval_metrics import evaluate_prediction
from keygraph.dataset.data import GovReportAdapter
from keygraph.logging_utils import log_metrics_to_csv
from keygraph.models import load_model_and_tokenizer

@torch.no_grad()
def compute_perplexity(model, tokenizer, text, stride=512):
    """
    Computes the perplexity of a model on a given text using a sliding window.
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

def full_kv_generate(model, tokenizer, prompt, max_new_tokens):
    """
    Generates text using the standard full KV cache and measures performance.
    The tokenizer will automatically truncate prompts longer than the model's context window.
    """
    device = model.device
    metrics = {}
    
    # The tokenizer truncates long prompts by default, which is the behavior for this baseline.
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings).to(device)
    input_ids_len = inputs.input_ids.shape[1]

    # --- Performance Measurement Setup ---
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    start_time = time.perf_counter()

    # Generate text using the standard `generate` method
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    end_time = time.perf_counter()
    # --- End of Measurement ---
    
    generated_ids = outputs[0][input_ids_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # --- Calculate Metrics ---
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

    dataset_adapter = GovReportAdapter(args.dataset_dir)
    samples = dataset_adapter.get_samples("validation", args.num_samples)

    output_dir = "runs/baseline_full_kv_govreport"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "results.csv")

    for i, sample in enumerate(tqdm(samples, desc="Processing Samples")):
        prompt = dataset_adapter.format_prompt(sample)
        
        generated_text, metrics = full_kv_generate(
            model, tokenizer, prompt, args.max_new_tokens
        )
        
        ground_truth = sample["summary"]
        
        # Calculate Perplexity (if enabled)
        if args.compute_ppl:
            ppl = compute_perplexity(model, tokenizer, ground_truth)
            metrics['perplexity'] = ppl
        
        # Calculate ROUGE scores
        eval_scores = evaluate_prediction("summarization", generated_text, ground_truth)
        metrics.update(eval_scores)
        
        # Add metadata for logging
        metrics["sample_id"] = sample.get("id", f"govreport_{i}")
        metrics["baseline"] = "full_kv"
        metrics["model"] = args.model_dir
        metrics["max_new_tokens"] = args.max_new_tokens
        
        log_metrics_to_csv(csv_file, metrics)
        
    print("\n" + "="*50)
    print("BASELINE RUN COMPLETE")
    print(f"Results for {len(samples)} samples saved to {csv_file}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full KV Cache Baseline on GovReport")
    parser.add_argument("--model_dir", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Path to model directory")
    parser.add_argument("--dataset_dir", type=str, default="ccdv/govreport-summarization", help="Path to GovReport dataset directory")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to process")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--compute_ppl", action="store_true", help="Enable perplexity calculation")
    args = parser.parse_args()
    main(args)
