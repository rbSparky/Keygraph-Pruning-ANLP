import torch
import time
import argparse
import os
import sys
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
# Import your existing data adapter and evaluation functions
# Make sure the paths are correct based on your project structure
from keygraph.eval_metrics import evaluate_prediction
from keygraph.dataset.data import GovReportAdapter
from keygraph.logging_utils import log_metrics_to_csv

# Import from the StreamingLLM library
from keygraph.streaming.streaming_llm.enable_streaming_llm import enable_streaming_llm
from keygraph.models import load_model_and_tokenizer # Using your model loader

def streaming_llm_generate(model, tokenizer, prompt, max_new_tokens, kv_cache):
    """
    Generates text using the StreamingLLM KV cache and measures performance.
    """
    device = model.device
    metrics = {}
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]

    # --- Performance Measurement ---
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    start_time = time.perf_counter()

    # The key difference: we manage past_key_values manually
    past_key_values = None
    generated_ids = []

    # Process the prompt first to fill the KV cache
    outputs = model(input_ids=input_ids, use_cache=True)
    past_key_values = kv_cache(outputs.past_key_values)
    
    # Start generation from the last token of the prompt
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        # Apply the streaming cache logic to the updated KV states
        past_key_values = kv_cache(outputs.past_key_values)
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())

        if pred_token_idx == tokenizer.eos_token_id:
            break

    end_time = time.perf_counter()
    # --- End of Measurement ---
    
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
    # Load model and tokenizer using your function
    model, tokenizer = load_model_and_tokenizer(args.model_dir, "cuda")

    # --- Enable StreamingLLM ---
    # This patches the model's attention mechanism and creates the cache handler
    print(f"Enabling StreamingLLM with start_size={args.start_size} and recent_size={args.recent_size}")
    kv_cache = enable_streaming_llm(
        model, start_size=args.start_size, recent_size=args.recent_size
    )
    # -------------------------

    dataset_adapter = GovReportAdapter(args.dataset_dir)
    samples = dataset_adapter.get_samples("validation", args.num_samples)

    output_dir = f"runs/baseline_streamingllm_govreport"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "results.csv")

    for i, sample in enumerate(samples):
        print(f"\nProcessing sample {i + 1}/{len(samples)}")
        prompt = dataset_adapter.format_prompt(sample)
        
        generated_text, metrics = streaming_llm_generate(
            model, tokenizer, prompt, args.max_new_tokens, kv_cache
        )
        
        ground_truth = sample["summary"]
        eval_scores = evaluate_prediction("summarization", generated_text, ground_truth)
        metrics.update(eval_scores)
        
        # Add metadata for logging
        metrics["sample_id"] = sample.get("id", f"govreport_{i}")
        metrics["baseline"] = "streamingllm"
        
        log_metrics_to_csv(csv_file, metrics)
        
        print(f"Tokens/sec: {metrics['tokens_per_second']:.2f}")
        print(f"Peak VRAM: {metrics['peak_vram_mb']:.2f} MB")
        print(f"ROUGE-L: {metrics['rougeL']:.4f}")
        print(f"Generated text: {generated_text[:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run StreamingLLM on GovReport")
    parser.add_argument("--model_dir", required=True, help="Path to model directory")
    parser.add_argument("--dataset_dir", required=True, help="Path to GovReport dataset directory (can be 'ccdv/govreport-summarization')")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--start_size", type=int, default=4, help="Number of attention sinks (initial tokens)")
    parser.add_argument("--recent_size", type=int, default=1020, help="Size of the recent token window")
    args = parser.parse_args()
    main(args)