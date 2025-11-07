import torch
import time
import argparse
import os
import sys
from tqdm import tqdm

# Add the parent directory to the path to allow imports from keygraph
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from your existing keygraph package
from keygraph.eval_metrics import evaluate_prediction
from keygraph.dataset.data import GovReportAdapter
from keygraph.logging_utils import log_metrics_to_csv
from keygraph.models import load_model_and_tokenizer

# Import the SnapKV patching functions
# from keygraph.snap_kv.snpakv.monkeypatch.monkeypatch import replace_llama, replace_mistral, replace_mixtral, replace_tinyllama
from keygraph.snap_kv.snapkv.monkeypatch.monkeypatch import replace_tinyllama, replace_llama, replace_mistral, replace_mixtral
def load_and_patch_model(model_dir, device):
    """
    Loads a model and tokenizer, then applies the appropriate SnapKV patch.
    """
    model, tokenizer = load_model_and_tokenizer(model_dir, device, use_flash_attention_2=True)
    
    model_type = model.config.model_type.lower()
    
    if "llama" in model_type:
        # This will catch both regular LLaMA and TinyLlama
        print(f"Detected LLaMA family model type: {model_type}")
        replace_tinyllama() # Use the dedicated patch for robustness
    elif "mistral" in model_type:
        print(f"Detected Mistral family model type: {model_type}")
        replace_mistral()
    elif "mixtral" in model_type:
        print(f"Detected Mixtral family model type: {model_type}")
        replace_mixtral()
    else:
        warnings.warn(f"Model type '{model_type}' not explicitly supported by this script's patching logic. SnapKV may not be enabled.")
        
    return model, tokenizer

@torch.no_grad()
def compute_perplexity(model, tokenizer, text, stride=512):
    """
    Computes the perplexity of a model on a given text.
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
def snapkv_generate(model, tokenizer, prompt, max_new_tokens):
    """
    Generates text using a SnapKV-patched model and measures performance.
    """
    device = model.device
    metrics = {}
    
    # Ensure prompt is not longer than model's max length minus generation tokens
    max_input_length = model.config.max_position_embeddings - max_new_tokens
    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_input_length, truncation=True).to(device)
    input_ids = inputs.input_ids

    # Performance Measurement Setup
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    start_time = time.perf_counter()

    # Generate text using the standard .generate() method
    outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    
    end_time = time.perf_counter()
    
    generated_ids = outputs[0][input_ids.shape[1]:]
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model, tokenizer, and apply SnapKV patch
    model, tokenizer = load_and_patch_model(args.model_dir, device)

    dataset_adapter = GovReportAdapter(args.dataset_dir)
    samples = dataset_adapter.get_samples("validation", args.num_samples)

    output_dir = "runs/snapkv_govreport"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "results.csv")

    for i, sample in enumerate(tqdm(samples, desc="Evaluating SnapKV")):
        print(f"\n--- Processing sample {i + 1}/{len(samples)} ---")
        prompt = dataset_adapter.format_prompt(sample)
        
        generated_text, metrics = snapkv_generate(
            model, tokenizer, prompt, args.max_new_tokens
        )
        
        ground_truth = sample["summary"]
        
        if args.compute_ppl:
            print("Calculating perplexity on ground truth summary...")
            ppl = compute_perplexity(model, tokenizer, ground_truth)
            metrics['perplexity'] = ppl
            print(f"Perplexity: {ppl:.2f}")

        eval_scores = evaluate_prediction("summarization", generated_text, ground_truth)
        metrics.update(eval_scores)
        
        metrics["sample_id"] = sample.get("id", f"govreport_{i}")
        metrics["baseline"] = "snapkv"
        metrics["model"] = args.model_dir
        metrics["max_new_tokens"] = args.max_new_tokens
        log_metrics_to_csv(csv_file, metrics)
        
        print(f"Tokens/sec: {metrics['tokens_per_second']:.2f}")
        print(f"Peak VRAM: {metrics['peak_vram_mb']:.2f} MB")
        if 'rougeL' in metrics: print(f"ROUGE-L: {metrics['rougeL']:.4f}")
        print(f"Generated text: {generated_text[:200]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SnapKV Evaluation on GovReport")
    parser.add_argument("--model_dir", type=str, required=True, help="Path or Hugging Face identifier for the model (e.g., 'TinyLlama/TinyLlama-1.1B-Chat-v1.0').")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the GovReport dataset directory.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to evaluate.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens for generation.")
    parser.add_argument("--compute_ppl", action="store_true", help="Enable perplexity calculation on the ground truth.")
    args = parser.parse_args()
    main(args)


# python3 ./run_snapkv_govreport.py \
#     --model_dir "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
#     --dataset_dir "ccdv/govreport-summarization" \
#     --num_samples 10 \
#     --max_new_tokens 256