import torch
import time
import argparse
import os
import sys
import csv

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


# ============================================================================
# LLAMA2 CHAT TEMPLATE
# ============================================================================
LLAMA2_CHAT_TEMPLATE = r"""{% set sys = '' %}
{% for m in messages %}
    {% if m['role'] == 'system' %}{% set sys = m['content'] %}{% endif %}
{% endfor %}
{{ bos_token }}
{% for m in messages %}
    {% if m['role'] == 'user' %}
        {% if loop.first and sys %}
[INST] <<SYS>>
{{ sys }}
<</SYS>>
{{ m['content'] }} [/INST]
        {% else %}
[INST] {{ m['content'] }} [/INST]
        {% endif %}
    {% elif m['role'] == 'assistant' %}
{{ m['content'] }}{{ eos_token }}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}[INST] {% endif %}"""


def maybe_set_llama2_chat_template(tokenizer, model_id: str):
    """
    Set Llama2 chat template if the model is a Llama2 instruct/chat model
    and doesn't already have a template.
    """
    tmpl = getattr(tokenizer, "chat_template", None)
    mid = (model_id or "").lower()
    if not tmpl and ("llama-2" in mid or "llama2" in mid) and ("instruct" in mid or "chat" in mid):
        print("Setting Llama2 chat template...")
        tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE


def make_prompt(tokenizer, document: str) -> str:
    """
    Create a prompt for summarization using chat template if available.
    Falls back to simple format if no chat template exists.
    """
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": "You are a precise scientific summarizer."},
            {"role": "user", "content": "Summarize this government report into 4-6 sentences focusing on the main findings, methods, and implications.\n\n" + document},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Fallback for models without chat template
    return f"Summarize the following government report into 4-6 sentences:\n\n{document}\n\nSummary:"


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================
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
    Generates text using StreamingLLM with detailed prefill/decode profiling.
    Truncates input to 7000 tokens.
    """
    # Get model's device dynamically (important for quantized models)
    device = next(model.parameters()).device
    
    # Helper functions for VRAM tracking
    def reset_vram():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    
    def vram_peaks():
        if not torch.cuda.is_available():
            return {"peak_alloc_MB": float("nan"), "peak_reserved_MB": float("nan")}
        torch.cuda.synchronize()
        return {
            "peak_alloc_MB": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
            "peak_reserved_MB": round(torch.cuda.max_memory_reserved() / (1024**2), 2),
        }
    
    # 1. Tokenization with 7K truncation
    print("Tokenizing prompt (truncating to 7000 tokens)...")
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=True,  
        max_length=7000,
        padding=False
    ).to(device)
    
    input_ids = inputs.input_ids
    prompt_tokens = input_ids.shape[1]
    print(f"Prompt token count after truncation: {prompt_tokens}")
    
    # --- PREFILL PHASE ---
    reset_vram()
    t_prefill_start = time.perf_counter()
    
    # CRITICAL FIX: Don't initialize past_key_values to None
    # Let the first forward pass create it
    prompt_chunk_size = 512
    num_chunks = (prompt_tokens + prompt_chunk_size - 1) // prompt_chunk_size
    
    print(f"Processing prompt in {num_chunks} chunks...")
    
    try:
        # Process first chunk WITHOUT past_key_values to initialize cache
        first_chunk = input_ids[:, 0:min(prompt_chunk_size, prompt_tokens)]
        outputs = model(
            input_ids=first_chunk, 
            use_cache=True,
            return_dict=True
        )
        
        # Now apply StreamingLLM eviction to get proper cache
        past_key_values = kv_cache(outputs.past_key_values)
        
        # Process remaining chunks
        for i in range(prompt_chunk_size, prompt_tokens, prompt_chunk_size):
            chunk = input_ids[:, i:min(i + prompt_chunk_size, prompt_tokens)]
            
            outputs = model(
                input_ids=chunk, 
                past_key_values=past_key_values, 
                use_cache=True,
                return_dict=True
            )
            
            # Apply StreamingLLM eviction after each chunk
            past_key_values = kv_cache(outputs.past_key_values)
        
        prefill_time = time.perf_counter() - t_prefill_start
        prefill_stats = {
            "time_s": round(prefill_time, 3),
            **vram_peaks()
        }
        print("Prompt processing complete.")
        
    except Exception as e:
        print(f"\nError during prefill: {e}\n")
        import traceback
        traceback.print_exc()
        
        prefill_stats = {
            "time_s": round(time.perf_counter() - t_prefill_start, 3),
            **vram_peaks()
        }
        
        metrics = {
            "prefill_time_s": prefill_stats["time_s"],
            "prefill_peak_alloc_MB": prefill_stats.get("peak_alloc_MB", float("nan")),
            "prefill_peak_reserved_MB": prefill_stats.get("peak_reserved_MB", float("nan")),
            "decode_time_s": 0,
            "decode_peak_alloc_MB": float("nan"),
            "decode_peak_reserved_MB": float("nan"),
            "tokens_generated": 0,
            "total_time_s": prefill_stats["time_s"],
            "tokens_per_second": 0,
            "latency_per_token": 0,
            "input_tokens": int(prompt_tokens)
        }
        return f"ERROR: {e}", metrics
    
    # --- DECODE PHASE ---
    reset_vram()
    t_decode_start = time.perf_counter()
    
    generated_ids = []
    # Get the next token prediction from the last chunk's output
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    
    try:
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=pred_token_idx, 
                past_key_values=past_key_values, 
                use_cache=True,
                return_dict=True
            )
            
            # Apply StreamingLLM eviction after each new token
            past_key_values = kv_cache(outputs.past_key_values)
            
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            token_id = pred_token_idx.item()
            generated_ids.append(token_id)
            
            if token_id == tokenizer.eos_token_id:
                break
        
        decode_time = time.perf_counter() - t_decode_start
        decode_stats = {
            "time_s": round(decode_time, 3),
            **vram_peaks()
        }
        
    except Exception as e:
        print(f"\nError during decode: {e}\n")
        import traceback
        traceback.print_exc()
        
        decode_time = time.perf_counter() - t_decode_start
        decode_stats = {
            "time_s": round(decode_time, 3),
            **vram_peaks()
        }
    
    # --- FINALIZE & RETURN ---
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    tokens_generated = len(generated_ids)
    total_time_s = prefill_stats["time_s"] + decode_stats["time_s"]
    tokens_per_second = round(tokens_generated / decode_stats["time_s"], 2) if decode_stats["time_s"] > 0 else 0
    latency_per_token = round(decode_stats["time_s"] / tokens_generated, 4) if tokens_generated > 0 else 0
    
    metrics = {
        "prefill_time_s": prefill_stats["time_s"],
        "prefill_peak_alloc_MB": prefill_stats["peak_alloc_MB"],
        "prefill_peak_reserved_MB": prefill_stats["peak_reserved_MB"],
        "decode_time_s": decode_stats["time_s"],
        "decode_peak_alloc_MB": decode_stats["peak_alloc_MB"],
        "decode_peak_reserved_MB": decode_stats["peak_reserved_MB"],
        "tokens_generated": tokens_generated,
        "total_time_s": total_time_s,
        "tokens_per_second": tokens_per_second,
        "latency_per_token": latency_per_token,
        "input_tokens": int(prompt_tokens)
    }
    
    return generated_text, metrics


def save_prediction_to_csv(csv_file: str, sample_id: str, input_text: str, 
                           prediction: str, reference: str, append: bool = True):
    """
    Save prediction data to CSV for BERTScore calculation.
    
    Args:
        csv_file: Path to the CSV file
        sample_id: Unique identifier for the sample
        input_text: The input document text
        prediction: The model's generated summary
        reference: The ground truth summary
        append: Whether to append to existing file or create new
    """
    file_exists = os.path.isfile(csv_file)
    mode = 'a' if append and file_exists else 'w'
    
    with open(csv_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header if creating new file
        if mode == 'w' or not file_exists:
            writer.writerow(['sample_id', 'input', 'prediction', 'reference'])
        
        # Write the data row
        writer.writerow([sample_id, input_text, prediction, reference])


def main(args):
    # Clear CUDA cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_dir, 
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Set Llama2 chat template if applicable
    maybe_set_llama2_chat_template(tokenizer, args.model_dir)
    
    # Enable StreamingLLM
    print(f"Enabling StreamingLLM with start_size={args.start_size} and recent_size={args.recent_size}")
    kv_cache = enable_streaming_llm(
        model,
        start_size=args.start_size,
        recent_size=args.recent_size
    )
    
    # Load dataset
    dataset_adapter = GovReportAdapter(args.dataset_dir)
    samples = dataset_adapter.get_samples("test", args.num_samples)
    
    # Setup output
    output_dir = "runs/baseline_streamingllm_govreport"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "results.csv")
    predictions_csv = os.path.join(output_dir, "streamingllm_govreport_predictions.csv")
    
    # Process samples
    for i, sample in enumerate(tqdm(samples, desc="Processing Samples")):
        try:
            print(f"\n--- Processing sample {i + 1}/{len(samples)} ---")
            
            # Get the input document
            input_document = sample.get("report", sample.get("document", ""))
            ground_truth = sample["summary"]
            sample_id = sample.get("id", f"govreport_{i}")
            
            # Create prompt using chat template
            prompt = make_prompt(tokenizer, input_document)
            
            # Debug: Show if chat template was used
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                print("Using chat template for prompt formatting")
            else:
                print("Using simple prompt formatting (no chat template)")
            
            # Generate text
            generated_text, metrics = streaming_llm_generate(
                model, tokenizer, prompt, args.max_new_tokens, kv_cache
            )
            
            # Save to predictions CSV
            save_prediction_to_csv(
                predictions_csv,
                sample_id=sample_id,
                input_text=input_document,
                prediction=generated_text,
                reference=ground_truth,
                append=(i > 0)
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
            eval_scores = evaluate_prediction("summarization", generated_text, ground_truth)
            metrics.update(eval_scores)
            
            # Add metadata
            metrics["sample_id"] = sample_id
            metrics["baseline"] = "streamingllm"
            metrics["model"] = args.model_dir
            metrics["max_new_tokens"] = args.max_new_tokens
            metrics["streaming_start_size"] = args.start_size
            metrics["streaming_recent_size"] = args.recent_size
            metrics["used_chat_template"] = bool(hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template)
            
            # Log to CSV
            log_metrics_to_csv(csv_file, metrics)
            
            # Print results
            print(f"Tokens/sec: {metrics['tokens_per_second']:.2f}")
            print(f"Latency/token: {metrics['latency_per_token']:.4f}s")
            if 'rougeL' in metrics:
                print(f"ROUGE-L: {metrics['rougeL']:.4f}")
            if 'f1' in metrics:
                print(f"F1 Score: {metrics['f1']:.4f}")
            if 'exact_match' in metrics:
                print(f"Exact Match: {metrics['exact_match']:.4f}")
            print(f"Generated text: {generated_text[:100]}...")
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✓ Results saved to: {csv_file}")
    print(f"✓ Predictions saved to: {predictions_csv}")
    print(f"\nTo calculate BERTScore, run:")
    print(f"  python -m bert_score.score --lang en \\")
    print(f"    --predictions {predictions_csv} \\")
    print(f"    --references {predictions_csv}")


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