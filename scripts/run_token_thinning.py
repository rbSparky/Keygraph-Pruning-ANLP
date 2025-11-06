"""
Run Top-K Token Thinning on GovReport and NarrativeQA

Compatible with existing evaluation metrics and dataset adapters.
"""

import torch
import time
import argparse
import os
import sys
import re
import string
import csv
from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Import your existing modules
from token_thinning import topk_generate
# Use your existing dataset adapters and metrics
# Assuming these are in the same directory or adjust the import path
from keygraph.dataset.data import GovReportAdapter, NarrativeQAAdapter, QasperAdapter


# --- EVALUATION FUNCTIONS ---

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, ground_truth):
    """Compute exact match between prediction and ground truth."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def compute_f1(prediction, ground_truth):
    """Compute F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common_tokens = set(pred_tokens) & set(gt_tokens)
    if len(common_tokens) == 0:
        return 0.0
    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def compute_rouge_scores(prediction, ground_truth):
    """Compute ROUGE scores between prediction and ground truth."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


def evaluate_prediction(task_type, prediction, ground_truths):
    """
    Evaluate a prediction based on the task type.
    Compatible with your existing evaluation pipeline.
    """
    if task_type == "summarization":
        if isinstance(ground_truths, list):
            ground_truth = ground_truths[0] if ground_truths else ""
        else:
            ground_truth = ground_truths

        scores = compute_rouge_scores(prediction, ground_truth)
        
        # Add F1 and EM for summarization
        scores['f1'] = compute_f1(prediction, ground_truth)
        scores['exact_match'] = compute_exact_match(prediction, ground_truth)
        
        return scores

    elif task_type in ["qa", "qasper", "narrativeqa"]:
        if not isinstance(ground_truths, list):
            ground_truths = [ground_truths]

        em_scores = [compute_exact_match(prediction, gt) for gt in ground_truths]
        f1_scores = [compute_f1(prediction, gt) for gt in ground_truths]

        max_em = max(em_scores) if em_scores else 0.0
        max_f1 = max(f1_scores) if f1_scores else 0.0

        return {'exact_match': max_em, 'f1': max_f1}
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def log_metrics_to_csv(csv_file, metrics):
    """
    Log metrics to CSV file.
    Simple implementation - you can replace with your actual logging function.
    """
    # Check if file exists to write header
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(metrics)

# --- UPDATED MODEL LOADING FUNCTION ---

def load_model_and_tokenizer(model_dir: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the model and tokenizer correctly once."""
    print(f"Loading model and tokenizer from {model_dir}")

    # 1. Load the tokenizer a single time with all required settings
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=False,
        trust_remote_code=True,
        padding_side="left"  # Important for generation
    )

    # 2. Set the pad token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load the model a single time using the recommended `device_map`
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=False,
        trust_remote_code=True,
        torch_dtype=torch.float16,   # Use float16 for better performance on GPUs
        device_map='auto'            # Automatically handle device placement (GPU/CPU)
    )

    # 4. Set the model to evaluation mode
    model.eval()
    print("Model and tokenizer loaded successfully.")
    
    return model, tokenizer


def main(args):
    """Main evaluation function."""
    
    # --- MODIFIED ---
    # Setup device is no longer needed, load_model_and_tokenizer handles it.
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_dir)
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    if args.dataset == "govreport":
        dataset_adapter = GovReportAdapter(args.dataset_dir)
        task_type = "summarization"
        ground_truth_key = "summary"
    elif args.dataset == "narrativeqa":
        dataset_adapter = NarrativeQAAdapter(args.dataset_dir)
        task_type = "qa"
        ground_truth_key = "answers"
    elif args.dataset == "qasper":
        dataset_adapter = QasperAdapter(args.dataset_dir)
        task_type = "qasper"
        ground_truth_key = "answers"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Get samples
    samples = dataset_adapter.get_samples(args.split, args.num_samples)
    print(f"Loaded {len(samples)} samples from {args.dataset}")
    
    # Setup output directory
    output_dir = f"runs/topk_{args.dataset}_k{args.k}_recent{args.recent}_method{args.method}"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "results.csv")
    
    print(f"\n{'='*80}")
    print(f"Running Top-K Token Thinning Evaluation")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_dir}")
    print(f"Method: {args.method}")
    print(f"K: {args.k}")
    print(f"Protect Recent: {args.recent}")
    print(f"Max New Tokens: {args.max_new_tokens}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    # Process samples
    all_metrics = []
    
    for i, sample in enumerate(tqdm(samples, desc="Processing Samples")):
        print(f"\n--- Sample {i + 1}/{len(samples)} ---")
        
        # Format prompt
        prompt = dataset_adapter.format_prompt(sample)
        
        # Generate with top-k thinning
        generated_text, perf_metrics = topk_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            k=args.k,
            protect_recent=args.recent,
            method=args.method,
            max_new_tokens=args.max_new_tokens,
            compute_ppl=args.compute_ppl
        )
        
        # Get ground truth
        ground_truth = sample[ground_truth_key]
        
        # Evaluate quality
        eval_scores = evaluate_prediction(task_type, generated_text, ground_truth)
        
        # Combine metrics
        metrics = {**perf_metrics, **eval_scores}
        
        # Add metadata
        metrics["sample_id"] = sample.get("id", f"{args.dataset}_{i}")
        metrics["baseline"] = "topk_thinning"
        metrics["model"] = args.model_dir
        metrics["dataset"] = args.dataset
        metrics["max_new_tokens"] = args.max_new_tokens
        metrics["topk_k"] = args.k
        metrics["topk_recent"] = args.recent
        metrics["topk_method"] = args.method
        
        # Log to CSV
        log_metrics_to_csv(csv_file, metrics)
        all_metrics.append(metrics)
        
        # Print progress
        print(f"Tokens/sec: {metrics['tokens_per_second']:.2f}")
        print(f"Latency/token: {metrics['latency_per_token']:.4f}s")
        print(f"Peak VRAM: {metrics['peak_vram_mb']:.2f} MB")
        print(f"Cache Size: {metrics['kv_cache_size']} tokens")
        print(f"Compression: {metrics['compression_ratio']:.2f}x")
        
        if 'rougeL' in metrics:
            print(f"ROUGE-L: {metrics['rougeL']:.4f}")
        if 'f1' in metrics:
            print(f"F1: {metrics['f1']:.4f}")
        if 'exact_match' in metrics:
            print(f"EM: {metrics['exact_match']:.4f}")
        if 'perplexity' in metrics:
            print(f"Perplexity: {metrics['perplexity']:.2f}")
        
        print(f"Generated (first 100 chars): {generated_text[:100]}...")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}")
    
    if not all_metrics:
        print("No samples processed.")
        return

    avg_tokens_per_sec = sum(m['tokens_per_second'] for m in all_metrics) / len(all_metrics)
    avg_latency = sum(m['latency_per_token'] for m in all_metrics) / len(all_metrics)
    avg_vram = sum(m['peak_vram_mb'] for m in all_metrics) / len(all_metrics)
    avg_cache_size = sum(m['kv_cache_size'] for m in all_metrics) / len(all_metrics)
    avg_compression = sum(m['compression_ratio'] for m in all_metrics) / len(all_metrics)
    
    print(f"\nPerformance:")
    print(f"  Avg Tokens/sec: {avg_tokens_per_sec:.2f}")
    print(f"  Avg Latency/token: {avg_latency:.4f}s")
    print(f"  Avg Peak VRAM: {avg_vram:.2f} MB")
    print(f"  Avg Cache Size: {avg_cache_size:.1f} tokens")
    print(f"  Avg Compression: {avg_compression:.2f}x")
    
    print(f"\nQuality:")
    if 'f1' in all_metrics[0]:
        avg_f1 = sum(m['f1'] for m in all_metrics) / len(all_metrics)
        print(f"  Avg F1: {avg_f1:.4f}")
    
    if 'exact_match' in all_metrics[0]:
        avg_em = sum(m['exact_match'] for m in all_metrics) / len(all_metrics)
        print(f"  Avg EM: {avg_em:.4f}")
    
    if 'rougeL' in all_metrics[0]:
        avg_rougeL = sum(m['rougeL'] for m in all_metrics) / len(all_metrics)
        print(f"  Avg ROUGE-L: {avg_rougeL:.4f}")
    
    if 'perplexity' in all_metrics[0]:
        # Handle cases where perplexity might not have been computed for all
        ppl_values = [m['perplexity'] for m in all_metrics if 'perplexity' in m]
        if ppl_values:
            avg_ppl = sum(ppl_values) / len(ppl_values)
            print(f"  Avg Perplexity: {avg_ppl:.2f}")
    
    print(f"\n{'='*80}")
    print(f"✓ Results saved to: {csv_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Top-K Token Thinning Evaluation")
    
    parser.add_argument("--model_dir", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Path or name of model directory")
    parser.add_argument("--dataset", type=str, default="govreport",
                        choices=["govreport", "narrativeqa", "qasper"],
                        help="Dataset to evaluate on")
    parser.add_argument("--dataset_dir", type=str, default="ccdv/govreport-summarization",
                        help="Path to dataset directory")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum new tokens to generate")
    parser.add_argument("--k", type=int, default=128,
                        help="Number of tokens to keep in KV cache")
    parser.add_argument("--recent", type=int, default=64,
                        help="Number of recent tokens to protect")
    parser.add_argument("--method", type=str, default="attention",
                        choices=["attention", "uniform", "recency"],
                        help="Token selection method")
    parser.add_argument("--compute_ppl", action="store_true",
                        help="Enable perplexity calculation")
    
    args = parser.parse_args()
    
    main(args)


# python3 run_token_thinning.py \
#     --model_dir "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
#     --dataset_dir "narrativeqa" \
#     --num_samples 100 \
#     --k 128 \
#     --recent 64 \
#     --method attention \
#     --max_new_tokens 64 \
#     --compute_ppl