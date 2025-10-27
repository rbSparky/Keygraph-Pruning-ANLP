import os
import sys
import argparse
import json
import yaml
import torch
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from keygraph.utils import set_seed, get_device
from keygraph.models import load_model_and_tokenizer
from keygraph.dataset.data import GovReportAdapter, NarrativeQAAdapter, QasperAdapter
from keygraph.eval_metrics import evaluate_prediction
from keygraph.logging_utils import log_metrics_to_csv, log_metrics_to_jsonl
from keygraph.method.keygraph_cache import KeyGraphCache
from keygraph.method.attention_patch import keygraph_attention_patch


def load_config(config_path="../configs/paths.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_dataset_adapter(dataset_name, dataset_dir):
    """Get the appropriate dataset adapter."""
    if dataset_name == "govreport":
        return GovReportAdapter(dataset_dir)
    elif dataset_name == "narrativeqa":
        return NarrativeQAAdapter(dataset_dir)
    elif dataset_name == "qasper":
        return QasperAdapter(dataset_dir)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


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


def keygraph_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    r_dim: int = 32,
    knn_k: int = 16,
    tau: float = 0.8,
    rescue: bool = True,
    rescue_probe_size: int = 6,
    upper_layers_only: bool = False,
    compute_ppl: bool = False,
    generated_text_for_ppl: str = None):
    """
    Generate using KeyGraph pruning method.

    Returns:
        generated_text (str): The generated text
        metrics (dict): Performance metrics
    """
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    keygraph_cache = KeyGraphCache(
        model, tokenizer, prompt,
        r_dim=r_dim,
        tau=tau,
        knn_k=knn_k,
        rescue=rescue,
        rescue_probe_size=rescue_probe_size,
        upper_layers_only=upper_layers_only)

    compression_ratio = keygraph_cache.get_compression_ratio()
    kv_bytes_saved = keygraph_cache.get_kv_bytes_saved()

    with keygraph_attention_patch(model, keygraph_cache) as patched_cache:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                past_key_values=None,
                use_cache=True)

    end_time = time.time()

    generated_tokens = outputs.sequences.shape[1] - inputs.input_ids.shape[1]
    total_time = end_time - start_time
    tokens_per_second = generated_tokens / total_time if total_time > 0 else 0
    latency_per_token = total_time / generated_tokens if generated_tokens > 0 else 0  # ADDED
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    generated_text = tokenizer.decode(
        outputs.sequences[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True)

    metrics = {
        "tokens_generated": generated_tokens,
        "total_time_seconds": total_time,
        "tokens_per_second": tokens_per_second,
        "latency_per_token": latency_per_token,  # ADDED
        "peak_vram_mb": peak_memory,
        "kv_cache_method": "keygraph",
        "compression_ratio": compression_ratio,
        "kv_bytes_saved_mb": kv_bytes_saved / (1024 ** 2),
        "r_dim": r_dim,
        "knn_k": knn_k,
        "tau": tau}

    # ADDED: Calculate perplexity if requested
    if compute_ppl and generated_text:
        print("Calculating perplexity on generated text...")
        ppl = compute_perplexity(model, tokenizer, generated_text)
        metrics['perplexity'] = ppl
        print(f"Perplexity: {ppl:.2f}")

    return generated_text, metrics


def main():
    parser = argparse.ArgumentParser(description="Run KeyGraph pruning experiments")
    parser.add_argument("--model_dir", required=True, help="Path to model directory")
    parser.add_argument("--dataset", required=True, choices=["govreport", "narrativeqa", "qasper"],
                        help="Dataset to use")
    parser.add_argument("--dataset_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to process")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum new tokens to generate")
    parser.add_argument("--r_dim", type=int, default=32, help="Random projection dimension")
    parser.add_argument("--knn_k", type=int, default=16, help="Number of neighbors for kNN")
    parser.add_argument("--tau", type=float, default=0.8, help="Cosine similarity threshold")
    parser.add_argument("--no_rescue", action="store_true", help="Disable rescue expansion")
    parser.add_argument("--rescue_probe_size", type=int, default=6, help="Rescue probe size")
    parser.add_argument("--upper_layers_only", action="store_true", help="Process only upper layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_preds", action="store_true", help="Save predictions to file")
    parser.add_argument("--compute_ppl", action="store_true", help="Enable perplexity calculation")  # ADDED

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(args.model_dir, device)
    dataset_adapter = get_dataset_adapter(args.dataset, args.dataset_dir)
    samples = dataset_adapter.get_samples("test", args.num_samples)

    output_dir = f"runs/keygraph_{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    csv_file = os.path.join(output_dir, "results.csv")
    preds_file = os.path.join(output_dir, "predictions.jsonl") if args.save_preds else None

    all_metrics = []

    for i, sample in enumerate(samples):
        print(f"\nProcessing sample {i + 1}/{len(samples)}")
        print(f"Sample ID: {sample.get('id', 'N/A')}")

        prompt = dataset_adapter.format_prompt(sample)
        print(f"Prompt length: {len(prompt)} characters")

        generated_text, metrics = keygraph_generate(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            r_dim=args.r_dim,
            knn_k=args.knn_k,
            tau=args.tau,
            rescue=not args.no_rescue,
            rescue_probe_size=args.rescue_probe_size,
            upper_layers_only=args.upper_layers_only,
            compute_ppl=args.compute_ppl)  # ADDED

        metrics["sample_id"] = sample.get("id", f"{args.dataset}_{i}")
        metrics["dataset"] = args.dataset
        metrics["max_new_tokens"] = args.max_new_tokens

        if args.dataset == "govreport":
            task_type = "summarization"
            ground_truth = sample["summary"]
        else:
            task_type = "qa"
            ground_truth = sample["answers"]

        if "summary" in sample or "answers" in sample:
            eval_scores = evaluate_prediction(task_type, generated_text, ground_truth)
            metrics.update(eval_scores)

        log_metrics_to_csv(csv_file, metrics)
        if preds_file:
            pred_record = {
                "sample_id": sample.get("id", f"{args.dataset}_{i}"),
                "prompt": prompt,
                "generated": generated_text,
                "ground_truth": ground_truth}
            log_metrics_to_jsonl(preds_file, pred_record)

        all_metrics.append(metrics)

        # UPDATED: Print all metrics
        print(f"Generated {metrics['tokens_generated']} tokens")
        print(f"Tokens/sec: {metrics['tokens_per_second']:.2f}")
        print(f"Latency/token: {metrics['latency_per_token']:.4f}s")  # ADDED
        print(f"Peak VRAM: {metrics['peak_vram_mb']:.2f} MB")
        print(f"Compression ratio: {metrics['compression_ratio']:.4f}")
        print(f"KV bytes saved: {metrics['kv_bytes_saved_mb']:.2f} MB")
        if "rougeL" in metrics:
            print(f"ROUGE-L: {metrics['rougeL']:.4f}")
        if "f1" in metrics:
            print(f"F1: {metrics['f1']:.4f}")
        if "exact_match" in metrics:
            print(f"EM: {metrics['exact_match']:.4f}")

        print(f"Generated text: {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")

    # UPDATED: Calculate and print averages for all metrics
    if all_metrics:
        print("\n" + "=" * 80)
        print("AVERAGE METRICS ACROSS ALL SAMPLES")
        print("=" * 80)
        
        metrics_to_average = [
            'tokens_per_second', 'latency_per_token', 'peak_vram_mb',
            'compression_ratio', 'kv_bytes_saved_mb', 'tokens_generated'
        ]
        if args.compute_ppl:
            metrics_to_average.append('perplexity')
        
        if args.dataset == "govreport":
            metrics_to_average.extend(['rouge1', 'rouge2', 'rougeL', 'f1', 'exact_match'])
        else:
            metrics_to_average.extend(['exact_match', 'f1'])
        
        avg_metrics = {}
        for metric_name in metrics_to_average:
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            if values:
                avg_metrics[metric_name] = sum(values) / len(values)
        
        # Print averages
        print(f"Average Tokens/sec: {avg_metrics.get('tokens_per_second', 0):.2f}")
        print(f"Average Latency/token: {avg_metrics.get('latency_per_token', 0):.4f}s")
        print(f"Average Peak VRAM: {avg_metrics.get('peak_vram_mb', 0):.2f} MB")
        print(f"Average Compression Ratio: {avg_metrics.get('compression_ratio', 0):.4f}")
        print(f"Average KV Bytes Saved: {avg_metrics.get('kv_bytes_saved_mb', 0):.2f} MB")
        print(f"Average Tokens Generated: {avg_metrics.get('tokens_generated', 0):.2f}")
        
        if args.dataset == "govreport":
            print(f"Average ROUGE-1: {avg_metrics.get('rouge1', 0):.4f}")
            print(f"Average ROUGE-2: {avg_metrics.get('rouge2', 0):.4f}")
            print(f"Average ROUGE-L: {avg_metrics.get('rougeL', 0):.4f}")
            print(f"Average F1: {avg_metrics.get('f1', 0):.4f}")
            print(f"Average EM: {avg_metrics.get('exact_match', 0):.4f}")
        else:
            print(f"Average F1: {avg_metrics.get('f1', 0):.4f}")
            print(f"Average EM: {avg_metrics.get('exact_match', 0):.4f}")
        
        if args.compute_ppl:
            print(f"Average Perplexity: {avg_metrics.get('perplexity', 0):.2f}")
        
        # Save average metrics
        avg_csv_file = os.path.join(output_dir, "average_results.csv")
        avg_metrics["dataset"] = args.dataset
        avg_metrics["model"] = args.model_dir
        avg_metrics["num_samples"] = len(samples)
        avg_metrics["kv_cache_method"] = "keygraph"
        log_metrics_to_csv(avg_csv_file, avg_metrics)
        
        print(f"\nAverage metrics saved to: {avg_csv_file}")
        print("=" * 80)


if __name__ == "__main__":
    main()

 



# python3 run_streamingllm_narrativeqa.py     --model_dir "TinyLlama/TinyLlama-1.1B-Chat-v1.0"     --dataset_dir "ccdv/govreport-summarization"     --num_samples 100     --max_new_tokens 256


# python run_keygraph.py \
#     --model_dir "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
#     --dataset "govreport" \
#     --dataset_dir "ccdv/govreport-summarization" \
#     --num_samples 100 \
#     --max_new_tokens 256 \
#     --compute_ppl
# python3 run_keygraph.py \
#     --model_dir "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
#     --dataset "narrativeqa" \
#     --dataset_dir "narrativeqa" \
#     --num_samples 100 \
#     --max_new_tokens 64 \
#     --compute_ppl