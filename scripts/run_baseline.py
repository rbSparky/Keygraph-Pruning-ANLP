import os
import sys
import argparse
import json
import yaml
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from keygraph.utils import set_seed, get_device
from keygraph.models import load_model_and_tokenizer
from keygraph.dataset.data import GovReportAdapter, NarrativeQAAdapter, QasperAdapter
from keygraph.baseline.full_kv import full_kv_generate
from keygraph.baseline.sliding_window import sliding_window_generate
from keygraph.eval_metrics import evaluate_prediction
from keygraph.logging_utils import log_metrics_to_csv, log_metrics_to_jsonl


def load_config(config_path="configs/paths.yaml"):
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


def main():
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument("--model_dir", required=True, help="Path to model directory")
    parser.add_argument("--dataset", required=True, choices=["govreport", "narrativeqa", "qasper"],
                        help="Dataset to use")
    parser.add_argument("--dataset_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to process")
    parser.add_argument("--baseline", choices=["full", "window"], required=True,
                        help="Baseline method to use")
    parser.add_argument("--window", type=int, default=1024, help="Window size for sliding window")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum new tokens to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_preds", action="store_true", help="Save predictions to file")
    parser.add_argument("--compute_ppl", action="store_true", help="Enable perplexity calculation")  # ADDED

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(args.model_dir, device)
    dataset_adapter = get_dataset_adapter(args.dataset, args.dataset_dir)
    samples = dataset_adapter.get_samples("test", args.num_samples)

    output_dir = f"runs/baseline_{args.baseline}_{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    csv_file = os.path.join(output_dir, "results.csv")
    preds_file = os.path.join(output_dir, "predictions.jsonl") if args.save_preds else None

    all_metrics = []

    for i, sample in enumerate(samples):
        print(f"\nProcessing sample {i + 1}/{len(samples)}")
        print(f"Sample ID: {sample.get('id', 'N/A')}")

        prompt = dataset_adapter.format_prompt(sample)
        print(f"Prompt length: {len(prompt)} characters")

        if args.baseline == "full":
            generated_text, metrics = full_kv_generate(
                model, tokenizer, prompt, args.max_new_tokens)
        elif args.baseline == "window":
            generated_text, metrics = sliding_window_generate(
                model, tokenizer, prompt, args.max_new_tokens, args.window)

        # ADDED: Calculate perplexity on generated text
        if args.compute_ppl and generated_text:
            print("Calculating perplexity on generated text...")
            ppl = compute_perplexity(model, tokenizer, generated_text)
            metrics['perplexity'] = ppl
            print(f"Perplexity: {ppl:.2f}")

        metrics["sample_id"] = sample.get("id", f"{args.dataset}_{i}")
        metrics["dataset"] = args.dataset
        metrics["baseline"] = args.baseline
        metrics["window_size"] = args.window if args.baseline == "window" else "N/A"
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

        # UPDATED: Print all metrics including latency
        print(f"Generated {metrics['tokens_generated']} tokens")
        print(f"Tokens/sec: {metrics['tokens_per_second']:.2f}")
        print(f"Latency/token: {metrics.get('latency_per_token', 0):.4f}s")  # ADDED
        print(f"Peak VRAM: {metrics['peak_vram_mb']:.2f} MB")
        if "rougeL" in metrics:
            print(f"ROUGE-L: {metrics['rougeL']:.4f}")
        if "f1" in metrics:
            print(f"F1: {metrics['f1']:.4f}")
        if "exact_match" in metrics:
            print(f"EM: {metrics['exact_match']:.4f}")

        print(f"Generated text: {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")

    # UPDATED: Complete average metrics calculation
    if all_metrics:
        print("\n" + "=" * 80)
        print("AVERAGE METRICS ACROSS ALL SAMPLES")
        print("=" * 80)
        
        # Define metrics to average
        metrics_to_average = [
            'tokens_per_second', 'latency_per_token', 'peak_vram_mb', 'tokens_generated'
        ]
        if args.compute_ppl:
            metrics_to_average.append('perplexity')
        
        if args.dataset == "govreport":
            metrics_to_average.extend(['rouge1', 'rouge2', 'rougeL', 'f1', 'exact_match'])
        else:
            metrics_to_average.extend(['exact_match', 'f1'])
        
        # Calculate averages
        avg_metrics = {}
        for metric_name in metrics_to_average:
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            if values:
                avg_metrics[metric_name] = sum(values) / len(values)
        
        # Print all averages
        print(f"Average Tokens/sec: {avg_metrics.get('tokens_per_second', 0):.2f}")
        print(f"Average Latency/token: {avg_metrics.get('latency_per_token', 0):.4f}s")
        print(f"Average Peak VRAM: {avg_metrics.get('peak_vram_mb', 0):.2f} MB")
        print(f"Average Tokens Generated: {avg_metrics.get('tokens_generated', 0):.2f}")
        
        if args.dataset == "govreport":
            print(f"Average ROUGE-1: {avg_metrics.get('rouge1', 0):.4f}")
            print(f"Average ROUGE-2: {avg_metrics.get('rouge2', 0):.4f}")
            print(f"Average ROUGE-L: {avg_metrics.get('rougeL', 0):.4f}")
            print(f"Average F1: {avg_metrics.get('f1', 0):.4f}")
            print(f"Average EM: {avg_metrics.get('exact_match', 0):.4f}")
        else:

            print(f"Average ROUGE-1: {avg_metrics.get('rouge1', 0):.4f}")
            print(f"Average ROUGE-2: {avg_metrics.get('rouge2', 0):.4f}")
            print(f"Average ROUGE-L: {avg_metrics.get('rougeL', 0):.4f}")
            print(f"Average F1: {avg_metrics.get('f1', 0):.4f}")
            print(f"Average EM: {avg_metrics.get('exact_match', 0):.4f}")
        
        if args.compute_ppl:
            print(f"Average Perplexity: {avg_metrics.get('perplexity', 0):.2f}")
        
        # Save average metrics to separate file
        avg_csv_file = os.path.join(output_dir, "average_results.csv")
        avg_metrics["dataset"] = args.dataset
        avg_metrics["baseline"] = args.baseline
        avg_metrics["model"] = args.model_dir
        avg_metrics["num_samples"] = len(samples)
        if args.baseline == "window":
            avg_metrics["window_size"] = args.window
        log_metrics_to_csv(avg_csv_file, avg_metrics)
        
        print(f"\nDetailed results saved to: {csv_file}")
        print(f"Average metrics saved to: {avg_csv_file}")
        print("=" * 80)


if __name__ == "__main__":
    main()





# python3 run_baseline.py \
#     --model_dir "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
#     --dataset "govreport" \
#     --dataset_dir "ccdv/govreport-summarization" \
#     --baseline "full" \
#     --num_samples 100 \
#     --max_new_tokens 256 \
#     --compute_ppl 

# python3 run_baseline.py \
#     --model_dir "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
#     --dataset "narrativeqa" \
#     --dataset_dir "narrativeqa" \
#     --baseline "full" \
#     --num_samples 100 \
#     --max_new_tokens 64 \
#     --compute_ppl
