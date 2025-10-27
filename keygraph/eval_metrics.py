import torch
import time
import argparse
import os
import sys
import re
import string
from tqdm import tqdm
from rouge_score import rouge_scorer

# Assuming your project structure is as you've set it up
# You might need to adjust these imports based on your actual file locations
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# from keygraph.eval_metrics import evaluate_prediction
# from keygraph.dataset.data import GovReportAdapter
# from keygraph.logging_utils import log_metrics_to_csv
# from keygraph.streaming.streaming_llm.enable_streaming_llm import enable_streaming_llm
# from keygraph.models import load_model_and_tokenizer

# NOTE: For this example to be self-contained, I'm including dummy versions
# of the functions you'd import. Replace them with your actual imports.
# ----- DUMMY IMPORTS (REPLACE WITH YOURS) -----
def log_metrics_to_csv(csv_file, metrics):
    """Dummy function for logging. Replace with your actual implementation."""
    # This is just a placeholder to make the script runnable.
    # Your actual function will write to a CSV.
    # print(f"Logging to {csv_file}: {metrics}")
    pass

class GovReportAdapter:
    """Dummy class. Replace with your actual implementation."""
    def __init__(self, dataset_dir):
        print(f"Loading dataset from {dataset_dir} (dummy)")
    def get_samples(self, split, num_samples):
        # Return dummy data
        return [{"id": i, "document": "This is a long document.", "summary": "This is a summary."} for i in range(num_samples)]
    def format_prompt(self, sample):
        return f"Summarize: {sample['document']}"

def load_model_and_tokenizer(model_dir, device):
    """Dummy function. Replace with your actual implementation."""
    # This requires transformers to be installed: pip install transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading dummy model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def enable_streaming_llm(model, start_size, recent_size):
    """Dummy function. Replace with your actual implementation."""
    print(f"Enabling StreamingLLM (dummy) with start={start_size}, recent={recent_size}")
    # The actual function modifies the model. This is a placeholder.
    # We will just return a dummy cache object.
    return lambda past_key_values: past_key_values
# ----- END OF DUMMY IMPORTS -----


# --- Your Provided Metric Functions ---
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

# --- MODIFIED evaluate_prediction FUNCTION ---
def evaluate_prediction(task_type, prediction, ground_truths):
    """Evaluate a prediction based on the task type."""
    if task_type == "summarization":
        if isinstance(ground_truths, list):
            ground_truth = ground_truths[0] if ground_truths else ""
        else:
            ground_truth = ground_truths

        scores = compute_rouge_scores(prediction, ground_truth)
        
        # ADDED: Calculate F1 and EM for summarization
        scores['f1'] = compute_f1(prediction, ground_truth)
        scores['exact_match'] = compute_exact_match(prediction, ground_truth)
        
        return scores

    elif task_type in ["qa", "qasper"]:
        if not isinstance(ground_truths, list):
            ground_truths = [ground_truths]

        em_scores = [compute_exact_match(prediction, gt) for gt in ground_truths]
        f1_scores = [compute_f1(prediction, gt) for gt in ground_truths]

        max_em = max(em_scores) if em_scores else 0.0
        max_f1 = max(f1_scores) if f1_scores else 0.0

        return {'exact_match': max_em, 'f1': max_f1}
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

# --- Your Main Script Logic (with minor addition) ---
@torch.no_grad()
def compute_perplexity(model, tokenizer, text, stride=512):
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
    device = model.device
    metrics = {}
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings - max_new_tokens).to(device)
    input_ids = inputs.input_ids

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    start_time = time.perf_counter()

    past_key_values = None
    prompt_chunk_size = 512
    for i in range(0, input_ids.shape[1], prompt_chunk_size):
        chunk = input_ids[:, i:i + prompt_chunk_size]
        with torch.no_grad():
            outputs = model(input_ids=chunk, past_key_values=past_key_values, use_cache=True)
        past_key_values = kv_cache(outputs.past_key_values)

    generated_ids = []
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values, use_cache=True)
        past_key_values = kv_cache(outputs.past_key_values)
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        if pred_token_idx.item() == tokenizer.eos_token_id:
            break

    end_time = time.perf_counter()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    total_time = end_time - start_time
    tokens_generated = len(generated_ids)
    metrics['tokens_per_second'] = tokens_generated / total_time if total_time > 0 else 0
    metrics['tokens_generated'] = tokens_generated
    metrics['peak_vram_mb'] = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

    return generated_text, metrics

def main(args):
    model, tokenizer = load_model_and_tokenizer(args.model_dir, "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Enabling StreamingLLM with start_size={args.start_size} and recent_size={args.recent_size}")
    kv_cache = enable_streaming_llm(model, start_size=args.start_size, recent_size=args.recent_size)
    dataset_adapter = GovReportAdapter(args.dataset_dir)
    samples = dataset_adapter.get_samples("validation", args.num_samples)
    output_dir = "runs/baseline_streamingllm_govreport"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "results.csv")

    for i, sample in enumerate(tqdm(samples, desc="Processing Samples")):
        print(f"\n--- Processing sample {i + 1}/{len(samples)} ---")
        prompt = dataset_adapter.format_prompt(sample)
        generated_text, metrics = streaming_llm_generate(model, tokenizer, prompt, args.max_new_tokens, kv_cache)
        ground_truth = sample["summary"]
        
        if args.compute_ppl:
            print("Calculating perplexity on ground truth summary...")
            ppl = compute_perplexity(model, tokenizer, ground_truth)
            metrics['perplexity'] = ppl
            print(f"Perplexity: {ppl:.2f}")

        # The call to evaluate_prediction now returns F1 score as well
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
        # *** NEW: Print the F1 score ***
        if 'f1' in metrics: print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Generated text: {generated_text[:100]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run StreamingLLM on GovReport")
    parser.add_argument("--model_dir", type=str, default="gpt2", help="Path or name of model directory")
    parser.add_argument("--dataset_dir", type=str, default="./data/govreport", help="Path to GovReport dataset directory")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of samples")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--start_size", type=int, default=4, help="Number of attention sinks")
    parser.add_argument("--recent_size", type=int, default=512, help="Size of the recent token window")
    parser.add_argument("--compute_ppl", action="store_true", help="Enable perplexity calculation")
    args = parser.parse_args()
    main(args)