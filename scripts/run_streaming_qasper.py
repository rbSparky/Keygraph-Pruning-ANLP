import torch
import time
import argparse
import os
import sys
import csv
import re
import string
import ast
import json
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional, Callable, Dict, List, Any
from datasets import load_dataset  # <-- Added

# --- Assume these imports are correct ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from keygraph.eval_metrics import best_over_refs_f1, evaluate_prediction
from keygraph.dataset.data import GovReportAdapter # Keep for GovReport
from keygraph.logging_utils import log_metrics_to_csv
from keygraph.streaming.streaming_llm.enable_streaming_llm import enable_streaming_llm
from keygraph.models import load_model_and_tokenizer

# LLAMA2 CHAT TEMPLATE (as before)
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
    tmpl = getattr(tokenizer, "chat_template", None)
    mid = (model_id or "").lower()
    if not tmpl and ("llama-2" in mid or "llama2" in mid) and ("instruct" in mid or "chat" in mid):
        print("Setting Llama2 chat template...")
        tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE

# --- QASPER HELPER FUNCTIONS (Copied from qasper_eval.py) ---
# ============================================================================

def get_para_matrix(row: Dict[str, Any]) -> List[List[str]]:
    ft = (row.get("full_text") or {})
    paras = ft.get("paragraphs", []) or []
    norm = []
    for sec in paras:
        if isinstance(sec, list):
            norm.append([p for p in sec if isinstance(p, str)])
        elif isinstance(sec, str):
            norm.append([sec])
        else:
            norm.append([])
    return norm

def flatten_paras(para_matrix: List[List[str]]) -> List[str]:
    out = []
    for sec in para_matrix:
        out.extend(sec)
    return out

def build_fulltext_from_row(row: Dict[str, Any]) -> str:
    title = row.get("title", "")
    abstract = row.get("abstract", "")
    ft = row.get("full_text", {}) or {}
    section_names = ft.get("section_name", []) or []
    paragraphs = ft.get("paragraphs", []) or []

    chunks = []
    if title: chunks.append(f"# {title}\n")
    if abstract: chunks.append("## Abstract\n" + abstract.strip() + "\n")

    n_sections = max(len(section_names), len(paragraphs))
    for i in range(n_sections):
        sname = section_names[i] if i < len(section_names) else f"Section {i+1}"
        chunks.append(f"## {sname}")
        paras = paragraphs[i] if i < len(paragraphs) else []
        if isinstance(paras, list):
            for p in paras:
                if isinstance(p, str) and p.strip():
                    chunks.append(p.strip())
        elif isinstance(paras, str) and paras.strip():
            chunks.append(paras.strip())
        chunks.append("")
    return "\n".join(chunks).strip()

def extract_reference_texts(answer_group: Dict[str, Any]) -> List[str]:
    refs = []
    answers = (answer_group or {}).get("answer", []) or []
    for a in answers:
        if a.get("unanswerable", False):
            refs.append("UNANSWERABLE")
            continue
        yes_no = a.get("yes_no", None)
        if isinstance(yes_no, bool):
            refs.append("yes" if yes_no else "no"); continue
        ffa = (a.get("free_form_answer") or "").strip()
        if ffa:
            refs.append(ffa); continue
        spans = a.get("extractive_spans", []) or []
        if spans:
            refs.append(" ".join(s.strip() for s in spans if isinstance(s, str) and s.strip()))
    seen = set(); uniq = []
    for r in refs:
        k = r.strip().lower()
        if k and k not in seen:
            uniq.append(r.strip()); seen.add(k)
    return uniq

def _safe_pick_paragraph(para_matrix: List[List[str]], sec_idx: int, para_idx: int) -> str:
    if 0 <= sec_idx < len(para_matrix):
        sec = para_matrix[sec_idx]
        if 0 <= para_idx < len(sec):
            return sec[para_idx]
    return ""

def extract_gold_evidence_texts(answer_group: Dict[str, Any], para_matrix: List[List[str]]) -> List[str]:
    out_sets: List[str] = []
    flattened = flatten_paras(para_matrix)

    def handle_ev_set(ev_set) -> str:
        pieces = []
        for item in (ev_set or []):
            if isinstance(item, str):
                if item.strip(): pieces.append(item.strip()); continue
            if isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str) and txt.strip():
                    pieces.append(txt.strip()); continue
                sec = item.get("section"); par = item.get("paragraph")
                if isinstance(sec, int) and isinstance(par, int):
                    t = _safe_pick_paragraph(para_matrix, sec, par)
                    if t: pieces.append(t); continue
                idx = item.get("index")
                if isinstance(idx, int) and 0 <= idx < len(flattened):
                    t = flattened[idx]; 
                    if t: pieces.append(t); continue
            if isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(x, int) for x in item):
                t = _safe_pick_paragraph(para_matrix, item[0], item[1])
                if t: pieces.append(t); continue
            if isinstance(item, int) and 0 <= item < len(flattened):
                t = flattened[item]
                if t: pieces.append(t); continue
        return " ".join(pieces).strip()

    answers = (answer_group or {}).get("answer", []) or []
    any_explicit = False
    for a in answers:
        ev = a.get("evidence", None)
        if not ev: continue
        any_explicit = True
        if isinstance(ev, list) and ev and all(isinstance(x, list) for x in ev):
            for ev_set in ev:
                cat = handle_ev_set(ev_set)
                if cat: out_sets.append(cat)
        elif isinstance(ev, list):
            cat = handle_ev_set(ev)
            if cat: out_sets.append(cat)
    if not any_explicit:
        for a in answers:
            spans = a.get("extractive_spans", []) or []
            if spans:
                cat = " ".join(s.strip() for s in spans if isinstance(s, str) and s.strip())
                if cat: out_sets.append(cat)

    seen = set(); uniq = []
    for s in out_sets:
        k = s.strip().lower()
        if k and k not in seen:
            uniq.append(s.strip()); seen.add(k)
    return uniq

def flatten_qasper_split(ds) -> List[Dict[str, Any]]:
    out = []
    for row in ds:
        qas = row.get("qas", {}) or {}
        questions = qas.get("question", []) or []
        answers_list = qas.get("answers", []) or []
        qids = qas.get("question_id", []) or []

        para_matrix = get_para_matrix(row)
        fulltext = build_fulltext_from_row(row)
        for i, q in enumerate(questions):
            ans_group = answers_list[i] if i < len(answers_list) else {}
            refs = extract_reference_texts(ans_group)
            ev_refs = extract_gold_evidence_texts(ans_group, para_matrix)
            qid = qids[i] if i < len(qids) else f"{row.get('id','unknown')}_{i}"
            out.append({
                "doc_id": row.get("id", ""),
                "title": row.get("title", ""),
                "fulltext": fulltext,
                "question_id": qid,
                "question": q,
                "references": refs,       # gold answers (strings)
                "evidence_refs": ev_refs,  # gold evidence strings (each is one set)
            })
    return out

# --- NEW QASPER ADAPTER ---
class QasperAdapter:
    """Adapter for the QASPER dataset."""
    def __init__(self, dataset_dir: Optional[str] = None):
        # dataset_dir is ignored, we load from HF
        print("Loading QASPER dataset from Hugging Face...")
        self.dataset = load_dataset("allenai/qasper")
        print(f"-> Successfully loaded dataset. Available splits: {list(self.dataset.keys())}")

    def get_samples(self, split: str = "test", num_samples: int = 10) -> List[Dict[str, Any]]:
        """Flatten QASPER dataset into individual question samples."""
        if split not in self.dataset:
            raise ValueError(f"Split '{split}' not found in dataset. Available: {list(self.dataset.keys())}")
        
        flat_samples = flatten_qasper_split(self.dataset[split])
        return flat_samples[:min(num_samples, len(flat_samples))]

# --- PROMPTING ---
# ============================================================================

# 1. GovReport Prompter
def make_govreport_prompt(tokenizer, document: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a precise scientific summarizer."},
            {"role": "user", "content": "Summarize this government report into 4-6 sentences focusing on the main findings, methods, and implications.\n\n" + document},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Summarize the following government report into 4-6 sentences:\n\n{document}\n\nSummary:"

# 2. QASPER Prompter (from qasper_eval.py)
INSTRUCT_JSON = (
    "You are a precise scientific QA assistant. Use only the provided paper text. "
    "If the question is not answerable from the text, reply with answer 'UNANSWERABLE'.\n\n"
    "Return a STRICT JSON object with keys exactly:\n"
    '{ "answer": "<string>", "evidence": ["<verbatim span 1>", "<verbatim span 2>"] }\n'
    "- 'answer' must be 1-3 sentences or 'UNANSWERABLE'.\n"
    "- 'evidence' must be 0-5 verbatim snippets copied from the paper text (no paraphrases).\n"
    "No extra commentary."
)

def make_qasper_prompt(tokenizer, title: str, fulltext: str, question: str) -> str:
    content = (f"{INSTRUCT_JSON}\n\n"
               f"Paper title: {title}\n\nPaper text:\n{fulltext}\n\n"
               f"Question: {question}\nJSON:")
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role":"system","content":"You are a precise scientific QA assistant."},
                    {"role":"user","content":content}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return content

# 3. QASPER Output Parser (from qasper_eval.py)
def _extract_json_like(text: str) -> str | None:
    try:
        start = text.index("{"); end = text.rindex("}") + 1
        return text[start:end]
    except ValueError:
        return None

def parse_model_output(text: str) -> Tuple[str, List[str]]:
    """
    Returns (answer, evidence_list). Evidence list may be empty if parsing fails.
    """
    raw = _extract_json_like(text)
    if raw:
        try:
            obj = json.loads(raw)
            ans = (obj.get("answer") or "").strip()
            ev = obj.get("evidence") or []
            ev = [e.strip() for e in ev if isinstance(e, str) and e.strip()]
            return ans, ev
        except Exception:
            try:
                obj = ast.literal_eval(raw)
                ans = (obj.get("answer") or "").strip()
                ev = obj.get("evidence") or []
                ev = [e.strip() for e in ev if isinstance(e, str) and e.strip()]
                return ans, ev
            except Exception:
                pass
    # Heuristic fallback
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    ans = lines[0][:512] if lines else ""
    ev = re.findall(r'"([^"]{10,400})"', text)
    ev = [s.strip() for s in ev][:5]
    return ans, ev

# --- UTILITY FUNCTIONS (as before) ---
# ============================================================================

@torch.no_grad()
def compute_perplexity(model, tokenizer, text, stride=512):
    # (Your compute_perplexity function... no changes)
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

def _cuda_phase_begin():
    # (Your _cuda_phase_begin function... no changes)
    if not torch.cuda.is_available():
        return 0, 0
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()
    return base_alloc, base_reserved

def _cuda_phase_end(base_alloc, base_reserved):
    # (Your _cuda_phase_end function... no changes)
    if not torch.cuda.is_available():
        return {
            "peak_alloc_MB": float("nan"), "peak_reserved_MB": float("nan"),
            "delta_alloc_MB": float("nan"), "delta_reserved_MB": float("nan"),
            "base_alloc_MB": float("nan"), "base_reserved_MB": float("nan"),
        }
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    abs_alloc = max(peak_alloc, base_alloc)
    abs_reserved = max(peak_reserved, base_reserved)
    return {
        "peak_alloc_MB": round(abs_alloc / (1024**2), 2),
        "peak_reserved_MB": round(abs_reserved / (1024**2), 2),
        "delta_alloc_MB": round(max(0, peak_alloc - base_alloc) / (1024**2), 2),
        "delta_reserved_MB": round(max(0, peak_reserved - base_reserved) / (1024**2), 2),
        "base_alloc_MB": round(base_alloc / (1024**2), 2),
        "base_reserved_MB": round(base_reserved / (1024**2), 2),
    }

def save_prediction_to_csv(csv_file: str, sample_id: str, input_text: str, 
                          prediction: str, reference: str, append: bool = True):
    # (Your save_prediction_to_csv function... no changes)
    file_exists = os.path.isfile(csv_file)
    mode = 'a' if append and file_exists else 'w'
    with open(csv_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if mode == 'w' or not file_exists:
            writer.writerow(['sample_id', 'input', 'prediction', 'reference'])
        writer.writerow([sample_id, input_text, prediction, reference])
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
    pref_base_alloc, pref_base_reserved = _cuda_phase_begin()
    t_prefill_start = time.perf_counter()

    try:
        # ... your existing prefill code (chunked passes, kv_cache updates) ...
        prefill_time = time.perf_counter() - t_prefill_start
        prefill_stats = {"time_s": round(prefill_time, 3), **_cuda_phase_end(pref_base_alloc, pref_base_reserved)}
        print("Prompt processing complete.")
    except Exception as e:
        prefill_stats = {"time_s": round(time.perf_counter() - t_prefill_start, 3), **_cuda_phase_end(pref_base_alloc, pref_base_reserved)}

    
    # CRITICAL FIX: Don't initialize past_key_values to None
    # Let the first forward pass create it
    prompt_chunk_size = 5000
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
    # --- DECODE PHASE ---
    dec_base_alloc, dec_base_reserved = _cuda_phase_begin()
    t_decode_start = time.perf_counter()

    generated_ids = []
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    try:
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            past_key_values = kv_cache(outputs.past_key_values)
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            token_id = pred_token_idx.item()
            generated_ids.append(token_id)
            if token_id == tokenizer.eos_token_id:
                break

        decode_time = time.perf_counter() - t_decode_start
        decode_stats = {"time_s": round(decode_time, 3), **_cuda_phase_end(dec_base_alloc, dec_base_reserved)}
    except Exception as e:
        decode_time = time.perf_counter() - t_decode_start
        decode_stats = {"time_s": round(decode_time, 3), **_cuda_phase_end(dec_base_alloc, dec_base_reserved)}

    
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

def main(args):
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_dir, 
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    maybe_set_llama2_chat_template(tokenizer, args.model_dir)
    
    # Enable StreamingLLM
    print(f"Enabling StreamingLLM with start_size={args.start_size} and recent_size={args.recent_size}")
    kv_cache = enable_streaming_llm(
        model,
        start_size=args.start_size,
        recent_size=args.recent_size
    )
    
    # --- NEW: Conditional Dataset Loading ---
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == "govreport":
        if not args.dataset_dir:
            print("ERROR: --dataset_dir is required for govreport.")
            sys.exit(1)
        dataset_adapter = GovReportAdapter(args.dataset_dir)
        task_type = "summarization"
        samples = dataset_adapter.get_samples("test", args.num_samples)
    
    elif args.dataset == "qasper":
        # dataset_dir is optional for qasper, loads from HF
        dataset_adapter = QasperAdapter(args.dataset_dir) 
        task_type = "qa"
        samples = dataset_adapter.get_samples("test", args.num_samples) # QasperAdapter handles this
    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Setup output
    output_dir = f"runs/baseline_streamingllm_{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "results.csv")
    predictions_csv = os.path.join(output_dir, f"streamingllm_{args.dataset}_predictions.csv")
    
    # --- Process samples ---
    for i, sample in enumerate(tqdm(samples, desc=f"Processing {args.dataset} Samples")):
        try:
            print(f"\n--- Processing sample {i + 1}/{len(samples)} ---")
            
            # --- NEW: Conditional Prompting & GT ---
            if task_type == "summarization":
                input_document = sample.get("report", "")
                ground_truth_list = [sample.get("summary", "")]
                prompt = make_govreport_prompt(tokenizer, input_document)
                sample_id = sample.get("id", f"govreport_{i}")
                
            elif task_type == "qa":
                input_document = sample["fulltext"]
                ground_truth_list = sample["references"] # Gold answers
                gold_evidence_list = sample["evidence_refs"] # <-- GET GOLD EVIDENCE
                prompt = make_qasper_prompt(tokenizer, sample["title"], input_document, sample["question"])
                sample_id = sample.get("question_id", f"qasper_{i}")
            
            else:
                continue # Should not happen

            # Generate text
            generated_text, metrics = streaming_llm_generate(
                model, tokenizer, prompt, args.max_new_tokens, kv_cache
            )
            
            # --- NEW: Conditional Output Parsing ---
            if task_type == "summarization":
                pred_answer = generated_text
            elif task_type == "qa":
                pred_answer, pred_evidence = parse_model_output(generated_text)
                
                # 1. Calculate Answer F1
                eval_scores = evaluate_prediction("qa", pred_answer, ground_truth_list)
                
                # 2. Calculate Evidence F1 (THE NEW PART)
                pred_evs_concat = " ".join(dict.fromkeys(pred_evidence)) # De-duplicate
                if gold_evidence_list:
                    evidence_f1 = best_over_refs_f1(pred_evs_concat, gold_evidence_list)
                else:
                    evidence_f1 = 0.0 if pred_evs_concat else 1.0 # Handle no-gold-evidence cases
                
                eval_scores['evidence_f1'] = evidence_f1 # Add to scores
            metrics.update(eval_scores)
            # Save raw prediction
            save_prediction_to_csv(
               predictions_csv,
               sample_id=sample_id,
               input_text=prompt, # Save the full prompt for context
               prediction=generated_text, # Save the raw model output
               reference=" | ".join(ground_truth_list), # Join all references
               append=(i > 0)
            )
            
            # Compute perplexity (if requested)
            if args.compute_ppl and len(pred_answer) > 50:
                print(f"Calculating perplexity on generated answer...")
                try:
                    ppl = compute_perplexity(model, tokenizer, pred_answer)
                    metrics['perplexity'] = ppl
                    print(f"Perplexity: {ppl:.2f}")
                except Exception as e:
                    print(f"Warning: Could not compute perplexity: {e}")
                    metrics['perplexity'] = float('inf')
            
            # --- NEW: Conditional Evaluation ---
            # We assume 'evaluate_prediction' can handle both tasks
            # based on the 'task_type' string.
            print(f"Evaluating task type: {task_type}")
            if task_type == "summarization":
                eval_scores = evaluate_prediction("summarization", pred_answer, ground_truth_list[0])
            elif task_type == "qa":
                # Assumes evaluate_prediction knows how to handle lists of refs for "qa"
                eval_scores = evaluate_prediction("qa", pred_answer, ground_truth_list)

            metrics.update(eval_scores)
            
            # Add metadata
            metrics["sample_id"] = sample_id
            metrics["baseline"] = "streamingllm"
            metrics["dataset"] = args.dataset
            metrics["model"] = args.model_dir
            metrics["max_new_tokens"] = args.max_new_tokens
            metrics["streaming_start_size"] = args.start_size
            metrics["streaming_recent_size"] = args.recent_size
            
            # Log to CSV
            log_metrics_to_csv(csv_file, metrics)
            
            # Print results (all metrics are now included)
            print(f"Tokens/sec: {metrics['tokens_per_second']:.2f}")
            print(f"Latency/token: {metrics['latency_per_token']:.4f}s")
            print(f"Prefill PeakVRAM: {metrics['prefill_peak_alloc_MB']:.2f} MB")
            print(f"Decode PeakVRAM: {metrics['decode_peak_alloc_MB']:.2f} MB")
            print(f"Tokens Generated: {metrics['tokens_generated']}")
            if 'perplexity' in metrics: print(f"Perplexity: {metrics['perplexity']:.2f}")
            if 'rouge1' in metrics: print(f"ROUGE-1: {metrics['rouge1']:.4f}")
            if 'rouge2' in metrics: print(f"ROUGE-2: {metrics['rouge2']:.4f}")
            if 'rougeL' in metrics: print(f"ROUGE-L: {metrics['rougeL']:.4f}")
            if 'rougeLsum' in metrics: print(f"ROUGE-Lsum: {metrics['rougeLsum']:.4f}")
            if 'f1' in metrics:
                print(f"Answer F1 Score: {metrics['f1']:.4f}")
            if 'evidence_f1' in metrics:
                print(f"Evidence F1 Score: {metrics['evidence_f1']:.4f}")
            if 'exact_match' in metrics: print(f"Exact Match: {metrics['exact_match']:.4f}")
            
            print(f"Generated Answer: {pred_answer[:100]}...")
            
        except Exception as e:
            print(f"FATAL Error processing sample {i} (sample_id: {sample_id}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✓ Results saved to: {csv_file}")
    print(f"✓ Predictions saved to: {predictions_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run StreamingLLM on GovReport or QASPER")
    
    # --- MODIFIED ARGS ---
    parser.add_argument("--dataset", type=str, required=True, choices=["govreport", "qasper"],
                        help="Dataset to run evaluation on.")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Path to dataset directory (REQUIRED for govreport, ignored for qasper).")
    # --- END MODIFIED ARGS ---
    
    parser.add_argument("--model_dir", required=True, help="Path to model directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--start_size", type=int, default=4, help="Number of attention sinks")
    parser.add_argument("--recent_size", type=int, default=1020, help="Size of the recent token window")
    parser.add_argument("--compute_ppl", action="store_true", help="Enable perplexity calculation")
    
    args = parser.parse_args()
    main(args)



# python3 run_streaming_qasper.py \
#     --dataset qasper \
#     --model_dir "togethercomputer/LLaMA-2-7B-32K-Instruct" \
#     --num_samples 50 \
#     --max_new_tokens 192 \
#     --start_size 4 \
#     --recent_size 1020