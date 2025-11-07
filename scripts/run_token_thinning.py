"""
Run Top-K Token Thinning on GovReport, NarrativeQA, and QASPER

- Uses FULL prompt (no chunking in this script).
- Exact VRAM peaks per-phase are computed INSIDE topk_generate.
- Computes full ROUGE (P/R/F) incl. rougeLsum for summarization.
- 4-bit NF4 quantization enabled by default (no dials).
- QASPER support: strict JSON prompting, output parsing, Answer F1 + Evidence F1.
"""

import torch
import time
import argparse
import os
import sys
import re
import string
import csv
import json
import ast
from tqdm import tqdm
from rouge_score import rouge_scorer
from typing import Tuple, Dict, List, Any, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

# Make local repo imports available
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Generation core
from token_thinning import topk_generate

# Datasets
from keygraph.dataset.data import GovReportAdapter, NarrativeQAAdapter
# QASPER: we'll load from HF if adapter isn't available in keygraph
try:
    from keygraph.dataset.data import QasperAdapter as KG_QasperAdapter  # optional
    HAVE_KG_QASPER = True
except Exception:
    KG_QasperAdapter = None
    HAVE_KG_QASPER = False

# Metrics helpers (for evidence F1)
try:
    from keygraph.eval_metrics import best_over_refs_f1
except Exception:
    # Fallback: simple token-level best F1 over refs
    def _tokens(s: str):
        return [t for t in re.sub(r"[^\w\s]", " ", s.lower()).split() if t]
    def _f1(a: str, b: str) -> float:
        A, B = _tokens(a), _tokens(b)
        if not A or not B: return 0.0
        inter = len(set(A) & set(B))
        if inter == 0: return 0.0
        p = inter / len(A); r = inter / len(B)
        return 2*p*r/(p+r)
    def best_over_refs_f1(pred: str, refs: List[str]) -> float:
        return max((_f1(pred, r) for r in refs), default=0.0)

# --------------------------------------------------------------------------------------
# LLaMA-2 chat template (kept from original)
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
        tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE

# --------------------------------------------------------------------------------------
# Text normalization / basic metrics (kept)

def truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    ids = tokenizer(text, truncation=True, max_length=max_tokens, add_special_tokens=False)["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True)

def save_prediction_to_csv(csv_file: str, sample_id: str, input_text: str,
                           prediction: str, reference: str, append: bool = True):
    file_exists = os.path.isfile(csv_file)
    mode = 'a' if append and file_exists else 'w'
    with open(csv_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if mode == 'w' or not file_exists:
            writer.writerow(['sample_id', 'input', 'prediction', 'reference'])
        writer.writerow([sample_id, input_text, prediction, reference])

def normalize_answer(s):
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
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0

def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common_tokens = set(pred_tokens) & set(gt_tokens)
    if len(common_tokens) == 0:
        return 0.0
    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

# --------- full ROUGE incl. Lsum (P/R/F)

def compute_rouge_all(prediction: str, ground_truth: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)
    out = {}
    for name, score in scores.items():
        out[f'{name}_p'] = score.precision
        out[f'{name}_r'] = score.recall
        out[f'{name}_f'] = score.fmeasure
    # convenience aliases (F1)
    out['rouge1'] = out['rouge1_f']
    out['rouge2'] = out['rouge2_f']
    out['rougeL'] = out['rougeL_f']
    out['rougeLsum'] = out['rougeLsum_f']
    return out

# For QA-style metrics (NarrativeQA/QASPER answers)

def evaluate_prediction(task_type, prediction, ground_truths):
    if task_type == "summarization":
        if isinstance(ground_truths, list):
            ground_truth = ground_truths[0] if ground_truths else ""
        else:
            ground_truth = ground_truths
        scores = compute_rouge_all(prediction, ground_truth)
        scores['f1'] = compute_f1(prediction, ground_truth)
        scores['exact_match'] = compute_exact_match(prediction, ground_truth)
        return scores
    elif task_type in ["qa", "qasper", "narrativeqa"]:
        if not isinstance(ground_truths, list):
            ground_truths = [ground_truths]
        em_scores = [compute_exact_match(prediction, gt) for gt in ground_truths]
        f1_scores = [compute_f1(prediction, gt) for gt in ground_truths]
        return {'exact_match': max(em_scores) if em_scores else 0.0,
                'f1': max(f1_scores) if f1_scores else 0.0}
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

# CSV logging

def log_metrics_to_csv(csv_file, metrics):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

# --------------------------------------------------------------------------------------
# Model loading (4-bit NF4 by default)

def _default_bnb_compute_dtype():
    # Prefer bf16 when supported; else fp16
    if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    return torch.float16

def load_model_and_tokenizer(model_dir: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"Loading model and tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=False,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    maybe_set_llama2_chat_template(tokenizer, model_dir)
    quantization_config = None
    tried_quant = False

    if BitsAndBytesConfig is not None and torch.cuda.is_available():
        tried_quant = True
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False,
                bnb_4bit_compute_dtype=_default_bnb_compute_dtype(),
            )
            print("Quantization: 4-bit NF4 (compute dtype auto)")
        except Exception as e:
            print(f"Warning: Could not construct BitsAndBytesConfig: {e}. Falling back to non-quantized.")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=False,
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=None if quantization_config else (torch.float16 if torch.cuda.is_available() else None),
            quantization_config=quantization_config,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        # Hard fallback if bitsandbytes not installed / incompatible
        if tried_quant:
            print(f"Warning: 4-bit NF4 load failed: {e}. Falling back to fp16.")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=False,
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )

    model.eval()
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

# --------------------------------------------------------------------------------------
# Prompt builders

def make_govreport_prompt(tokenizer, document: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a precise scientific summarizer."},
            {"role": "user", "content": "Summarize this government report into 4-6 sentences focusing on the main findings, methods, and implications.\n\n" + document},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Summarize the following government report into 4-6 sentences:\n\n{document}\n\nSummary:"

# QASPER instruction and prompt (STRICT JSON)
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

# QASPER output parsing

def _extract_json_like(text: str) -> Optional[str]:
    try:
        start = text.index("{"); end = text.rindex("}") + 1
        return text[start:end]
    except ValueError:
        return None

def parse_model_output(text: str) -> Tuple[str, List[str]]:
    raw = _extract_json_like(text)
    if raw:
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(raw)
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

# --------------------------------------------------------------------------------------
# QASPER dataset loader (HF fallback)

# We only import datasets if needed to avoid optional dependency errors for other tasks
try:
    from datasets import load_dataset  # type: ignore
except Exception:
    load_dataset = None

# Helpers to normalize QASPER structure

def _get_para_matrix(row: Dict[str, Any]) -> List[List[str]]:
    ft = (row.get("full_text") or {})
    paras = ft.get("paragraphs", []) or []
    norm: List[List[str]] = []
    for sec in paras:
        if isinstance(sec, list):
            norm.append([p for p in sec if isinstance(p, str)])
        elif isinstance(sec, str):
            norm.append([sec])
        else:
            norm.append([])
    return norm

def _flatten_paras(para_matrix: List[List[str]]) -> List[str]:
    out: List[str] = []
    for sec in para_matrix:
        out.extend(sec)
    return out

def _build_fulltext_from_row(row: Dict[str, Any]) -> str:
    title = row.get("title", "")
    abstract = row.get("abstract", "")
    ft = row.get("full_text", {}) or {}
    section_names = ft.get("section_name", []) or []
    paragraphs = ft.get("paragraphs", []) or []

    chunks: List[str] = []
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

def _safe_pick_paragraph(para_matrix: List[List[str]], sec_idx: int, para_idx: int) -> str:
    if 0 <= sec_idx < len(para_matrix):
        sec = para_matrix[sec_idx]
        if 0 <= para_idx < len(sec):
            return sec[para_idx]
    return ""

def _extract_reference_texts(answer_group: Dict[str, Any]) -> List[str]:
    refs: List[str] = []
    answers = (answer_group or {}).get("answer", []) or []
    for a in answers:
        if a.get("unanswerable", False):
            refs.append("UNANSWERABLE"); continue
        yes_no = a.get("yes_no", None)
        if isinstance(yes_no, bool):
            refs.append("yes" if yes_no else "no"); continue
        ffa = (a.get("free_form_answer") or "").strip()
        if ffa:
            refs.append(ffa); continue
        spans = a.get("extractive_spans", []) or []
        if spans:
            refs.append(" ".join(s.strip() for s in spans if isinstance(s, str) and s.strip()))
    seen = set(); uniq: List[str] = []
    for r in refs:
        k = r.strip().lower()
        if k and k not in seen:
            uniq.append(r.strip()); seen.add(k)
    return uniq

def _extract_gold_evidence_texts(answer_group: Dict[str, Any], para_matrix: List[List[str]]) -> List[str]:
    out_sets: List[str] = []
    flattened = _flatten_paras(para_matrix)

    def handle_ev_set(ev_set) -> str:
        pieces: List[str] = []
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
                    t = flattened[idx]
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

    seen = set(); uniq: List[str] = []
    for s in out_sets:
        k = s.strip().lower()
        if k and k not in seen:
            uniq.append(s.strip()); seen.add(k)
    return uniq


def _flatten_qasper_split(ds) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in ds:
        qas = row.get("qas", {}) or {}
        questions = qas.get("question", []) or []
        answers_list = qas.get("answers", []) or []
        qids = qas.get("question_id", []) or []

        para_matrix = _get_para_matrix(row)
        fulltext = _build_fulltext_from_row(row)
        for i, q in enumerate(questions):
            ans_group = answers_list[i] if i < len(answers_list) else {}
            refs = _extract_reference_texts(ans_group)
            ev_refs = _extract_gold_evidence_texts(ans_group, para_matrix)
            qid = qids[i] if i < len(qids) else f"{row.get('id','unknown')}_{i}"
            out.append({
                "doc_id": row.get("id", ""),
                "title": row.get("title", ""),
                "fulltext": fulltext,
                "question_id": qid,
                "question": q,
                "references": refs,
                "evidence_refs": ev_refs,
            })
    return out

class QasperHFAdapter:
    """Adapter that loads QASPER from HF and flattens to per-question samples."""
    def __init__(self, dataset_dir: Optional[str] = None):
        if load_dataset is None:
            raise RuntimeError("datasets library not available but required for QASPER")
        print("Loading QASPER dataset from Hugging Face...")
        self.dataset = load_dataset("allenai/qasper")
        print(f"-> Successfully loaded dataset. Available splits: {list(self.dataset.keys())}")
    def get_samples(self, split: str = "test", num_samples: int = 10) -> List[Dict[str, Any]]:
        if split not in self.dataset:
            raise ValueError(f"Split '{split}' not found in dataset. Available: {list(self.dataset.keys())}")
        flat = _flatten_qasper_split(self.dataset[split])
        return flat[:min(num_samples, len(flat))]

# --------------------------------------------------------------------------------------
# Main

def main(args):
    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    print(f"\nLoading {args.dataset} dataset...")
    if args.dataset == "govreport":
        dataset_adapter = GovReportAdapter(args.dataset_dir)
        task_type = "summarization"
        ground_truth_key = "summary"
        evidence_key = None
    elif args.dataset == "narrativeqa":
        dataset_adapter = NarrativeQAAdapter(args.dataset_dir)
        task_type = "qa"
        ground_truth_key = "answers"
        evidence_key = None
    elif args.dataset == "qasper":
        if HAVE_KG_QASPER and KG_QasperAdapter is not None:
            dataset_adapter = KG_QasperAdapter(args.dataset_dir)  # if user already implemented one
            # We will expect keys similar to our HF adapter; if not, we'll defensively handle below
        else:
            dataset_adapter = QasperHFAdapter(args.dataset_dir)
        task_type = "qa"
        ground_truth_key = "references"
        evidence_key = "evidence_refs"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    samples = dataset_adapter.get_samples(args.split, args.num_samples)
    print(f"Loaded {len(samples)} samples from {args.dataset}")

    output_dir = f"runs/topk_{args.dataset}_k{args.k}_recent{args.recent}_method{args.method}"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "results.csv")

    print(f"\n{'='*80}")
    print("Running Top-K Token Thinning Evaluation (Exact Decode Peak, Full Prompt, 4-bit NF4)")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_dir}")
    print(f"Method: {args.method}")
    print(f"K: {args.k}")
    print(f"Protect Recent: {args.recent}")
    print(f"Max New Tokens: {args.max_new_tokens}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    all_metrics: List[Dict[str, Any]] = []
    predictions_csv = os.path.join(output_dir, f"token_thinning_{args.dataset}_predictions.csv")

    for i, sample in enumerate(tqdm(samples, desc="Processing Samples")):
        print(f"\n--- Sample {i + 1}/{len(samples)} ---")

        # ---------------- GovReport ----------------
        if args.dataset == "govreport":
            input_document = sample.get("report", sample.get("document", ""))
            # Build chat-style prompt for summarization
            prompt = make_govreport_prompt(tokenizer, input_document)
            ground_truth = sample[ground_truth_key]
            sample_id = sample.get("id", f"govreport_{i}")

            generated_text, perf_metrics = topk_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                k=args.k,
                protect_recent=args.recent,
                method=args.method,
                max_new_tokens=args.max_new_tokens,
                compute_ppl=args.compute_ppl,
            )

            save_prediction_to_csv(
                predictions_csv,
                sample_id=sample_id,
                input_text=input_document,
                prediction=generated_text,
                reference=ground_truth,
                append=(i > 0)
            )

            eval_scores = evaluate_prediction("summarization", generated_text, ground_truth)
            metrics = {**perf_metrics, **eval_scores}

        # ---------------- NarrativeQA ----------------
        elif args.dataset == "narrativeqa":
            # Expect adapter to provide a formatted prompt
            if hasattr(dataset_adapter, 'format_prompt'):
                prompt = dataset_adapter.format_prompt(sample)
            else:
                # Fallback minimal prompt
                context = sample.get('document', '')
                question = sample.get('question', '')
                prompt = f"Answer the question using ONLY the context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
            ground_truth = sample[ground_truth_key]
            sample_id = sample.get("id", f"narrativeqa_{i}")

            generated_text, perf_metrics = topk_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                k=args.k,
                protect_recent=args.recent,
                method=args.method,
                max_new_tokens=args.max_new_tokens,
                compute_ppl=args.compute_ppl,
            )

            save_prediction_to_csv(
                predictions_csv,
                sample_id=sample_id,
                input_text=prompt,
                prediction=generated_text,
                reference=ground_truth if isinstance(ground_truth, str) else " | ".join(ground_truth),
                append=(i > 0)
            )

            eval_scores = evaluate_prediction("qa", generated_text, ground_truth)
            metrics = {**perf_metrics, **eval_scores}

        # ---------------- QASPER ----------------
        else:
            # QASPER sample shape (expected):
            # title, fulltext, question, question_id, references(list[str]), evidence_refs(list[str])
            title = sample.get("title", "")
            fulltext = sample.get("fulltext") or sample.get("document", "")
            question = sample.get("question", sample.get("query", ""))
            prompt = make_qasper_prompt(tokenizer, title, fulltext, question)
            sample_id = sample.get("question_id", sample.get("id", f"qasper_{i}"))

            generated_text, perf_metrics = topk_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                k=args.k,
                protect_recent=args.recent,
                method=args.method,
                max_new_tokens=args.max_new_tokens,
                compute_ppl=args.compute_ppl,
            )

            # Parse STRICT JSON answer/evidence
            pred_answer, pred_evidence = parse_model_output(generated_text)

            gold_answers = sample.get(ground_truth_key, sample.get("answers", []))
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]

            eval_scores = evaluate_prediction("qa", pred_answer, gold_answers)

            # Evidence F1 (best over gold evidence sets)
            gold_evidence_list = sample.get(evidence_key, sample.get("evidence", [])) if evidence_key else []
            pred_evs_concat = " ".join(dict.fromkeys(pred_evidence))  # de-duplicate while preserving order
            if gold_evidence_list:
                evidence_f1 = best_over_refs_f1(pred_evs_concat, gold_evidence_list)
            else:
                evidence_f1 = 0.0 if pred_evs_concat else 1.0
            eval_scores['evidence_f1'] = evidence_f1

            # For CSV, store raw model output and gold references for analysis
            save_prediction_to_csv(
                predictions_csv,
                sample_id=sample_id,
                input_text=prompt,
                prediction=generated_text,
                reference=" | ".join(gold_answers),
                append=(i > 0)
            )

            # Merge perf + eval
            metrics = {**perf_metrics, **eval_scores}

        # Common metadata
        metrics["sample_id"] = sample.get("id", sample.get("question_id", f"{args.dataset}_{i}"))
        metrics["baseline"] = "topk_thinning"
        metrics["model"] = args.model_dir
        metrics["dataset"] = args.dataset
        metrics["max_new_tokens"] = args.max_new_tokens
        metrics["topk_k"] = args.k
        metrics["topk_recent"] = args.recent
        metrics["topk_method"] = args.method
        metrics["quantized"] = "4bit_nf4"

        log_metrics_to_csv(csv_file, metrics)
        all_metrics.append(metrics)

        # Console summary
        print(f"Tokens/sec (decode): {metrics.get('tokens_per_second', 0.0):.2f}")
        print(f"Latency/token (decode): {metrics.get('latency_per_token', 0.0):.4f}s")
        if 'peak_vram_prefill_mb' in metrics:
            print(f"Peak VRAM (prefill): {metrics['peak_vram_prefill_mb']:.2f} MB")
        if 'peak_vram_decode_mb' in metrics:
            print(f"Peak VRAM (decode): {metrics['peak_vram_decode_mb']:.2f} MB")
        if 'peak_vram_total_mb' in metrics:
            print(f"Peak VRAM (total):  {metrics['peak_vram_total_mb']:.2f} MB")
        if 'kv_cache_size' in metrics:
            print(f"KV cache size: {metrics['kv_cache_size']} tokens")
        if 'compression_ratio' in metrics:
            print(f"Compression: {metrics['compression_ratio']:.2f}x")

        if args.dataset == "govreport":
            if 'rougeL' in metrics:
                print(f"ROUGE-1/2/L: {metrics.get('rouge1',0):.4f} / {metrics.get('rouge2',0):.4f} / {metrics.get('rougeL',0):.4f}")
                if 'rougeLsum' in metrics:
                    print(f"ROUGE-Lsum: {metrics['rougeLsum']:.4f}")
        else:
            if 'f1' in metrics:
                print(f"Answer F1: {metrics['f1']:.4f}")
            if 'exact_match' in metrics:
                print(f"Exact Match: {metrics['exact_match']:.4f}")
            if args.dataset == "qasper" and 'evidence_f1' in metrics:
                print(f"Evidence F1: {metrics['evidence_f1']:.4f}")

        if 'perplexity' in metrics:
            print(f"Perplexity: {metrics['perplexity']:.2f}")

        shown_text = generated_text if args.dataset == "govreport" else (pred_answer if args.dataset == "qasper" else generated_text)
        print(f"Generated (first 100 chars): {shown_text[:100]}...")

    # Summary
    print(f"\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}")

    if not all_metrics:
        print("No samples processed.")
        return

    n = len(all_metrics)
    avg = lambda k: sum(m.get(k, 0.0) for m in all_metrics) / n

    print(f"\nPerformance:")
    if any('tokens_per_second' in m for m in all_metrics):
        print(f"  Avg Tokens/sec (decode): {avg('tokens_per_second'):.2f}")
    if any('latency_per_token' in m for m in all_metrics):
        print(f"  Avg Latency/token (decode): {avg('latency_per_token'):.4f}s")
    for k in ['peak_vram_prefill_mb','peak_vram_decode_mb','peak_vram_total_mb','kv_cache_size','compression_ratio']:
        if any(k in m for m in all_metrics):
            val = avg(k)
            unit = ' MB' if 'vram' in k else (' tokens' if k=='kv_cache_size' else '')
            print(f"  Avg {k}: {val:.2f}{unit}")

    print(f"\nQuality:")
    if any('f1' in m for m in all_metrics):
        print(f"  Avg Answer F1: {avg('f1'):.4f}")
    if any('exact_match' in m for m in all_metrics):
        print(f"  Avg EM: {avg('exact_match'):.4f}")
    if any('evidence_f1' in m for m in all_metrics):
        print(f"  Avg Evidence F1: {avg('evidence_f1'):.4f}")
    if any('rougeL' in m for m in all_metrics):
        print(f"  Avg ROUGE-1/2/L: {avg('rouge1'):.4f} / {avg('rouge2'):.4f} / {avg('rougeL'):.4f}")
    if any('rougeLsum' in m for m in all_metrics):
        print(f"  Avg ROUGE-Lsum: {avg('rougeLsum'):.4f}")
    if any('perplexity' in m for m in all_metrics):
        ppl_values = [m['perplexity'] for m in all_metrics if 'perplexity' in m]
        if ppl_values:
            print(f"  Avg Perplexity: {sum(ppl_values)/len(ppl_values):.2f}")

    print(f"\n{'='*80}")
    print(f"✓ Results saved to: {csv_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Top-K Token Thinning Evaluation (Exact Decode Peak, Full Prompt, 4-bit NF4)")
    parser.add_argument("--model_dir", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--dataset", type=str, default="govreport",
                        choices=["govreport", "narrativeqa", "qasper"])
    parser.add_argument("--dataset_dir", type=str, default="ccdv/govreport-summarization",
                        help="Path or HF repo id for dataset (GovReport requires this; QASPER ignores if using HF)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--recent", type=int, default=64)
    parser.add_argument("--method", type=str, default="attention",
                        choices=["attention", "uniform", "recency"])
    parser.add_argument("--compute_ppl", action="store_true")
    args = parser.parse_args()
    main(args)
