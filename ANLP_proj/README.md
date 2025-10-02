# KeyGraph Pruning

An efficient KV cache pruning method for transformer models that uses graph clustering to reduce memory usage while maintaining generation quality.

## Overview

This project implements KeyGraph Pruning, a novel technique for reducing the memory footprint of transformer KV caches during generation. The method works by:

1. Building a keygraph from the prompt's KV cache
2. Clustering similar keys using random projections and kNN
3. Using cluster representatives instead of full keys during generation
4. Optionally expanding clusters during rescue when needed

## Project Structure

```
keygraph-pruning/
  README.md
  scripts/
    prepare_datasets.py        # verifies local datasets (no downloads)
    run_baseline.py            # Full KV + Sliding Window baselines
    run_keygraph.py            # our method
    eval_metrics.py            # ROUGE/EM/F1/PPL
    profile_gpu.py             # GPU + torch info
  keygraph/
    __init__.py
    utils.py
    logging_utils.py
    data.py                    # dataset adapters for GOV/NQA/QASPER
    models.py                  # load TinyLlama + generate helper
    baseline/
      full_kv.py
      sliding_window.py
    method/
      keygraph_core.py         # Phase A: descriptors, kNN, clusters
      keygraph_cache.py        # per-layer representative cache
      attention_patch.py       # Phase B: representative attention + log-mass
  configs/
    paths.yaml                 # absolute paths to MODEL_DIR, GOV_DIR, NQA_DIR, QASPER_DIR
    experiments.yaml           # experiment grids for ablations
  runs/                        # auto-created for logs/artifacts
```

## Local Paths

All experiments run fully offline using locally stored assets:

- **Model (TinyLlama, Transformers format):**
  `MODEL_DIR = $HOME/offline_bundle/models/TinyLlama-1.1B-Chat-v1.0`
- **Datasets (Hugging Face local snapshots):**
  - GovReport summarization: `GOV_DIR = $HOME/offline_bundle/datasets/ccdv__govreport_summarization`
  - NarrativeQA: `NQA_DIR = $HOME/offline_bundle/datasets/deepmind__narrativeqa`
  - QASPER: `QASPER_DIR = $HOME/offline_bundle/datasets/allenai__qasper`

## Commands

### GPU/Profile info

```bash
python keygraph-pruning/scripts/profile_gpu.py
```

### Verify datasets (no downloads)

```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python keygraph-pruning/scripts/prepare_datasets.py \
  --gov_dir   "$HOME/offline_bundle/datasets/ccdv__govreport_summarization" \
  --nqa_dir   "$HOME/offline_bundle/datasets/deepmind__narrativeqa" \
  --qasper_dir "$HOME/offline_bundle/datasets/allenai__qasper"
```

### Baselines (GovReport, TinyLlama)

```bash
MODEL_DIR="$HOME/offline_bundle/models/TinyLlama-1.1B-Chat-v1.0"
GOV_DIR="$HOME/offline_bundle/datasets/ccdv__govreport_summarization"

python keygraph-pruning/scripts/run_baseline.py \
  --model_dir "$MODEL_DIR" --dataset govreport --dataset_dir "$GOV_DIR" \
  --baseline full --num_samples 5 --max_new_tokens 128

python keygraph-pruning/scripts/run_baseline.py \
  --model_dir "$MODEL_DIR" --dataset govreport --dataset_dir "$GOV_DIR" \
  --baseline window --window 1024 --num_samples 5 --max_new_tokens 128
```

### Our method (GovReport, TinyLlama)

```bash
python keygraph-pruning/scripts/run_keygraph.py \
  --model_dir "$MODEL_DIR" --dataset govreport --dataset_dir "$GOV_DIR" \
  --num_samples 5 --max_new_tokens 128 --r_dim 32 --knn_k 16 --tau 0.8 --rescue true
```

## Results

Results are saved to the `runs/` directory with metrics logged in CSV format.
