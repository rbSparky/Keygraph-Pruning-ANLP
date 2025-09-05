# KeyGraph Pruning - Usage Guide (redundant)

## Overview

This project implements KeyGraph Pruning, an efficient KV cache pruning method for transformer models that uses graph clustering to reduce memory usage while maintaining generation quality.

## Installation

Make sure you have the required dependencies installed:

```bash
pip install torch transformers datasets rouge_score
```

## Usage

### 1. Prepare Datasets

First, verify that your datasets are properly set up:

```bash
python scripts/prepare_datasets.py
```

### 2. Run Baseline Experiments

You can run baseline experiments (Full KV and Sliding Window):

```bash
# Full KV baseline
python scripts/run_baseline.py --model_dir /path/to/model --dataset govreport --dataset_dir /path/to/dataset --baseline full

# Sliding Window baseline
python scripts/run_baseline.py --model_dir /path/to/model --dataset govreport --dataset_dir /path/to/dataset --baseline window --window 1024
```

### 3. Run KeyGraph Pruning

Run experiments using the KeyGraph pruning method:

```bash
python scripts/run_keygraph.py --model_dir /path/to/model --dataset govreport --dataset_dir /path/to/dataset
```

### 4. Configuration

Edit `configs/paths.yaml` to set the paths to your model and datasets:

```yaml
model_dir:  "${HOME}/offline_bundle/models/TinyLlama-1.1B-Chat-v1.0"
gov_dir:    "${HOME}/offline_bundle/datasets/ccdv__govreport_summarization"
nqa_dir:    "${HOME}/offline_bundle/datasets/deepmind__narrativeqa"
qasper_dir: "${HOME}/offline_bundle/datasets/allenai__qasper"

offline: true
local_files_only: true
use_gpu: true
```

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