#!/usr/bin/env python3
"""
Test script to verify KeyGraph implementation components (Phase B/C APIs)
"""

import sys
import os
import torch

HERE = os.path.dirname(__file__)
PARENT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)    # prefer local files keygraph_core.py / keygraph_cache.py
sys.path.insert(0, PARENT)  # fallback for package layout


from keygraph.method.keygraph_core import (
    build_descriptors_unrope,
    build_knn_and_clusters,
    aggregate_reps_from_labels,   # make sure it's exported in that module
)
from keygraph.method.keygraph_cache import KeyGraphCache


def test_keygraph_core():
    """Test the core KeyGraph functions with the new Phase B/C contracts."""
    print("Testing KeyGraph core functions...")

    torch.manual_seed(0)
    num_heads = 4
    seq_len   = 10
    head_dim  = 8
    r_dim     = 4

    # Fake per-head keys/values
    keys_per_head   = torch.randn(num_heads, seq_len, head_dim)
    values_per_head = torch.randn(num_heads, seq_len, head_dim)

    # Descriptors
    pos_idx = torch.arange(seq_len)
    phi, rp_matrix = build_descriptors_unrope(keys_per_head, pos_idx, r=r_dim)
    print(f"  Descriptor shape: {phi.shape}")
    assert phi.shape == (seq_len, r_dim), f"Expected {(seq_len, r_dim)}, got {phi.shape}"

    # kNN + clustering (dict return)
    out = build_knn_and_clusters(phi, tau=0.5, k=3, mutual=True)
    assert isinstance(out, dict), "build_knn_and_clusters must return a dict"
    for key in ("neighbors_idx", "neighbors_sim", "labels"):
        assert key in out, f"Missing key from build_knn_and_clusters: {key}"

    labels = out["labels"]  # [S]
    assert labels.shape == (seq_len,), f"labels wrong shape: {labels.shape}"
    C = int(labels.max().item()) + 1 if labels.numel() else 0
    print(f"  Number of clusters: {C}")
    assert C > 0, "Should have at least one cluster"

    # Aggregation via labels (Phase B path)
    K_star, V_star, cluster_sizes = aggregate_reps_from_labels(keys_per_head, values_per_head, labels)
    print(f"  Representatives shape K*: {tuple(K_star.shape)} / V*: {tuple(V_star.shape)}")
    assert K_star.shape == (num_heads, C, head_dim), f"K_star wrong shape: {K_star.shape}"
    assert V_star.shape == (num_heads, C, head_dim), f"V_star wrong shape: {V_star.shape}"
    assert cluster_sizes.shape == (C,), f"cluster_sizes wrong shape: {cluster_sizes.shape}"

    # Quick numeric sanity for one head/cluster
    c0 = 0
    idx0 = (labels == c0).nonzero(as_tuple=True)[0]
    if idx0.numel() > 0:
        ref = keys_per_head[0, idx0, :].to(torch.float32).mean(dim=0)
        got = K_star[0, c0, :].to(torch.float32)
        # tolerance depends on dtype; this path is CPU fp32, keep it tight
        assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5), "K_star mean mismatch (head 0, cluster 0)"

    print("  ✓ KeyGraph core functions passed!\n")


def test_keygraph_cache():
    """Test the KeyGraph cache (prefill path only)."""
    print("Testing KeyGraph cache...")

    class MockModel:
        def __init__(self):
            self.device = torch.device("cpu")

        class Config:
            def __init__(self):
                self.num_hidden_layers = 2
                self.num_attention_heads = 4
                self.hidden_size = 32

        config = Config()

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            input_ids = kwargs.get("input_ids")
            batch_size, seq_len = input_ids.shape
            num_layers = self.config.num_hidden_layers
            num_heads  = self.config.num_attention_heads
            head_dim   = self.config.hidden_size // num_heads

            past_key_values = []
            for _ in range(num_layers):
                # [B, H, S, Dh] like HF
                k = torch.randn(batch_size, num_heads, seq_len, head_dim)
                v = torch.randn(batch_size, num_heads, seq_len, head_dim)
                past_key_values.append((k, v))

            class MockOutput:
                def __init__(self, pkv):
                    self.past_key_values = pkv

            return MockOutput(tuple(past_key_values))

    class MockTokenizedInput(dict):
        def to(self, device):
            return self

    class MockTokenizer:
        def __init__(self):
            self.pad_token = "<pad>"
            self.eos_token = "</s>"

        def __call__(self, text, return_tensors="pt", padding=True, truncation=True):
            input_ids = torch.randint(0, 100, (1, 20))
            attention_mask = torch.ones(1, 20)
            return MockTokenizedInput({
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })

        def to(self, device):
            return self

    mock_model = MockModel()
    mock_tokenizer = MockTokenizer()
    prompt = "This is a test prompt for KeyGraph pruning."

    try:
        cache = KeyGraphCache(
            mock_model, mock_tokenizer, prompt,
            r_dim=8, tau=0.7, knn_k=4
        )
        # Basic stats that should exist from your implementation
        total_clusters = getattr(cache, "total_clusters", None)
        total_positions = getattr(cache, "total_positions", None)
        print(f"  Total clusters: {total_clusters}")
        print(f"  Total positions: {total_positions}")

        # Compression ratio helper if present; else compute naive ratio if possible
        try:
            cr = cache.get_compression_ratio()
        except Exception:
            if total_clusters is not None and total_positions:
                cr = float(total_clusters) / float(total_positions)
            else:
                cr = float("nan")
        print(f"  Compression ratio: {cr:.4f}")

        print("  ✓ KeyGraph cache created successfully!\n")
    except Exception as e:
        print(f"  ✗ KeyGraph cache creation failed: {e}\n")
        import traceback
        traceback.print_exc()
        raise  # re-raise so main() marks failure


def main():
    """Run all tests"""
    print("Running KeyGraph implementation tests...\n")
    try:
        test_keygraph_core()
        test_keygraph_cache()
        print("🎉 All tests passed!")
        return True
    except Exception as e:
        print(f"❌ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
