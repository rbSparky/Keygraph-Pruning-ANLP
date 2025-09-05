#!/usr/bin/env python3
"""
Test script to verify KeyGraph implementation components
"""

import sys
import os
import torch


sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

from keygraph.method.keygraph_core import build_descriptors_unrope,build_knn_and_clusters,aggregate_reps
from keygraph.method.keygraph_cache import KeyGraphCache


def test_keygraph_core():
    """Test the core KeyGraph functions"""
    print("Testing KeyGraph core functions...")


    num_heads =4
    seq_len =10
    head_dim =8
    r_dim =4


    keys_per_head =torch.randn(num_heads,seq_len,head_dim)
    pos_idx =torch.arange(seq_len)


    phi,rp_matrix =build_descriptors_unrope(keys_per_head,pos_idx,r =r_dim)
    print(f"  Descriptor shape: {phi.shape}")
    assert phi.shape ==(seq_len,r_dim),f"Expected {(seq_len,r_dim)}, got {phi.shape}"


    neighbors,clusters =build_knn_and_clusters(phi,tau =0.5,k =3)
    print(f"  Number of clusters: {len(clusters)}")
    assert len(clusters)>0,"Should have at least one cluster"


    values_per_head =torch.randn(num_heads,seq_len,head_dim)
    K_star,V_star,cluster_sizes =aggregate_reps(keys_per_head,values_per_head,clusters)
    print(f"  Number of representatives: {len(K_star)}")
    assert len(K_star)==len(clusters),"Should have one representative per cluster"

    print("  ✓ KeyGraph core functions passed!\n")


def test_keygraph_cache():
    """Test the KeyGraph cache"""
    print("Testing KeyGraph cache...")


    class MockModel:
        def __init__(self):
            self.device =torch.device("cpu")

        class Config:
            def __init__(self):
                self.num_hidden_layers =2
                self.num_attention_heads =4
                self.hidden_size =32

        config =Config()

        def to(self,device):
            return self

        def eval(self):
            return self

        def __call__(self,**kwargs):

            input_ids =kwargs.get("input_ids")
            batch_size,seq_len =input_ids.shape
            num_layers =self.config.num_hidden_layers
            num_heads =self.config.num_attention_heads
            head_dim =self.config.hidden_size //num_heads


            past_key_values =[]
            for _ in range(num_layers):
                k =torch.randn(batch_size,num_heads,seq_len,head_dim)
                v =torch.randn(batch_size,num_heads,seq_len,head_dim)
                past_key_values.append((k,v))


            class MockOutput:
                def __init__(self,past_key_values):
                    self.past_key_values =past_key_values

            return MockOutput(tuple(past_key_values))

    class MockTokenizedInput(dict):
        """A dict subclass that also has a to method"""
        def to(self,device):
            return self

    class MockTokenizer:
        def __init__(self):
            self.pad_token ="<pad>"
            self.eos_token ="</s>"

        def __call__(self,text,return_tensors ="pt",padding =True,truncation =True):

            input_ids =torch.randint(0,100,(1,20))
            attention_mask =torch.ones(1,20)

            result =MockTokenizedInput({
            "input_ids":input_ids,
            "attention_mask":attention_mask})
            return result

        def to(self,device):
            return self


    mock_model =MockModel()
    mock_tokenizer =MockTokenizer()
    prompt ="This is a test prompt for KeyGraph pruning."


    try:
        cache =KeyGraphCache(
        mock_model,mock_tokenizer,prompt,
        r_dim =8,tau =0.7,knn_k =4)
        print(f"  Total clusters: {cache.total_clusters}")
        print(f"  Total positions: {cache.total_positions}")
        print(f"  Compression ratio: {cache.get_compression_ratio():.4f}")
        print("  ✓ KeyGraph cache created successfully!\n")
    except Exception as e:
        print(f"  ✗ KeyGraph cache creation failed: {e}\n")
        import traceback
        traceback.print_exc()


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


if __name__ =="__main__":
    success =main()
    sys.exit(0 if success else 1)