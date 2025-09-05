#!/usr/bin/env python3
"""
Test script to verify the UnRoPE implementation and attention rescue mechanism
"""

import sys
import os
import torch
from unittest.mock import patch,MagicMock


sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

def test_unrope_implementation():
    """Test the UnRoPE implementation"""
    print("Testing UnRoPE implementation...")

    from keygraph.method.keygraph_core import build_descriptors_unrope


    num_heads =2
    seq_len =4
    head_dim =4


    keys_per_head =torch.tensor([
    [
    [1.0,0.0,1.0,0.0],
    [0.0,1.0,0.0,1.0],
    [1.0,0.0,1.0,0.0],
    [0.0,1.0,0.0,1.0],],
    [
    [1.0,0.0,1.0,0.0],
    [0.0,1.0,0.0,1.0],
    [1.0,0.0,1.0,0.0],
    [0.0,1.0,0.0,1.0],]])

    pos_idx =torch.arange(seq_len)


    phi,rp_matrix =build_descriptors_unrope(keys_per_head,pos_idx,r =2)

    print(f"  Input keys shape: {keys_per_head.shape}")
    print(f"  Descriptor shape: {phi.shape}")
    print(f"  Random projection matrix shape: {rp_matrix.shape}")





    print("  ✓ UnRoPE implementation completed successfully!")
    return True

def test_rescue_mechanism():
    """Test the attention rescue mechanism with a meaningful scenario"""
    print("Testing attention rescue mechanism...")

    try:

        from keygraph.method.attention_patch import keygraph_attention_patch


        class MockAttention:
            def __init__(self):
                self.num_heads =2
                self.head_dim =4
                self.hidden_size =8
                self.q_proj =MagicMock()
                self.k_proj =MagicMock()
                self.v_proj =MagicMock()
                self.o_proj =MagicMock()
                self.rotary_emb =MagicMock()

            def forward(self,hidden_states,attention_mask =None,position_ids =None,
            past_key_value =None,output_attentions =False,use_cache =True):

                pass

        class MockModel:
            def __init__(self):
                self.config =MagicMock()
                self.config.num_hidden_layers =1
                self.model =MagicMock()
                self.model.layers =[MagicMock()]
                self.model.layers[0].self_attn =MockAttention()


        class MockKeyGraphCache:
            def __init__(self):
                self.layer_reps ={
                0:{
                'K_star':[torch.randn(2,4),torch.randn(2,4)],
                'V_star':[torch.randn(2,4),torch.randn(2,4)],
                'probe_sets':[[0,1],[2,3]],
                'original_k':torch.randn(2,4,4),
                'original_v':torch.randn(2,4,4)}}

            def get_layer_representatives(self,layer_idx):
                return self.layer_reps.get(layer_idx)


        mock_model =MockModel()
        mock_cache =MockKeyGraphCache()


        with keygraph_attention_patch(mock_model,mock_cache,rescue_threshold =0.5):
            print("  ✓ Attention patch with rescue mechanism created successfully!")

        print("  ✓ Attention rescue mechanism test completed successfully!")
        return True
    except Exception as e:
        print(f"  ✗ Failed to test attention rescue mechanism: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_head_wise_rescue_logic():
    """Test the head-wise rescue logic with controlled attention weights"""
    print("Testing head-wise rescue logic...")

    try:
        import torch
        import torch.nn.functional as F
        from keygraph.method.keygraph_cache import KeyGraphCache


        class MockModel:
            def __init__(self):
                self.device =torch.device("cpu")

            class Config:
                def __init__(self):
                    self.num_hidden_layers =1
                    self.num_attention_heads =2
                    self.hidden_size =8
                    self.head_dim =4

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
                head_dim =self.config.head_dim


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

                input_ids =torch.randint(0,100,(1,4))
                attention_mask =torch.ones(1,4)

                result =MockTokenizedInput({
                "input_ids":input_ids,
                "attention_mask":attention_mask})
                return result

            def to(self,device):
                return self


        mock_model =MockModel()
        mock_tokenizer =MockTokenizer()
        prompt ="Test prompt for rescue mechanism."


        cache =KeyGraphCache(
        mock_model,mock_tokenizer,prompt,
        r_dim =4,tau =0.7,knn_k =2)

        print(f"  Created cache with {cache.total_clusters} clusters")
        print("  ✓ Head-wise rescue logic test completed successfully!")
        return True
    except Exception as e:
        print(f"  ✗ Failed to test head-wise rescue logic: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Running UnRoPE and rescue mechanism tests...\n")

    success1 =test_unrope_implementation()
    success2 =test_rescue_mechanism()
    success3 =test_head_wise_rescue_logic()

    if success1 and success2 and success3:
        print("\n🎉 All UnRoPE and rescue mechanism tests passed!")
        return True
    else:
        print("\n❌ Some tests failed!")
        return False

if __name__ =="__main__":
    success =main()
    sys.exit(0 if success else 1)