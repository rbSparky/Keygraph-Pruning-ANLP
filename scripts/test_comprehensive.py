#!/usr/bin/env python3
"""
Comprehensive test script to verify KeyGraph implementation
"""

import sys
import os
import torch


sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

def test_all_components():
    """Test all KeyGraph components"""
    print("Testing all KeyGraph components...")


    try:
        from keygraph.method.keygraph_core import build_descriptors_unrope,build_knn_and_clusters,aggregate_reps


        num_heads =4
        seq_len =10
        head_dim =8
        r_dim =4


        keys_per_head =torch.randn(num_heads,seq_len,head_dim)
        pos_idx =torch.arange(seq_len)


        phi,rp_matrix =build_descriptors_unrope(keys_per_head,pos_idx,r =r_dim)
        assert phi.shape ==(seq_len,r_dim),f"Expected {(seq_len,r_dim)}, got {phi.shape}"


        neighbors,clusters =build_knn_and_clusters(phi,tau =0.5,k =3)
        assert len(clusters)>0,"Should have at least one cluster"


        values_per_head =torch.randn(num_heads,seq_len,head_dim)
        K_star,V_star,cluster_sizes =aggregate_reps(keys_per_head,values_per_head,clusters)
        assert len(K_star)==len(clusters),"Should have one representative per cluster"

        print("  ✓ KeyGraph core functions passed!")
    except Exception as e:
        print(f"  ✗ KeyGraph core functions failed: {e}")
        return False


    try:
        from keygraph.method.keygraph_cache import KeyGraphCache


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


        cache =KeyGraphCache(
        mock_model,mock_tokenizer,prompt,
        r_dim =8,tau =0.7,knn_k =4)
        assert cache.total_clusters >0,"Should have created clusters"
        assert cache.total_positions >0,"Should have processed positions"
        assert 0 <=cache.get_compression_ratio()<=1,"Compression ratio should be between 0 and 1"

        print("  ✓ KeyGraph cache passed!")
    except Exception as e:
        print(f"  ✗ KeyGraph cache failed: {e}")
        import traceback
        traceback.print_exc()
        return False


    try:
        import scripts.run_keygraph
        import scripts.run_baseline
        import scripts.prepare_datasets
        print("  ✓ All scripts imported successfully!")
    except Exception as e:
        print(f"  ✗ Script import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def main():
    """Run all tests"""
    print("Running comprehensive KeyGraph implementation tests...\\n")

    success =test_all_components()

    if success:
        print("\\n🎉 All comprehensive tests passed!")
    else:
        print("\\n❌ Some comprehensive tests failed!")

    return success

if __name__ =="__main__":
    success =main()
    sys.exit(0 if success else 1)