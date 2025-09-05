import torch
import numpy as np
from keygraph.method.keygraph_core import build_descriptors_unrope,build_knn_and_clusters,aggregate_reps


class KeyGraphCache:
    """
    Cache for KeyGraph representatives, cluster sizes, and member lists.
    Built once after prefill.
    """

    def __init__(self,model,tokenizer,prompt,r_dim =32,tau =0.8,knn_k =16,
    rescue =True,rescue_probe_size =6,upper_layers_only =False):
        self.model =model
        self.tokenizer =tokenizer
        self.r_dim =r_dim
        self.tau =tau
        self.knn_k =knn_k
        self.rescue =rescue
        self.rescue_probe_size =rescue_probe_size
        self.upper_layers_only =upper_layers_only


        self._build_cache(prompt)

    def _build_cache(self,prompt):
        """
        Build the KeyGraph cache from the prompt.
        """

        inputs =self.tokenizer(
        prompt,
        return_tensors ="pt",
        padding =True,
        truncation =True).to(self.model.device)


        config =self.model.config
        num_layers =config.num_hidden_layers
        num_heads =config.num_attention_heads
        head_dim =config.hidden_size //num_heads


        if self.upper_layers_only:

            start_layer =(2 *num_layers)//3
            layers_to_process =list(range(start_layer,num_layers))
        else:
            layers_to_process =list(range(num_layers))


        with torch.no_grad():
            outputs =self.model(
            **inputs,
            output_attentions =False,
            output_hidden_states =False,
            use_cache =True,
            return_dict =True)


        past_key_values =outputs.past_key_values


        self.layer_reps ={}
        self.rp_matrices ={}
        self.total_clusters =0
        self.total_positions =0


        for layer_idx in layers_to_process:
            k,v =past_key_values[layer_idx]


            k =k.squeeze(0)
            v =v.squeeze(0)

            seq_len =k.shape[1]
            self.total_positions +=seq_len


            pos_idx =torch.arange(seq_len,device =k.device)


            phi,rp_matrix =build_descriptors_unrope(k,pos_idx,r =self.r_dim)
            self.rp_matrices[layer_idx]=rp_matrix


            neighbors,clusters =build_knn_and_clusters(
            phi,tau =self.tau,k =self.knn_k)

            self.total_clusters +=len(clusters)


            representatives =[]
            probe_sets =[]
            for cluster in clusters:


                rep =cluster[0]
                representatives.append(rep)


                probe_size =min(self.rescue_probe_size,len(cluster))
                probe_set =np.random.choice(cluster,size =probe_size,replace =False)
                probe_sets.append(probe_set.tolist())


            K_star,V_star,cluster_sizes =aggregate_reps(k,v,clusters)


            self.layer_reps[layer_idx]={
            'K_star':K_star,
            'V_star':V_star,
            'clusters':clusters,
            'cluster_sizes':cluster_sizes,
            'representatives':representatives,
            'probe_sets':probe_sets,
            'original_k':k,
            'original_v':v}


        self.full_kv_layers =[i for i in range(num_layers)if i not in layers_to_process]
        if self.full_kv_layers:
            self.full_past_key_values =tuple(past_key_values[i]for i in self.full_kv_layers)
        else:
            self.full_past_key_values =None

    def get_layer_representatives(self,layer_idx):
        """Get representatives for a specific layer."""
        if layer_idx in self.layer_reps:
            return self.layer_reps[layer_idx]
        return None

    def get_compression_ratio(self):
        """Get the estimated compression ratio."""
        if self.total_positions ==0:
            return 0
        return self.total_clusters /self.total_positions

    def get_kv_bytes_saved(self):
        """Estimate KV cache bytes saved."""


        config =self.model.config
        head_dim =config.hidden_size //config.num_attention_heads
        kv_size_per_token =2 *config.num_attention_heads *head_dim *4

        original_size =self.total_positions *kv_size_per_token
        compressed_size =self.total_clusters *kv_size_per_token

        return original_size -compressed_size