import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from keygraph.method.keygraph_cache import KeyGraphCache
import math


class RotatoryPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len, base=10000):
        super(RotatoryPositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.base = base
        self._create_rotary_embedding()
    def reset_parameters(self):
        self._create_rotary_embedding()

    def _create_rotary_embedding(self):
        theta = 1.0 / (self.base ** (torch.arange(0, self.embedding_dim, 2)[: (self.embedding_dim // 2)].float() / self.embedding_dim))
        self.register_buffer("theta", theta)
        seq_len = torch.arange(self.max_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum("i,j->ij", seq_len, self.theta).float()
        cache=torch.stack([idx_theta.cos(), idx_theta.sin()], dim=-1)
        self.register_buffer("cache", cache)
    def forward(self, x):
        B, S, H, d_k = x.shape
        x = x.view(B, S, H, d_k // 2, 2)
        rope = self.cache[:S].unsqueeze(0).unsqueeze(2)  
        cos, sin = rope[..., 0], rope[..., 1]
        x1, x2 = x[..., 0], x[..., 1]
        x_rotated = torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return x_rotated.flatten(-2)  





@contextmanager
def keygraph_attention_patch(model,keygraph_cache,rescue_threshold =0.5):
    """
    Context manager to patch attention with KeyGraph representatives.

    Args:
        model: The transformer model
        keygraph_cache: Prebuilt KeyGraphCache
        rescue_threshold: Threshold for rescue expansion
    """

    original_forward_methods ={}
    def create_patched_forward(layer_idx):
        rope = RotatoryPositionalEncoding(model.config.hidden_size // model.config.num_attention_heads, max_len=5000)

        def patched_forward(self,hidden_states,attention_mask =None,position_ids =None,
        past_key_value =None,output_attentions =False,use_cache =True):

            bsz,q_len,_ =hidden_states.size()
            query_states =self.q_proj(hidden_states)
            key_states =self.k_proj(hidden_states)
            value_states =self.v_proj(hidden_states)


            query_states =query_states.view(bsz,q_len,self.num_heads,self.head_dim)
            key_states =key_states.view(bsz,q_len,self.num_heads,self.head_dim)
            value_states =value_states.view(bsz,q_len,self.num_heads,self.head_dim).transpose(1,2)


            kv_seq_len =key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len +=past_key_value[0].shape[-2]

            query_states =rope(query_states)
            key_states =rope(key_states)
            query_states = query_states.transpose(1,2)
            key_states = key_states.transpose(1,2)

            layer_rep =keygraph_cache.get_layer_representatives(layer_idx)
            if layer_rep is not None and past_key_value is not None:

                K_star =layer_rep['K_star']
                V_star =layer_rep['V_star']
                cluster_sizes =layer_rep['cluster_sizes']


                if K_star and len(K_star)>0:


                    if isinstance(K_star[0],torch.Tensor)and K_star[0].dim()==2:

                        K_star_stacked =torch.stack(K_star,dim =1)
                        V_star_stacked =torch.stack(V_star,dim =1)
                    else:

                        K_star_stacked =torch.stack([k.unsqueeze(0)for k in K_star],dim =1)
                        V_star_stacked =torch.stack([v.unsqueeze(0)for v in V_star],dim =1)


                    K_star_stacked =K_star_stacked.unsqueeze(0).expand(bsz,-1,-1,-1)
                    V_star_stacked =V_star_stacked.unsqueeze(0).expand(bsz,-1,-1,-1)


                    past_key_value =(K_star_stacked,V_star_stacked)


            if past_key_value is not None:

                past_key,past_value =past_key_value
                key_states =torch.cat([past_key,key_states],dim =2)
                value_states =torch.cat([past_value,value_states],dim =2)


            attn_weights =torch.matmul(query_states,key_states.transpose(2,3))/(self.head_dim **0.5)


            if attention_mask is not None:
                attn_weights =attn_weights +attention_mask
                attn_weights =torch.max(attn_weights,torch.tensor(torch.finfo(attn_weights.dtype).min))


            attn_weights_softmax =F.softmax(attn_weights,dim =-1,dtype =torch.float32).to(query_states.dtype)


            entropy =-torch.sum(attn_weights_softmax *torch.log(attn_weights_softmax +1e-9),dim =-1)


            final_attn_output =torch.zeros(bsz,self.num_heads,q_len,self.head_dim,device =query_states.device,dtype =query_states.dtype)


            for h in range(self.num_heads):

                head_entropy =entropy[:,h,:]
                needs_rescue =torch.any(head_entropy <rescue_threshold)

                if needs_rescue and layer_rep is not None and 'probe_sets'in layer_rep and 'original_k'in layer_rep and 'original_v'in layer_rep:
                    num_clusters = V_star_stacked.shape[2] 

                    most_attended_cluster_h =torch.argmax(attn_weights_softmax[:,h,:,:],dim =-1)
                    if most_attended_cluster_h < num_clusters:
                        rescued_indices = probe_sets[most_attended_cluster_h]
                    else:
                        # Either skip rescue or handle current tokens differently (optional)
                        rescued_indices = None

                    probe_sets =layer_rep['probe_sets']
                    original_k =layer_rep['original_k']
                    original_v =layer_rep['original_v']
                    

                    probe_indices_set =set()
                    for b in range(bsz):
                        for q in range(q_len):
                            if head_entropy[b,q]<rescue_threshold:
                                cluster_idx =most_attended_cluster_h[b,q].item()
                                if cluster_idx <len(probe_sets):
                                    probe_indices_set.update(probe_sets[cluster_idx])

                    if probe_indices_set:

                        probe_indices =torch.tensor(list(probe_indices_set),device =original_k.device)


                        rescued_k_h =original_k[h,probe_indices,:]
                        rescued_v_h =original_v[h,probe_indices,:]



                        K_star_h =K_star_stacked[:,h,:,:]
                        V_star_h =V_star_stacked[:,h,:,:]


                        rescued_k_h_expanded =rescued_k_h.unsqueeze(0).expand(bsz,-1,-1)
                        rescued_v_h_expanded =rescued_v_h.unsqueeze(0).expand(bsz,-1,-1)


                        expanded_k_h =torch.cat([K_star_h,rescued_k_h_expanded],dim =1)
                        expanded_v_h =torch.cat([V_star_h,rescued_v_h_expanded],dim =1)


                        query_states_h =query_states[:,h,:,:]


                        attn_weights_h =torch.matmul(query_states_h,expanded_k_h.transpose(1,2))/(self.head_dim **0.5)


                        if attention_mask is not None:

                            expanded_attention_mask =F.pad(attention_mask,(0,rescued_k_h_expanded.shape[1]),"constant",0)
                            attn_weights_h =attn_weights_h +expanded_attention_mask
                            attn_weights_h =torch.max(attn_weights_h,torch.tensor(torch.finfo(attn_weights_h.dtype).min))


                        attn_weights_h_softmax =F.softmax(attn_weights_h,dim =-1,dtype =torch.float32).to(query_states.dtype)
                        attn_output_h =torch.matmul(attn_weights_h_softmax,expanded_v_h)
                        final_attn_output[:,h,:,:]=attn_output_h
                    else:
                        concat_values = torch.cat([V_star_stacked, value_states], dim=2)
                        attn_output_h =torch.matmul(attn_weights_softmax[:,h,:,:],concat_values[:,h,:,:])
                        final_attn_output[:,h,:,:]=attn_output_h
                else:
                    concat_values = torch.cat([V_star_stacked, value_states], dim=2)
                    attn_output_h =torch.matmul(attn_weights_softmax[:,h,:,:],concat_values[:,h,:,:])
                    final_attn_output[:,h,:,:]=attn_output_h


            attn_output =final_attn_output.transpose(1,2).contiguous()
            attn_output =attn_output.reshape(bsz,q_len,self.hidden_size)
            attn_output =self.o_proj(attn_output)

            if not output_attentions:
                attn_weights =None

            present_key_value = (key_states, value_states)
            return attn_output, attn_weights, present_key_value

        return patched_forward


    config =model.config
    num_layers =config.num_hidden_layers

    try:

        for i in range(num_layers):
            layer =model.model.layers[i]
            original_forward_methods[i]=layer.self_attn.forward

            layer.self_attn.forward =create_patched_forward(i)

        yield keygraph_cache

    finally:

        for layer_idx,original_method in original_forward_methods.items():
            layer =model.model.layers[layer_idx]
            layer.self_attn.forward =original_method


