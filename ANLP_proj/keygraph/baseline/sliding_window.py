import torch
import time
from typing import Tuple,Optional


class SlidingWindowKVCache:
    """Sliding window KV cache implementation."""

    def __init__(self,window_size:int =1024):
        self.window_size =window_size
        self.past_key_values =None

    def update(self,new_past_key_values):
        """Update the cache with new KV values, keeping only the last window_size tokens."""
        if new_past_key_values is None:
            self.past_key_values =None
            return

        if self.past_key_values is None:

            self.past_key_values =new_past_key_values
            return


        updated_kv =[]
        for layer_idx,(k,v)in enumerate(zip(new_past_key_values)):

            if k.size(2)>self.window_size:

                k =k[:,:,-self.window_size:,:]
                v =v[:,:,-self.window_size:,:]
            updated_kv.append((k,v))

        self.past_key_values =tuple(updated_kv)

    def get(self):
        """Get the current KV cache."""
        return self.past_key_values


def sliding_window_generate(
model,
tokenizer,
prompt:str,
max_new_tokens:int =128,
window_size:int =1024)->Tuple[str,dict]:
    """
    Generate using sliding window KV cache.

    Returns:
        generated_text (str): The generated text
        metrics (dict): Performance metrics
    """
    start_time =time.time()
    torch.cuda.reset_peak_memory_stats()


    kv_cache =SlidingWindowKVCache(window_size)


    inputs =tokenizer(
    prompt,
    return_tensors ="pt",
    padding =True,
    truncation =True).to(model.device)


    input_ids =inputs.input_ids
    generated_tokens =[]

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs =model(
            input_ids =input_ids,
            past_key_values =kv_cache.get(),
            use_cache =True)


        kv_cache.update(outputs.past_key_values)


        next_token_logits =outputs.logits[:,-1,:]
        next_token =torch.argmax(next_token_logits,dim =-1).unsqueeze(-1)


        generated_tokens.append(next_token.item())


        input_ids =next_token


        if next_token.item()==tokenizer.eos_token_id:
            break

    end_time =time.time()


    num_generated_tokens =len(generated_tokens)
    total_time =end_time -start_time
    tokens_per_second =num_generated_tokens /total_time if total_time >0 else 0
    peak_memory =torch.cuda.max_memory_allocated()/(1024 **2)


    generated_text =tokenizer.decode(generated_tokens,skip_special_tokens =True)

    metrics ={
    "tokens_generated":num_generated_tokens,
    "total_time_seconds":total_time,
    "tokens_per_second":tokens_per_second,
    "peak_vram_mb":peak_memory,
    "kv_cache_method":"sliding_window",
    "window_size":window_size}

    return generated_text,metrics