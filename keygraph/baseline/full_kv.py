import torch
import time
from typing import Tuple,Optional


def full_kv_generate(
model,
tokenizer,
prompt:str,
max_new_tokens:int =128)->Tuple[str,dict]:
    """
    Generate using full KV cache (baseline).

    Returns:
        generated_text (str): The generated text
        metrics (dict): Performance metrics
    """
    start_time =time.time()
    torch.cuda.reset_peak_memory_stats()


    inputs =tokenizer(
    prompt,
    return_tensors ="pt",
    padding =True,
    truncation =True).to(model.device)


    with torch.no_grad():
        outputs =model.generate(
        **inputs,
        max_new_tokens =max_new_tokens,
        do_sample =False,
        return_dict_in_generate =True,
        output_scores =True,
        past_key_values =None,
        use_cache =True)

    end_time =time.time()


    generated_tokens =outputs.sequences.shape[1]-inputs.input_ids.shape[1]
    total_time =end_time -start_time
    tokens_per_second =generated_tokens /total_time if total_time >0 else 0
    peak_memory =torch.cuda.max_memory_allocated()/(1024 **2)


    generated_text =tokenizer.decode(
    outputs.sequences[0][inputs.input_ids.shape[1]:],
    skip_special_tokens =True)

    metrics ={
    "tokens_generated":generated_tokens,
    "total_time_seconds":total_time,
    "tokens_per_second":tokens_per_second,
    "peak_vram_mb":peak_memory,
    "kv_cache_method":"full"}

    return generated_text,metrics