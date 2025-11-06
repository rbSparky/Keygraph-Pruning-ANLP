import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional

@torch.no_grad()
def full_kv_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 128
) -> Tuple[str, dict]:
    """
    Generates text using the model's standard KV cache.
    MODIFIED: This version DOES NOT TRUNCATE the prompt.
    This is intended to cause a CUDA OOM error on long contexts.
    """
    device = model.device
    metrics = {}

    # 1. Tokenize the prompt WITHOUT TRUNCATION
    print("Tokenizing full prompt (no truncation)...")
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,       # No padding, single sample
        truncation=False,    # <--- MODIFIED
        max_length=None      # <--- MODIFIED
    ).to(device)
    
    input_ids = inputs.input_ids
    prompt_tokens = input_ids.shape[1]
    print(f"Full prompt token count: {prompt_tokens}")

    # Check if prompt is too long for the model's max position
    max_pos = model.config.max_position_embeddings
    if prompt_tokens > max_pos:
        print(f"WARNING: Prompt ({prompt_tokens} tokens) exceeds model's max position embeddings ({max_pos}).")
        # Truncate only if it exceeds model's absolute limit
        input_ids = input_ids[:, -max_pos:]
        print(f"Truncated to {input_ids.shape[1]} tokens to fit model max length.")

    # 2. Reset VRAM and start timer
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    
    start_time = time.perf_counter()

    # 3. Generate text
    # This single call will attempt to build the *full* KV cache in memory
    try:
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            use_cache=True
        )
        
        end_time = time.perf_counter()
        
        # Decode generated text
        generated_ids = outputs.sequences[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        tokens_generated = len(generated_ids)

    except torch.cuda.OutOfMemoryError as e:
        print("\n" + "="*50)
        print(">>> TEST SUCCESSFUL: CUDA Out-of-Memory Error <<<")
        print("This confirms the 'Full Cache' baseline cannot handle the long context.")
        print("="*50 + "\n")
        generated_text = "OOM_ERROR"
        tokens_generated = 0
        end_time = time.perf_counter()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}\n")
        generated_text = "ERROR"
        tokens_generated = 0
        end_time = time.perf_counter()
    
    # 4. Calculate metrics
    total_time = end_time - start_time
    
    metrics['latency_per_token'] = total_time / tokens_generated if tokens_generated > 0 else 0
    metrics['tokens_per_second'] = tokens_generated / total_time if tokens_generated > 0 else 0
    metrics['tokens_generated'] = tokens_generated
    metrics['peak_vram_mb'] = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

    return generated_text, metrics