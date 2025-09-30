import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from typing import Tuple,Optional

def load_model_and_tokenizer(model_dir: str, device: torch.device) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the model and tokenizer correctly once."""
    print(f"Loading model and tokenizer from {model_dir}")

    # 1. Load the tokenizer a single time with all required settings
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=False,
        trust_remote_code=True,
        padding_side="left"  # Important for generation
    )

    # 2. Set the pad token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load the model a single time using the recommended `device_map`
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=False,
        trust_remote_code=True,
        torch_dtype=torch.float16,   # Use float16 for better performance on GPUs
        device_map='auto'            # Automatically handle device placement (GPU/CPU)
    )

    # 4. Set the model to evaluation mode
    model.eval()
    print("Model and tokenizer loaded successfully.")
    
    return model, tokenizer


def generate_with_model(
model:AutoModelForCausalLM,
tokenizer:AutoTokenizer,
prompt:str,
max_new_tokens:int =128,
temperature:float =0.7,
do_sample:bool =False,
past_key_values:Optional[Tuple]=None)->Tuple[str,torch.Tensor,Tuple]:
    """Generate text using the model with optional past key values."""


    inputs =tokenizer(
    prompt,
    return_tensors ="pt",
    padding =True,
    truncation =True,
    max_length =4096).to(model.device)


    with torch.no_grad():
        outputs =model.generate(
        **inputs,
        max_new_tokens =max_new_tokens,
        temperature =temperature,
        do_sample =do_sample,
        return_dict_in_generate =True,
        output_scores =True,
        output_attentions =False,
        output_hidden_states =False,
        past_key_values =past_key_values,
        use_cache =True)


    generated_text =tokenizer.decode(
    outputs.sequences[0][inputs.input_ids.shape[1]:],
    skip_special_tokens =True)

    return generated_text,outputs.sequences,outputs.past_key_values