import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from typing import Tuple,Optional


def load_model_and_tokenizer(model_dir:str,device:torch.device)->Tuple[AutoModelForCausalLM,AutoTokenizer]:
    """Load the TinyLlama model and tokenizer."""
    print(f"Loading model from {model_dir}")


    tokenizer =AutoTokenizer.from_pretrained(
    model_dir,
    local_files_only =True,
    padding_side ="left")


    if tokenizer.pad_token is None:
        tokenizer.pad_token =tokenizer.eos_token


    model =AutoModelForCausalLM.from_pretrained(
    model_dir,
    local_files_only =True,
    torch_dtype =torch.float16 if device.type =="cuda"else torch.float32).to(device)

    model.eval()
    print(f"Model loaded successfully")
    return model,tokenizer


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