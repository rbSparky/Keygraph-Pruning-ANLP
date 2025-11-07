import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple, Optional


def choose_runtime_dtype():
    """Choose the best dtype based on hardware capabilities."""
    if not torch.cuda.is_available():
        return torch.float32
    # Prefer bf16 on Ampere+; otherwise fp16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_model_and_tokenizer(model_dir: str, device: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the model and tokenizer with optional quantization."""
    print(f"Loading model and tokenizer from {model_dir}")
    
    # 1. Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=False,
        trust_remote_code=True,
        padding_side="left"  # Important for generation
    )
    
    # 2. Set the pad token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 3. Determine device and dtype
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = choose_runtime_dtype()
    
    print(f"Using device: {DEVICE}")
    print(f"Using dtype: {TORCH_DTYPE}")
    
    # 4. Setup quantization config if on CUDA
    quant_cfg = None
    use_quantization = False
    
    if DEVICE == "cuda":
        try:
            import bitsandbytes as bnb  # Check availability
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=TORCH_DTYPE,  # Match runtime dtype
            )
            use_quantization = True
            print("Using 4-bit quantization with BitsAndBytes.")
        except Exception as e:
            print(f"BitsAndBytes not available, using full precision: {e}")
            quant_cfg = None
    
    # 5. Load the model
    # CRITICAL: When using quantization, use minimal parameters
    if use_quantization:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=quant_cfg,
            device_map="auto",
            trust_remote_code=True,
            # Don't specify torch_dtype - handled by quantization_config
            # Don't specify low_cpu_mem_usage - can cause issues with quantization
        )
    else:
        # Only specify torch_dtype when NOT using quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=TORCH_DTYPE if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    
    # 6. Set to evaluation mode (don't move model - it's already on the right device)
    model.eval()
    
    print("Model and tokenizer loaded successfully.")
    print(f"Model device: {next(model.parameters()).device}")
    
    return model, tokenizer


def generate_with_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    do_sample: bool = False,
    past_key_values: Optional[Tuple] = None
) -> Tuple[str, torch.Tensor, Tuple]:
    """Generate text using the model with optional past key values."""
    
    # Get model's device dynamically
    model_device = next(model.parameters()).device
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096
    ).to(model_device)  # Use model's actual device
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
            output_attentions=False,
            output_hidden_states=False,
            past_key_values=past_key_values,
            use_cache=True
        )
    
    generated_text = tokenizer.decode(
        outputs.sequences[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    return generated_text, outputs.sequences, outputs.past_key_values