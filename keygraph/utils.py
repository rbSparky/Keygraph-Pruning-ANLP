import torch
import random
import numpy as np


def set_seed(seed:int =42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark =False


def get_device(use_gpu:bool =True):
    """Get the device to use for computations."""
    if use_gpu and torch.cuda.is_available():
        device =torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device =torch.device("cpu")
        print("Using CPU")
    return device


def format_metrics(metrics_dict):
    """Format metrics for display."""
    formatted ={}
    for key,value in metrics_dict.items():
        if isinstance(value,float):
            formatted[key]=f"{value:.4f}"
        else:
            formatted[key]=str(value)
    return formatted