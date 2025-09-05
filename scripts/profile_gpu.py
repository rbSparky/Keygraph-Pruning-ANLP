import os
import sys
import torch


sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))


def main():
    print("GPU Profiling Information")
    print("="*30)


    if torch.cuda.is_available():
        print(f"CUDA is available")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory /1024 **3:.2f} GB")


        try:
            import flash_attn
            print(f"\nFlash Attention: Available")
        except ImportError:
            print(f"\nFlash Attention: Not available")


        if hasattr(torch.nn.functional,'scaled_dot_product_attention'):
            print(f"Scaled Dot Product Attention: Available")
        else:
            print(f"Scaled Dot Product Attention: Not available")
    else:
        print("CUDA is not available. Running on CPU.")

    print(f"\nPyTorch version: {torch.__version__}")


if __name__ =="__main__":
    main()