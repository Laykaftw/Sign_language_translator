import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    print("GPU is available!")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is NOT available. Using CPU.")