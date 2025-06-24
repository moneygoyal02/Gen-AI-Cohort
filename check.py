import torch

print("PyTorch CUDA Version:", torch.version.cuda)
print("Is CUDA available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
