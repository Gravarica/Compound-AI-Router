# test_mps.py
import torch

# Check MPS availability properly
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device available")
else:
    device = torch.device("cpu")
    print("MPS device not available, using CPU")

# Create tensors on the device
a = torch.randn(3, 3, device=device)
b = torch.randn(3, 3, device=device)

# Test basic operations
c = a + b
print(c)