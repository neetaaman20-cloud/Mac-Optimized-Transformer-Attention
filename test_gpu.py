import torch

# In 2026, Macs use 'mps' for acceleration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print(f"✅ Success! MacBook Air GPU (MPS) is active. Result: {x.item()}")
else:
    print("❌ Apple GPU (MPS) not detected.")

print(f"PyTorch Version: {torch.__version__}")