import torch
import time
from model import TinyTransformer

def benchmark():
    # Use 'mps' for MacBook Air GPU
    device = torch.device("mps")
    print(f"🚀 Starting benchmark on: {device}")
    
    model = TinyTransformer(512).to(device)
    x = torch.randn(32, 128, 512).to(device)

    # Warmup - let the GPU "wake up"
    for _ in range(10): 
        _ = model(x)
    
    # Precise Timing
    start = time.time()
    for _ in range(100):
        _ = model(x)
    
    # Synchronize is important for accurate GPU timing
    torch.mps.synchronize() 
    end = time.time()
    
    print(f"✅ Finished 100 iterations in: {(end - start):.4f} seconds")

if __name__ == "__main__":
    benchmark()