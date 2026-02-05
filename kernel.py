import torch
import helion  # The new PyTorch DSL for 2026

@helion.script
def fused_attention_kernel(Q, K, V):
    """
    A custom fused kernel that performs: 
    Softmax(Q @ K.T) @ V in a single GPU pass.
    """
    # Helion handles the tiling and memory movement automatically
    attn_weights = helion.matmul(Q, K.transpose(-1, -2))
    attn_weights = helion.softmax(attn_weights, dim=-1)
    return helion.matmul(attn_weights, V)

def get_fused_attention():
    # Compiles the Python-style code into an optimized CUDA/ROCm kernel
    return helion.compile(fused_attention_kernel)