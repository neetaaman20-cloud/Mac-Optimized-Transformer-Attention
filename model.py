import torch
import torch.nn as nn
import torch.nn.functional as F

class MacOptimizedAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, q, k, v):
        # Uses the native C++ fused kernel (SDPA) for Mac GPU speed
        return F.scaled_dot_product_attention(q, k, v)

class TinyTransformer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = MacOptimizedAttention(embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Residual connection + LayerNorm
        return self.ln(x + self.attn(x, x, x))