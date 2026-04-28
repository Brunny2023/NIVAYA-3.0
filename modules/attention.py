import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FlashAttention3(nn.Module):
    """
    Simplified implementation of the FlashAttention-3 concept for PyTorch.
    In a real-world scenario, this would interface with CUDA kernels.
    Here we implement a memory-efficient scaled dot-product attention.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Using PyTorch's built-in scaled_dot_product_attention which uses FlashAttention if available
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask, 
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=mask is None
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)

class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet Block: A linear attention variant for high efficiency.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.delta_weight = nn.Parameter(torch.randn(embed_dim, embed_dim) / math.sqrt(embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        g = self.gate(x)
        # Linearized attention approximation
        delta = torch.matmul(x, self.delta_weight)
        out = x * g + delta * (1 - g)
        return self.out_proj(out)

class NivayaBlock(nn.Module):
    """
    Hybrid Block: Mixes Attention and Gated DeltaNet as per research.
    """
    def __init__(self, embed_dim, num_heads, is_attention=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        if is_attention:
            self.fn = FlashAttention3(embed_dim, num_heads)
        else:
            self.fn = GatedDeltaNet(embed_dim)
            
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        
    def forward(self, x, mask=None):
        # Hybrid Attention/DeltaNet path
        if isinstance(self.fn, FlashAttention3):
            x = x + self.fn(self.ln1(x), mask=mask)
        else:
            x = x + self.fn(self.ln1(x))
            
        # FFN path
        x = x + self.ffn(self.ln2(x))
        return x
