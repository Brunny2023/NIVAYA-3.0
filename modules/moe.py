import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseMoE(nn.Module):
    """
    Sparse Mixture-of-Experts (MoE) Layer for Nivaya 3.0.
    Features: Learned routing, Top-K experts, and Shared experts.
    """
    def __init__(self, embed_dim, num_experts=8, top_k=2, shared_experts=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_experts = shared_experts
        
        # Router: Learns which expert to send tokens to
        self.router = nn.Linear(embed_dim, num_experts)
        
        # Individual Experts (Feed-Forward Networks)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim)
            ) for _ in range(num_experts)
        ])
        
        # Shared Experts (Always active)
        self.shared_expert_layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim)
            ) for _ in range(shared_experts)
        ])
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        flat_x = x.view(-1, embed_dim)
        
        # 1. Routing
        router_logits = self.router(flat_x)
        weights = F.softmax(router_logits, dim=-1)
        
        # Get top-k experts
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True) # Re-normalize
        
        # 2. Expert Execution
        # In a production setting, this would be optimized with grouped gemm or custom kernels
        final_output = torch.zeros_like(flat_x)
        
        # Process shared experts
        shared_out = torch.zeros_like(flat_x)
        for i in range(self.shared_experts):
            shared_out += self.shared_expert_layer[i](flat_x)
        
        # Process routed experts
        for i in range(self.num_experts):
            # Mask for tokens assigned to this expert
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                # Find weights for this expert among top-k
                expert_mask = (top_k_indices == i)
                # Sum weights for this expert (in case it appears multiple times, though top-k ensures unique)
                expert_weights = (top_k_weights * expert_mask.float()).sum(dim=-1, keepdim=True)
                
                expert_out = self.experts[i](flat_x[mask])
                final_output[mask] += expert_weights[mask] * expert_out
                
        # Combine shared and routed outputs
        return (final_output + shared_out).view(batch_size, seq_len, embed_dim)

class NivayaMoEBlock(nn.Module):
    """
    Nivaya Block that uses MoE instead of a standard FFN.
    """
    def __init__(self, embed_dim, num_heads, num_experts=8):
        super().__init__()
        from .attention import FlashAttention3
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = FlashAttention3(embed_dim, num_heads)
        
        self.ln2 = nn.LayerNorm(embed_dim)
        self.moe = SparseMoE(embed_dim, num_experts=num_experts)
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.moe(self.ln2(x))
        return x
