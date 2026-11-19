import torch
import torch.nn as nn

class GatedCrossAttention(nn.Module):
    """
    Gated Cross-Attention module for video retrieval enhancement.
    Based on Otter's gated cross-attention mechanism.
    """
    def __init__(self, dim=768, dim_visual=768, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        # Cross-attention layer (using standard MultiheadAttention)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))
        
        # Feed-forward network with gating (using ModuleList like Otter)
        self.feed_forward = nn.ModuleList([
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult, bias=False),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim, bias=False),
        ])
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))
    
    def forward(self, x, media):
        """
        Args:
            x: [B, N, D] - original video features
            media: [B, M, D] - retrieved video features
        Returns:
            enhanced_features: [B, N, D] - enhanced video features
        """
        # Cross-attention: query from original video, key/value from retrieved videos
        attn_output, _ = self.attn(
            query=x,           # [B, N, D]
            key=media,         # [B, M, D]
            value=media        # [B, M, D]
        )
        
        # Apply attention gate (same as Otter: attn_output * gate.tanh() + x)
        x = attn_output * self.attn_gate.tanh() + x
        
        # Feed-forward with residual connection and gate (same as Otter)
        residual_x = x
        for ff in self.feed_forward:
            x = ff(x)
        x = x * self.ff_gate.tanh() + residual_x
        
        return x