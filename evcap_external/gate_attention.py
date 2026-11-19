import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, headwise_attn_output_gate=False, elementwise_attn_output_gate=False, qkv_bias=False, num_key_value_heads=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # 如果没有指定 num_key_value_heads，则使用与 num_heads 相同（标准 MHA）
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # 根据门控类型初始化QKV投影
        if headwise_attn_output_gate:
            # 头级别门控：Q投影包含 query 和 gate，K和V正常
            # Q: num_heads * head_dim + num_heads (每个头一个gate值)
            # K: num_key_value_heads * head_dim
            # V: num_key_value_heads * head_dim
            q_dim = self.num_heads * self.head_dim + self.num_heads
            k_dim = self.num_key_value_heads * self.head_dim
            v_dim = self.num_key_value_heads * self.head_dim
            total_dim = q_dim + k_dim + v_dim
            self.qkv_proj = nn.Linear(embed_dim, total_dim, bias=qkv_bias)
        elif elementwise_attn_output_gate:
            # 元素级别门控：Q投影包含 query 和 gate（每个元素一个gate值）
            # Q: num_heads * head_dim * 2 (query + gate)
            # K: num_key_value_heads * head_dim
            # V: num_key_value_heads * head_dim
            q_dim = self.num_heads * self.head_dim * 2
            k_dim = self.num_key_value_heads * self.head_dim
            v_dim = self.num_key_value_heads * self.head_dim
            total_dim = q_dim + k_dim + v_dim
            self.qkv_proj = nn.Linear(embed_dim, total_dim, bias=qkv_bias)
        else:
            # 标准QKV投影
            self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 门控类型标志
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.elementwise_attn_output_gate = elementwise_attn_output_gate
        
    def forward(self, x):
        B, L, D = x.shape
        
        # 1. QKV projection
        qkv_output = self.qkv_proj(x)
        
        if self.headwise_attn_output_gate:
            # 头级别门控：分离 query、gate、k、v
            # qkv_output shape: [B, L, q_dim + k_dim + v_dim]
            # q_dim = num_heads * head_dim + num_heads
            # k_dim = num_key_value_heads * head_dim
            # v_dim = num_key_value_heads * head_dim
            
            q_dim = self.num_heads * self.head_dim + self.num_heads
            k_dim = self.num_key_value_heads * self.head_dim
            v_dim = self.num_key_value_heads * self.head_dim
            
            q_with_gate, k, v = torch.split(qkv_output, [q_dim, k_dim, v_dim], dim=-1)
            
            # 重塑 q_with_gate 为 [B, L, num_heads, head_dim + 1]
            q_with_gate = q_with_gate.view(B, L, self.num_heads, -1)
            
            # 分离 query 和 gate
            q = q_with_gate[..., :self.head_dim]  # [B, L, num_heads, head_dim]
            gate_score = q_with_gate[..., self.head_dim:]  # [B, L, num_heads, 1]
            
            # 重塑 k, v
            k = k.view(B, L, self.num_key_value_heads, self.head_dim)
            v = v.view(B, L, self.num_key_value_heads, self.head_dim)
            
            # 重塑为多头注意力格式
            q = q.permute(0, 2, 1, 3)  # [B, num_heads, L, head_dim]
            k = k.permute(0, 2, 1, 3)  # [B, num_key_value_heads, L, head_dim]
            v = v.permute(0, 2, 1, 3)  # [B, num_key_value_heads, L, head_dim]
            gate_score = gate_score.permute(0, 2, 1, 3)  # [B, num_heads, L, 1]
            
        elif self.elementwise_attn_output_gate:
            # 元素级别门控：分离 query、gate、k、v
            # qkv_output shape: [B, L, q_dim + k_dim + v_dim]
            # q_dim = num_heads * head_dim * 2 (query + gate)
            # k_dim = num_key_value_heads * head_dim
            # v_dim = num_key_value_heads * head_dim
            
            q_dim = self.num_heads * self.head_dim * 2
            k_dim = self.num_key_value_heads * self.head_dim
            v_dim = self.num_key_value_heads * self.head_dim
            
            q_with_gate, k, v = torch.split(qkv_output, [q_dim, k_dim, v_dim], dim=-1)
            
            # 重塑 q_with_gate 为 [B, L, num_heads, head_dim * 2]
            q_with_gate = q_with_gate.view(B, L, self.num_heads, -1)
            
            # 分离 query 和 gate
            q = q_with_gate[..., :self.head_dim]  # [B, L, num_heads, head_dim]
            gate_score = q_with_gate[..., self.head_dim:]  # [B, L, num_heads, head_dim]
            
            # 重塑 k, v
            k = k.view(B, L, self.num_key_value_heads, self.head_dim)
            v = v.view(B, L, self.num_key_value_heads, self.head_dim)
            
            # 重塑为多头注意力格式
            q = q.permute(0, 2, 1, 3)  # [B, num_heads, L, head_dim]
            k = k.permute(0, 2, 1, 3)  # [B, num_key_value_heads, L, head_dim]
            v = v.permute(0, 2, 1, 3)  # [B, num_key_value_heads, L, head_dim]
            gate_score = gate_score.permute(0, 2, 1, 3)  # [B, num_heads, L, head_dim]
            
        else:
            # 标准模式：不使用门控
            qkv = qkv_output.reshape(B, L, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            gate_score = None
        
        # 2. Scaled dot-product attention
        # 如果使用 GQA，需要扩展 k 和 v 以匹配 q 的头数
        if self.num_heads != self.num_key_value_heads:
            # 扩展 k 和 v: [B, num_key_value_heads, L, head_dim] -> [B, num_heads, L, head_dim]
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        
        attn_out = F.scaled_dot_product_attention(q, k, v)
        # Shape: [B, num_heads, L, head_dim]
        
        # 3. Apply gating if enabled
        if gate_score is not None:
            # 应用 sigmoid 激活函数
            gate_scores = torch.sigmoid(gate_score)
            # 应用门控
            gated_out = attn_out * gate_scores
        else:
            gated_out = attn_out
        
        # 4. Concatenate heads and project
        gated_out = gated_out.permute(0, 2, 1, 3).reshape(B, L, D)
        output = self.out_proj(gated_out)
        
        return output