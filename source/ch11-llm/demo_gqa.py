import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups=8):
        super().__init__()
        assert num_heads % num_groups == 0, "Num heads must be divisible by num groups"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model // num_groups)
        self.v_proj = nn.Linear(d_model, d_model // num_groups)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.shape

        q = self.q_proj(q).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(k).view(batch_size, seq_len, self.num_groups, self.head_dim)
        v = self.v_proj(v).view(batch_size, seq_len, self.num_groups, self.head_dim)

        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_groups, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_groups, seq_len, head_dim]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_probs = torch.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_probs, v)  # [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)

        return attn_output


if __name__ == '__main__':
    from hiq.vis import print_model
    gqa = GroupedQueryAttention(d_model=512, num_heads=8, num_groups=8)
    print_model(gqa)
