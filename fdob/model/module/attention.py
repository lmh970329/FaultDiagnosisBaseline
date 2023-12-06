import torch
import torch.nn as nn



class ScaledDotProductAttention(nn.Module):
    def __init__(
            self, scale
    ) -> None:
        super().__init__()
        self.scale = scale


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # input shape : B, H, N, D
        attn = q @ k.transpose(-2, -1) / self.scale
        attn = torch.softmax(attn, dim=-1) # B, H, N, N

        x = attn @ v # B, H, N, D
        
        return x



class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, embed_dim, n_heads, attn_bias=False, attn_drop=0.0, proj_drop=0.0
    ) -> None:
        super().__init__()

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=attn_bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=attn_bias)

        self.head_dim = embed_dim // n_heads

        scale = self.head_dim**0.5
        self.attn = ScaledDotProductAttention(scale)

        self.attn_drop = nn.Dropout(attn_drop) if attn_drop else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop else nn.Identity()

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.reshape((B, N, 3, -1, self.head_dim)).permute((2, 0, 3, 1, 4))

        out = self.attn(q, k, v) # B, H, N, D

        out = out.transpose(-2, -3).reshape(B, N, C)

        out = self.proj(out)

        return out