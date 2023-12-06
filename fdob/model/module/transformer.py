import torch
import torch.nn as nn
from .attention import MultiHeadSelfAttention
from .mlp import MultiLayerPerceptron


class TransformerEncoder(nn.Module):

    def __init__(self, embed_dim, n_heads, mlp_ratio=4.0, attn_bias=True, mlp_bias=True, attn_drop=0.0, proj_drop=0.0, mlp_drop=0.0, act_layer=None, norm_layer=None, pre_norm=True) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads

        norm_layer = norm_layer or nn.LayerNorm

        self.mhsa = MultiHeadSelfAttention(embed_dim, n_heads, attn_bias=attn_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.mlp = MultiLayerPerceptron(embed_dim, mlp_ratio, mlp_drop, act_layer, mlp_bias)

        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)

        self.pre_norm = pre_norm


    def forward(self, x: torch.Tensor):
        if self.pre_norm:
            x = x + self.mhsa(self.norm1(x))
            x = x + self.mlp(self.norm2(x))

        else:
            x = self.norm1(x + self.mhsa(x))
            x = self.norm2(x + self.mlp(x))
        
        return x