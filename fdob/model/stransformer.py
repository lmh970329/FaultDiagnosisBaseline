import torch
import torch.nn as nn

from .module import TransformerEncoder


class STransformer(nn.Module):

    def __init__(self, n_classes=10, sample_length=2048, segment_len=32, embed_dim=64, n_heads=4, depth=8, mlp_ratio=4.0, attn_bias=True, mlp_bias=True, attn_drop=0.0, proj_drop=0.0, mlp_drop=0.0, act_layer=None, norm_layer=None) -> None:
        super().__init__()

        self.n_classes = n_classes
        self.segment_len = segment_len
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.depth = depth
        self.sample_length = sample_length

        n_segments = sample_length // segment_len

        self.signal_embed = nn.Conv1d(1, embed_dim, segment_len, segment_len, padding='valid', bias=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_segments + 1, embed_dim))

        act_layer = act_layer or nn.GELU
        norm_layer = norm_layer or nn.LayerNorm

        blocks = [
            TransformerEncoder(
                embed_dim,
                n_heads,
                mlp_ratio,
                attn_bias,
                mlp_bias,
                attn_drop,
                proj_drop,
                mlp_drop,
                act_layer,
                norm_layer,
                pre_norm=True
            ) for _ in range(depth)
        ]
        self.blocks = nn.Sequential(*blocks)

        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, n_classes)


    def forward(self, x: torch.Tensor):
        B, L, C = x.shape

        embed = self.signal_embed(x).transpose(-2, -1) # B, N, D
        embed = torch.concat((self.cls_token.expand(B, -1, -1), embed), dim=-2)
        embed = embed + self.pos_embed

        out = self.blocks(embed)
        out = out[:, 0, :]
        out = self.norm(out)
        
        logit = self.head(out)
        return logit