import torch
import torch.nn as nn


class BaselineAttention(nn.Module):
    def __init__(
        self, embed_dim, n_heads, attn_bias=False, attn_drop=0.0, proj_drop=0.0
    ) -> None:
        super().__init__()

        self.qkv = nn.Conv1d(embed_dim, 3 * embed_dim, 1, 1, bias=attn_bias)
        self.proj = nn.Conv1d(embed_dim, embed_dim, 1, 1, bias=attn_bias)

        self.head_dim = embed_dim // n_heads

        self.scale = self.head_dim**0.5

        self.attn_drop = nn.Dropout(attn_drop) if attn_drop else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop else nn.Identity()

    def forward(self, x: torch.Tensor):
        B, C, N = x.shape
        qkv = self.qkv(x)

        q, k, v = qkv.reshape((B, 3, -1, self.head_dim, N)).transpose(0, 1)

        attn = q.transpose(-2, -1) @ k / self.scale
        attn = torch.softmax(attn, dim=-1)

        x = attn @ v.transpose(-2, -1)  # B, n_heads, N, head_dim

        x = x.transpose(-2, -1).reshape(B, -1, N)

        x = self.proj(x)

        return x


class BaselineMLP(nn.Module):
    def __init__(self, embed_dim, exp_ratio, bias=False, mlp_drop=0.0) -> None:
        super().__init__()

        self.hidden = int(embed_dim * exp_ratio)

        self.fc1 = nn.Conv1d(embed_dim, self.hidden, 1, 1, bias=bias)
        self.fc2 = nn.Conv1d(self.hidden, embed_dim, 1, 1, bias=bias)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(mlp_drop) if mlp_drop else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class BaselineEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_heads=2,
        attn_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        mlp_ratio=2.0,
        mlp_bias=False,
        mlp_drop=0.0,
        res_drop=0.0,
    ) -> None:
        super().__init__()

        self.attn = BaselineAttention(
            embed_dim,
            n_heads,
            attn_bias=attn_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.mlp = BaselineMLP(embed_dim, mlp_ratio, bias=mlp_bias, mlp_drop=mlp_drop)
        self.norm = nn.BatchNorm1d(embed_dim)

        self.drop = nn.Dropout(res_drop) if res_drop else nn.Identity()

    def forward(self, x):
        x = self.norm(x + self.drop(self.attn(x)))
        x = self.norm(x + self.drop(self.mlp(x)))

        return x


class BaselineTransformer(nn.Module):
    def __init__(
        self,
        embed_dim=32,
        depth=4,
        n_classes=10,
        n_heads=1,
        attn_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        mlp_bias=False,
        mlp_ratio=4.0,
        mlp_drop=0.0,
        res_drop=0.0,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads

        self.embed = nn.Sequential(
            nn.Conv1d(1, embed_dim // 4, 64, 4, bias=False),
            nn.BatchNorm1d(embed_dim // 4),
            nn.ReLU(),
            # nn.Conv1d(embed_dim // 8, embed_dim // 4, 64, 2, bias=False),
            # nn.BatchNorm1d(embed_dim // 4),
            # nn.ReLU(),
            nn.Conv1d(embed_dim // 4, embed_dim // 2, 64, 4, bias=False),
            nn.BatchNorm1d(embed_dim // 2),
            nn.ReLU(),
            nn.Conv1d(embed_dim // 2, embed_dim, 64, 4, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim, 1))

        self.blocks = nn.Sequential(
            *[
                BaselineEncoder(
                    embed_dim,
                    n_heads,
                    attn_bias=attn_bias,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    mlp_bias=mlp_bias,
                    mlp_ratio=mlp_ratio,
                    mlp_drop=mlp_drop,
                    res_drop=res_drop,
                )
                for _ in range(depth)
            ]
        )

        self.head = nn.Linear(embed_dim, n_classes, bias=True)

    def forward(self, x):
        B, *_ = x.shape

        x = self.embed(x)
        x = torch.concat([torch.tile(self.cls_token, dims=(B, 1, 1)), x], dim=-1)

        x = self.blocks(x)

        x = self.head(x[:, :, 0])

        return x
