import torch
import torch.nn as nn
from functools import partial


class MultiLayerPerceptron(nn.Module):
    def __init__(self, embed_dim, exp_ratio=4.0, drop_rate=0.0, act_layer=None, bias=True) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        act_layer = act_layer or partial(nn.GELU, approximate='tanh')

        self.act_layer = act_layer()
        
        hidden_dim = int(embed_dim * exp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(drop_rate) if drop_rate else nn.Identity()


    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x