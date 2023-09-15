from typing import Union
from torch import Tensor
import torch.nn as nn
import torch
from torch.nn.common_types import _size_2_t
from typing import Union


class Conv2d(nn.Sequential):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[_size_2_t, str] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None,
            norm_layer=None,
            act_layer=None,
            drop_rate=0.0
        ) -> None:
        super().__init__()

        self.add_module(
            'conv',
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
                device=device,
                dtype=dtype
            )
        )

        if norm_layer:
            self.add_module(
                'norm',
                norm_layer(out_channels)
            )

        if act_layer:
            self.add_module(
                'act',
                act_layer()
            )

        dropout = nn.Dropout2d(drop_rate) if drop_rate else nn.Identity()

        self.add_module(
            'drop',
            dropout
        )


class Conv2dFeaturemapDrop(Conv2d):


    def __init__(
            self,
            in_channels:int,
            out_channels: int,
            kernel_size: _size_2_t,
            information_ratio,
            stride: _size_2_t = 1,
            padding: Union[_size_2_t, str] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1, 
            bias: bool = False,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None,
            norm_layer=None,
            act_layer=None,
            score_type='max'
        ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
            norm_layer,
            act_layer,
            drop_rate=0.0
        )

        self.information_ratio = information_ratio
        self.sparsity = 1. - information_ratio

        if not score_type in ['max', 'mean', 'l1', 'l2']:
            raise ValueError("'score_type' must be one of ['max', 'mean', 'l1', 'l2']\n"
                             f"{score_type} is not supported.")
        self.score_type = score_type


    @torch.no_grad()
    def get_binary_mask(self, out: torch.Tensor):

        if self.score_type == 'max':
            score = torch.max(out.abs().flatten(start_dim=2), dim=-1).values
        elif self.score_type == 'l1':
            score = torch.norm(out, p=1, dim=-1)
        else:
            score = torch.norm(out, p=2, dim=-1)

        mask = torch.ones_like(score)

        n_prune = int(score.shape[-1] * self.sparsity)

        indices = torch.topk(score, n_prune, largest=False).indices

        strides = torch.arange(0, out.shape[0], device=mask.device).reshape(-1, 1) * score.shape[-1]

        mask.view(-1)[indices + strides] = 0
        
        return mask.unsqueeze(-1).unsqueeze(-1)


    def forward(self, input: Tensor) -> Tensor:
        out = super().forward(input)
        mask = self.get_binary_mask(out)
        return out * mask