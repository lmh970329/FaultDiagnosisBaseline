import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_1_t
from .. import Conv1d
from typing import Any, Union


class PredictorLoss():

    def __init__(self) -> None:
        self.values = []


    def add(self, loss):
        self.values.append(loss)


    def get(self):
        return torch.stack(self.values).sum()
    

    def reset(self):
        self.values.clear()



class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, prob: Tensor) -> Any:
        return torch.round(prob)


    @staticmethod
    def backward(ctx: Any, grad_outputs) -> Any:
        return grad_outputs


class Conv1dFTWT(nn.Module):
    
    def __init__(
            self,
            in_channels:int,
            out_channels: int,
            kernel_size: _size_1_t,
            information_ratio,
            pred_loss: PredictorLoss,
            stride: _size_1_t = 1,
            padding: Union[_size_1_t, str] = 0,
            dilation: _size_1_t = 1,
            groups: int = 1, 
            bias: bool = False,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None,
            norm_layer=None,
            act_layer=None,
            drop_rate=0.
        ) -> None:
        super().__init__()
        self.conv_layer = Conv1d(
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
            drop_rate
        )

        self.pred_head = nn.Conv1d(in_channels, out_channels, 1, bias=False, device=device, dtype=dtype)

        self.pred_loss = pred_loss

        self.information_ratio = information_ratio


    @classmethod
    def from_exist_conv(cls, conv_layer: Conv1d, information_ratio: float):
        ftwt_module = cls(
            conv_layer.conv.in_channels,
            conv_layer.conv.out_channels,
            conv_layer.conv.kernel_size,
            information_ratio
        )
        ftwt_module.conv_layer = conv_layer
        return ftwt_module



    @torch.no_grad()
    def get_binary_mask_gt(self, out):
        gmp = torch.max(out.abs(), dim=-1).values

        mask = torch.ones_like(gmp)

        score_val, indices = torch.div(
            gmp,
            torch.sum(gmp, dim=-1, keepdim=True)
        ).sort(dim=-1, descending=True)

        cumulative = torch.cumsum(score_val, dim=-1)
        
        for sample_idx in range(out.shape[0]):
            prune_idx = indices[sample_idx][(cumulative > self.information_ratio)[sample_idx]]
            mask[sample_idx][prune_idx] = 0

        return mask.unsqueeze(-1)


    @torch.no_grad()
    def global_max_pool(self, input):
        return torch.max(input.abs(), dim=-1, keepdim=True).values


    def get_channel_mask(self, input: Tensor):
        logit = self.pred_head(self.global_max_pool(input).softmax(dim=-2))
        prob = torch.sigmoid(logit)
        return Round.apply(prob), logit


    def forward(self, input: Tensor) -> Tensor:
        mask, logit = self.get_channel_mask(input)
        out = self.conv_layer(input)

        if self.training:
            gt_mask = self.get_binary_mask_gt(out)
            self.pred_loss.add(F.binary_cross_entropy_with_logits(logit, gt_mask, reduction='none').sum(dim=1).mean())
            # self.pred_loss.add(F.binary_cross_entropy_with_logits(logit, gt_mask, reduction='mean'))

        return out * mask