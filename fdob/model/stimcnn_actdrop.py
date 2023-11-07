import torch
from torch import nn
from .module import Conv2d
import torch.nn.functional as F


class STIMCNN(nn.Module):
    """
    Implementation of the model by (Zhao et al. 2021),
     A new bearing fault diagnosis method based on signal-to-image
     mapping and convolutional neural network (STIM-CNN).

    (Zhao et al. 2021) Jing Zhao, Shaopu Yang, Qiang Li, Yongqiang Liu,
     Xiaohui Gu, and Wenpeng Liu, “A new bearing fault diagnosis method
     based on signal-to-image mapping and convolutional neural network,”
     Measurement, vol. 176, p. 109088, 2021,
     doi: 10.1016/j.measurement.2021.109088.
    """

    def __init__(self, in_planes: int = 1, n_classes: int = 10, act_layer=True, no_drop=False):
        """
        Parameters
        ----------
        in_planes: int
            The number of channels of input data.
        n_classes: int
            The number of classes of dataset.
        """
        act_layer = nn.ReLU if act_layer else None
        super(STIMCNN, self).__init__()
        self._conv_layers = nn.Sequential(
            Conv2d(in_planes, 32, 5, 1, "same", norm_layer=nn.BatchNorm2d, act_layer=act_layer),
            nn.MaxPool2d(2, 2),
            Conv2d(32, 64, 5, 1, "same", norm_layer=nn.BatchNorm2d, act_layer=act_layer),
            nn.MaxPool2d(2, 2),
        )

        drop_layer = nn.Identity if no_drop else nn.Dropout

        with torch.no_grad():
            dummy = torch.rand(1, 1, 28, 28)
            dummy = self._conv_layers(dummy)
            dummy = torch.flatten(dummy, 1)
            lin_input = dummy.shape[1]

        self._linear_layers = nn.Sequential(
            nn.Linear(lin_input, 1024),
            nn.ReLU(),
            drop_layer(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            drop_layer(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_layers(x)
        x = torch.flatten(x, 1)
        x = self._linear_layers(x)

        return x
