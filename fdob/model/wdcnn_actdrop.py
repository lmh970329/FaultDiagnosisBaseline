from functools import partial
import torch
from torch import nn
from .module import Conv1dChannelDrop, Conv1d, Conv1dActivationDrop
from .module.FTWT import Conv1dFTWT, PredictorLoss
import torch.nn.functional as F

class WDCNN(nn.Module):
    """
    Implementation of the model by (Zhang et al. 2017), Deep Convolutional
     Neural Networks with Wide First-layer Kernels (WDCNN).

    (Zhang et al. 2017) Wei Zhang, Gaoliang Peng, Chuanhao Li, Yuanhang Chen,
     and Zhujun Zhang, “A New Deep Learning Model for Fault Diagnosis with
     Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals,”
     Sensors, vol. 17, no. 2, p. 425, 2017, doi: 10.3390/s17020425.
    """

    def __init__(self, first_kernel: int = 64, n_classes: int = 10, information_ratio: float = None, drop_type: str = 'channel', drop_rate=0., score_type='max', act_layer=True) -> None:
        """
        Parameters
        ----------
        first_kernel: int
            The kernel size of the first conv layer.
        n_classes: int
            The number of classes of dataset.
        information_ratio: float
            The ratio of non-zero units for pruning of each convolution layers
        drop_type: str
            This specifies the level of units to be pruned 
        drop_rate: float
            A ratio for dropout layer
        score_type: str
            The type for scoring the convolution layer's activations
        """
        super(WDCNN, self).__init__()
        
        conv_layer = Conv1d
        if information_ratio is not None:
            if drop_type == 'ftwt':
                self.pred_loss = PredictorLoss()
                conv_layer = partial(Conv1dFTWT, pred_loss=self.pred_loss, information_ratio=information_ratio)
            elif drop_type == 'activation':
                conv_layer = partial(Conv1dActivationDrop, information_ratio=information_ratio)
            elif drop_type == 'channel':
                conv_layer = partial(Conv1dChannelDrop, information_ratio=information_ratio, score_type=score_type)
            else:
                raise ValueError(f"Not supported activation drop type : '{drop_type}'")

        act_layer = nn.ReLU if act_layer else None

        self.conv_layers = nn.Sequential(
            # Conv1
            conv_layer(1, 16, first_kernel, stride=16, padding=24, norm_layer=nn.BatchNorm1d, act_layer=act_layer, drop_rate=drop_rate),
            # Pool1
            torch.nn.MaxPool1d(2, 2),
            # Conv2
            conv_layer(16, 32, 3, stride=1, padding="same", norm_layer=nn.BatchNorm1d, act_layer=act_layer, drop_rate=drop_rate),
            # Pool2
            torch.nn.MaxPool1d(2, 2),
            # Conv3
            conv_layer(32, 64, 3, stride=1, padding="same", norm_layer=nn.BatchNorm1d, act_layer=act_layer, drop_rate=drop_rate),
            # Pool3
            torch.nn.MaxPool1d(2, 2),
            # Conv4
            conv_layer(64, 64, 3, stride=1, padding="same", norm_layer=nn.BatchNorm1d, act_layer=act_layer, drop_rate=drop_rate),
            # Pool4
            torch.nn.MaxPool1d(2, 2),
            # Conv5
            conv_layer(64, 64, 3, stride=1, padding=0, norm_layer=nn.BatchNorm1d, act_layer=act_layer, drop_rate=drop_rate),
            # Pool5
            torch.nn.MaxPool1d(2, 2),
        )

        with torch.no_grad():
            dummy = torch.rand(1, 1, 2048)
            dummy = self.conv_layers(dummy)
            dummy = torch.flatten(dummy, 1)
            lin_input = dummy.shape[1]

        self.linear_layers = nn.Sequential(
            torch.nn.Linear(lin_input, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)

        return x
    

