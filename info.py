import torch

from . import fdob
from .fdob import model
from .fdob import processing

hparam = {
    "sgd": {
        "optimizer": torch.optim.SGD,
        "n_params": 1,
        "param_names": ["lr"],
        "lb": [-4],
        "ub": [0],
        "reversed": [False],
    },
    "momentum": {
        "optimizer": torch.optim.SGD,
        "n_params": 2,
        "param_names": ["lr", "momentum"],
        "lb": [-4, -3],
        "ub": [0, 0],
        "reversed": [False, True],
    },
    "rmsprop": {
        "optimizer": torch.optim.RMSprop,
        "n_params": 4,
        "param_names": ["lr", "momentum", "alpha", "eps"],
        "lb": [-4, -3, -3, -10],
        "ub": [-1, 0, 0, 0],
        "reversed": [False, True, True, False],
    },
    "adam": {
        "optimizer": torch.optim.Adam,
        "n_params": 4,
        "param_names": ["lr", "beta1", "beta2", "eps"],
        "lb": [-4, -3, -4, -10],
        "ub": [-1, 0, -1, 0],
        "reversed": [False, True, True, False],
    },
}

model = {
    "stimcnn": {
        "model": model.STIMCNN,
        "sample_length": 784,
        "tf": [processing.NpToTensor(), processing.ToImage(28, 28, 1)],
    },
    "stftcnn": {
        "model": model.STFTCNN,
        "sample_length": 512,
        "tf": [
            processing.STFT(window_length=128, noverlap=120, nfft=128),
            processing.Resize(64, 64),
            processing.NpToTensor(),
            processing.ToImage(64, 64, 1),
        ],
    },
    "wdcnn": {
        "model": model.WDCNN,
        "sample_length": 2048,
        "tf": [processing.NpToTensor(), processing.ToSignal()],
    },
    "wdcnnrnn": {
        "model": model.WDCNNRNN,
        "sample_length": 4096,
        "tf": [processing.NpToTensor(), processing.ToSignal()],
    },
    "ticnn": {
        "model": model.TICNN,
        "sample_length": 2048,
        "tf": [processing.NpToTensor(), processing.ToSignal()],
    },
    "dcn": {
        "model": model.DCN,
        "sample_length": 784,
        "tf": [processing.NpToTensor(), processing.ToSignal()],
    },
    "srdcnn": {
        "model": model.SRDCNN,
        "sample_length": 1024,
        "tf": [processing.NpToTensor(), processing.ToSignal()],
    },
    "stransformer": {
        "model": model.STransformer,
        "sample_length": 2048,
        "tf": [processing.NpToTensor(), processing.ToSignal()],
    }
}
