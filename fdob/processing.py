from scipy import signal

import numpy as np
import torch

from skimage.transform import resize

from typing import List, Callable


class AWGN:
    """Generate additive white Gaussian noise (AWGN) for SNR(dB)"""

    def __init__(self, snr: float) -> None:
        self.snr = snr

    def __call__(self, x: np.ndarray) -> np.ndarray:
        sig_avg_watts = np.mean(x**2.0)
        snr_real = 10 ** (self.snr / 10)
        n0 = sig_avg_watts / snr_real
        noise_avg_db = np.sqrt(n0)
        wn = noise_avg_db * np.random.standard_normal(*x.shape).astype("float32")
        return wn + x


class CWT:
    """Generate the 2D spectogram using continous wavelet transform (CWT)"""

    def __init__(
        self,
        widths: int,
        img_size: List[int],
        wavelet: Callable[..., np.ndarray] = signal.ricker,
    ) -> None:
        self.widths = widths
        self.wavelet = wavelet
        self.img_size = img_size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        aran = self.widths + 1
        X = np.arange(1, aran)
        img = signal.cwt(x, self.wavelet, widths=X)
        img = resize(img, self.img_size)
        img = img.astype(np.float32)
        return img


class CWTSignal:
    """Generate the multi-channel 1D spectogram using CWT"""

    def __init__(
        self, channels: int, wavelet: Callable[..., np.ndarray] = signal.ricker
    ) -> None:
        self.widths = channels
        self.wavelet = wavelet

    def __call__(self, x: np.ndarray) -> np.ndarray:
        aran = self.widths + 1
        X = np.arange(1, aran)
        img = signal.cwt(x, self.wavelet, widths=X)
        img = img.astype(np.float32)
        return img


class STFT:
    """Generate the 2D spectogram using STFT"""

    def __init__(self, window_length, noverlap, nfft):
        self.window = signal.get_window("hann", window_length)
        self.window_length = window_length
        self.noverlap = noverlap
        self.nfft = nfft

    def __call__(self, x):
        return np.abs(
            signal.stft(
                x,
                window=self.window,
                nperseg=self.window_length,
                noverlap=self.noverlap,
                nfft=self.nfft,
            )[2]
        )


class Resize:
    """Resize images"""

    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, x):
        return resize(x, (self.h, self.w))


class Crop:
    """Crop images"""

    def __init__(self, x0, x1, y0, y1):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

    def __call__(self, x):
        return x[self.x0 : self.x1, self.y0 : self.y1]


class NpToTensor:
    """Convert numpy array to PyTorch Tensor"""

    def __call__(self, x):
        return torch.from_numpy(x)


class ToSignal:
    """Convert (N, ) 1D Tensor to (1, N) 2D Tensor"""

    def __call__(self, x):
        return x.view(1, -1)


class ToImage:
    """Convert (HWC) style Tensor to (CHW) Tensor"""

    def __init__(self, h, w, c):
        self.h = h
        self.w = w
        self.c = c

    def __call__(self, x):
        return x.view(self.c, self.h, self.w)


class Standardize:
    """Standardize inputs"""

    def __call__(self, x):
        u = torch.mean(x)
        std = torch.std(x)
        return (x - u) / std


class Normalize:
    """Normalize inputs"""

    def __call__(self, x):
        x_min = torch.min(x)
        x_max = torch.max(x)
        return (x - x_min) / (x_max - x_min)


class FFT:
    """Generate frequency-domain input using fast Fourier transform (FFT)"""

    def __call__(self, x):
        return torch.abs(torch.fft.fft(x))


class InfinityNorm:

    def __call__(self, x):
        x_max = torch.max(torch.abs(x))
        return x / x_max