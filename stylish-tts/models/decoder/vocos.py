from typing import Optional

import math
import random
import numpy as np
import time

import torch
from torch import nn
import torch.nn.functional as F

# from torch.nn.utils import weight_norm
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.nn import Conv1d
from ..common import init_weights, get_padding

from typing import Optional, Tuple
from scipy.signal import get_window

from utils import DecoderPrediction
from .harmonics import HarmonicGenerator


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdaINResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), style_dim=64):
        super(AdaINResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

        self.adain1 = nn.ModuleList(
            [
                AdaIN1d(style_dim, channels),
                AdaIN1d(style_dim, channels),
                AdaIN1d(style_dim, channels),
            ]
        )

        self.adain2 = nn.ModuleList(
            [
                AdaIN1d(style_dim, channels),
                AdaIN1d(style_dim, channels),
                AdaIN1d(style_dim, channels),
            ]
        )

        self.alpha1 = nn.ParameterList(
            [nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs1))]
        )
        self.alpha2 = nn.ParameterList(
            [nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs2))]
        )

    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2
        ):
            xt = n1(x, s)
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake1D
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim_in (int): Number of input channels.
        dim_out (int): Number of output channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        style_dim: int,
    ):
        super().__init__()
        self.dim_out = dim_out
        self.dwconv = nn.Conv1d(
            dim_in, dim_in, kernel_size=7, padding=3, groups=dim_in
        )  # depthwise conv
        # self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.norm = AdaINResBlock1(dim_in, 7, [1, 3, 5], style_dim)
        self.pwconv1 = nn.Linear(
            dim_in, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim_out)
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones(dim_out), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        residual = x[:, : self.dim_out, :]
        x = self.dwconv(x)
        x = self.norm(x, s)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class Generator(torch.nn.Module):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
    """

    def __init__(
        self,
        *,
        input_channels: int,
        dim: int,
        style_dim: int,
        intermediate_dim: int,
        num_layers: int,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        layer_scale_init_value: Optional[float] = None,
    ):
        super(Generator, self).__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers

        self.harmonic = HarmonicGenerator(
            sample_rate=sample_rate,
            dim_out=dim,
            win_length=win_length,
            hop_length=hop_length,
        )
        self.convnext = nn.ModuleList()

        for _ in range(num_layers):
            # Add ConvNeXt block
            self.convnext.append(
                ConvNeXtBlock(
                    dim_in=dim * 2,
                    dim_out=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    style_dim=style_dim,
                )
            )

        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)
        self.stft = ISTFTHead(
            dim=dim,
            n_fft=n_fft,
            hop_length=hop_length,
            padding="same",
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, s, f0, N) -> torch.Tensor:
        har = self.harmonic(f0, N)
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = torch.cat([x, har], dim=1)
            x = conv_block(x, s)
        x = self.final_layer_norm(x.transpose(1, 2))
        x = self.stft(x)
        return x


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.win_length = n_fft
        self.hop_length = hop_length
        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window, persistent=False)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


class ISTFTHead(torch.nn.Module):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        win_length (int): Length of the sliding window frames
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(
        self,
        dim: int,
        n_fft: int,
        hop_length: int,
        padding: str = "same",
    ):
        super(ISTFTHead, self).__init__()
        # self.filter_length = n_fft
        # self.win_length = n_fft
        # self.hop_length = hop_length
        # self.window = torch.from_numpy(
        #    get_window("hann", self.win_length, fftbins=True).astype(np.float32)
        # )

        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(
            mag, max=1e2
        )  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # S = mag * torch.exp(phase * 1j)
        # better directly produce the complex value
        S = mag * (x + 1j * y)
        audio = self.istft(S)
        return audio

    # def transform(self, input_data):
    #    forward_transform = torch.stft(
    #        input_data,
    #        self.filter_length,
    #        self.hop_length,
    #        self.win_length,
    #        window=self.window.to(input_data.device),
    #        return_complex=True,
    #    )

    #    return torch.abs(forward_transform), torch.angle(forward_transform)


class AdainResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        style_dim=64,
        actv=nn.LeakyReLU(0.2),
        upsample="none",
        dropout_p=0.0,
    ):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)

        if upsample == "none":
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(
                nn.ConvTranspose1d(
                    dim_in,
                    dim_in,
                    kernel_size=3,
                    stride=2,
                    groups=dim_in,
                    padding=1,
                    output_padding=1,
                )
            )

    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode="nearest")


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        dim_in,
        style_dim,
        intermediate_dim,
        num_layers,
        sample_rate,
        gen_istft_n_fft,
        gen_istft_win_length,
        gen_istft_hop_length,
    ):
        super().__init__()

        self.decode = nn.ModuleList()

        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)

        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True))

        self.F0_conv = weight_norm(
            nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
        )

        self.N_conv = weight_norm(
            nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
        )

        self.asr_res = nn.Sequential(
            weight_norm(nn.Conv1d(512, 64, kernel_size=1)),
        )

        self.generator = Generator(
            input_channels=dim_in,
            dim=dim_in,
            style_dim=style_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            sample_rate=sample_rate,
            n_fft=gen_istft_n_fft,
            win_length=gen_istft_win_length,
            hop_length=gen_istft_hop_length,
        )

    def forward(self, asr, F0_curve, N_curve, s, pretrain=False, probing=False):
        if self.training:
            downlist = [0, 3, 7]
            F0_down = downlist[random.randint(0, 2)]
            downlist = [0, 3, 7, 15]
            N_down = downlist[random.randint(0, 3)]
            if F0_down:
                F0_curve = (
                    nn.functional.conv1d(
                        F0_curve.unsqueeze(1),
                        torch.ones(1, 1, F0_down).to("cuda"),
                        padding=F0_down // 2,
                    ).squeeze(1)
                    / F0_down
                )
            if N_down:
                N_curve = (
                    nn.functional.conv1d(
                        N_curve.unsqueeze(1),
                        torch.ones(1, 1, N_down).to("cuda"),
                        padding=N_down // 2,
                    ).squeeze(1)
                    / N_down
                )

        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N_curve.unsqueeze(1))

        x = torch.cat([asr, F0, N], dim=1)
        x = self.encode(x, s)

        asr_res = self.asr_res(asr)

        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], dim=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False

        x = self.generator(x, s, F0_curve, N_curve)
        x = x.unsqueeze(1)
        return DecoderPrediction(audio=x)
