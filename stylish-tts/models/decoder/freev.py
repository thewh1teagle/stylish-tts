import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
import numpy as np
from munch import Munch
from librosa.filters import mel as librosa_mel_fn
import random
from scipy.signal import get_window

import math

# import torch
# from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from utils import DecoderPrediction
from .harmonics import HarmonicGenerator
from ..conv_next import ConvNeXtBlock
from ..common import get_padding

# import torch.nn.functional as F

mel_window = {}
inv_mel_window = {}


def inverse_mel(
    mel,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    in_dataset=False,
):
    global inv_mel_window, mel_window
    device = torch.device("cpu") if in_dataset else mel.device
    ps = param_string(sampling_rate, n_fft, num_mels, fmin, fmax, win_size, device)
    if ps in inv_mel_window:
        inv_basis = inv_mel_window[ps]
    else:
        if ps in mel_window:
            mel_basis, _ = mel_window[ps]
        else:
            mel_np = librosa_mel_fn(
                sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
            )
            mel_basis = torch.from_numpy(mel_np).float().to(device)
            hann_window = torch.hann_window(win_size).to(device)
            mel_window[ps] = (mel_basis.clone(), hann_window.clone())
        inv_basis = mel_basis.pinverse()
        inv_mel_window[ps] = inv_basis.clone()
    return inv_basis.to(device) @ spectral_de_normalize_torch(mel.to(device))


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def param_string(sampling_rate, n_fft, num_mels, fmin, fmax, win_size, device):
    return f"{sampling_rate}-{n_fft}-{num_mels}-{fmin}-{fmax}-{win_size}-{device}"


LRELU_SLOPE = 0.1


config_h = Munch(
    {
        "ASP_channel": 1025,
        "ASP_input_conv_kernel_size": 7,
        "PSP_channel": 512,
        "PSP_input_conv_kernel_size": 7,
        "PSP_output_R_conv_kernel_size": 7,
        "PSP_output_I_conv_kernel_size": 7,
        "num_mels": 80,
        "n_fft": 2048,
        "hop_size": 300,
        "win_size": 1200,
        "sampling_rate": 24000,
        "fmin": 50,
        "fmax": 550,
        "style_dim": 128,
        "intermediate_dim": 1536,
    }
)


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        h = config_h
        self.h = config_h
        self.dim = 512
        self.num_layers = 8
        window = torch.hann_window(h.win_size)
        self.register_buffer("window", window, persistent=False)

        self.harmonic = HarmonicGenerator(
            sample_rate=h.sampling_rate,
            dim_out=self.dim * 2,
            win_length=h.win_size,
            hop_length=h.hop_size,
            divisor=2,
        )

        self.ASP_harmonic_conv = torch.nn.Linear(
            self.dim,
            h.ASP_channel,
            bias=False,
            # h.ASP_input_conv_kernel_size,
            # 1,
            # padding=get_padding(h.ASP_input_conv_kernel_size, 1),
        )

        self.PSP_input_conv = Conv1d(
            h.num_mels,
            h.PSP_channel,
            h.PSP_input_conv_kernel_size,
            1,
            padding=get_padding(h.PSP_input_conv_kernel_size, 1),
        )

        self.PSP_output_R_conv = Conv1d(
            h.PSP_channel,
            h.n_fft // 2 + 1,
            h.PSP_output_R_conv_kernel_size,
            1,
            padding=get_padding(h.PSP_output_R_conv_kernel_size, 1),
        )
        self.PSP_output_I_conv = Conv1d(
            h.PSP_channel,
            h.n_fft // 2 + 1,
            h.PSP_output_I_conv_kernel_size,
            1,
            padding=get_padding(h.PSP_output_I_conv_kernel_size, 1),
        )

        self.phase_norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.phase_convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim_in=h.PSP_channel,
                    dim_out=h.PSP_channel,
                    intermediate_dim=h.intermediate_dim,
                    style_dim=h.style_dim,
                    dilation=[1, 3, 5],
                )
                for _ in range(self.num_layers)
            ]
        )
        self.amp_convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim_in=h.ASP_channel,
                    dim_out=h.ASP_channel,
                    intermediate_dim=h.intermediate_dim,
                    style_dim=h.style_dim,
                    dilation=[1, 3, 5],
                )
                for _ in range(1)
            ]
        )
        self.phase_final_layer_norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):  # (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, *, mel, style, pitch, energy):
        har_spec, har_phase = self.harmonic(pitch, energy)
        har_spec = har_spec.transpose(1, 2)
        har_spec = self.ASP_harmonic_conv(har_spec)
        har_spec = har_spec.transpose(1, 2)
        inv_amp = (
            inverse_mel(
                mel,
                self.h.n_fft,
                self.h.num_mels,
                self.h.sampling_rate,
                self.h.hop_size,
                self.h.win_size,
                self.h.fmin,
                self.h.fmax,
            )
            .abs()
            .clamp_min(1e-5)
        )
        logamp = inv_amp.log()
        for conv_block in self.amp_convnext:
            logamp = conv_block(logamp, style, har_spec)
        logamp = F.pad(logamp, pad=(0, 1), mode="replicate")

        pha = self.PSP_input_conv(mel)
        pha = self.phase_norm(pha.transpose(1, 2))
        pha = pha.transpose(1, 2)
        for conv_block in self.phase_convnext:
            pha = conv_block(pha, style, har_phase)
        pha = self.phase_final_layer_norm(pha.transpose(1, 2))
        pha = pha.transpose(1, 2)
        R = self.PSP_output_R_conv(pha)
        I = self.PSP_output_I_conv(pha)

        pha = torch.atan2(I, R)
        pha = F.pad(pha, pad=(0, 1), mode="replicate")
        # rea is the real part of the complex number
        rea = torch.exp(logamp) * torch.cos(pha)
        # imag is the imaginary part of the complex number
        imag = torch.exp(logamp) * torch.sin(pha)

        spec = torch.complex(rea, imag)

        audio = torch.istft(
            spec,
            self.h.n_fft,
            hop_length=self.h.hop_size,
            win_length=self.h.win_size,
            window=self.window,
            center=True,
        )

        return (
            audio.unsqueeze(1),
            logamp,
            pha,
            rea,
            imag,
        )


###################################################################


class ResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        actv=nn.LeakyReLU(0.2),
        normalize=False,
        downsample="none",
        dropout_p=0.2,
    ):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample_type = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)
        self.dropout_p = dropout_p

        if self.downsample_type == "none":
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(
                nn.Conv1d(
                    dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1
                )
            )

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def downsample(self, x):
        if self.downsample_type == "none":
            return x
        else:
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool1d(x, 2)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv1(x)
        x = self.pool(x)
        if self.normalize:
            x = self.norm2(x)

        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class Decoder(nn.Module):
    def __init__(
        self,
        dim_in=512,
        style_dim=128,
        residual_dim=64,
        dim_out=80,
        intermediate_dim=1536,
        num_layers=8,
    ):
        super().__init__()

        bottleneck_dim = dim_in * 2

        self.encode = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim_in=dim_in + 2,
                    dim_out=bottleneck_dim,
                    intermediate_dim=bottleneck_dim,
                    style_dim=style_dim,
                    dilation=[1],
                    activation=True,
                ),
                ConvNeXtBlock(
                    dim_in=bottleneck_dim,
                    dim_out=bottleneck_dim,
                    intermediate_dim=bottleneck_dim,
                    style_dim=style_dim,
                    dilation=[1],
                    activation=True,
                ),
            ]
        )

        self.decode1 = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim_in=bottleneck_dim + residual_dim + 2,
                    dim_out=bottleneck_dim,
                    intermediate_dim=bottleneck_dim,
                    style_dim=style_dim,
                    dilation=[1],
                    activation=True,
                ),
                ConvNeXtBlock(
                    dim_in=bottleneck_dim + residual_dim + 2,
                    dim_out=bottleneck_dim,
                    intermediate_dim=bottleneck_dim,
                    style_dim=style_dim,
                    dilation=[1],
                    activation=True,
                ),
                ConvNeXtBlock(
                    dim_in=bottleneck_dim + residual_dim + 2,
                    dim_out=dim_in,
                    intermediate_dim=bottleneck_dim,
                    style_dim=style_dim,
                    dilation=[1],
                    activation=True,
                ),
            ]
        )
        self.decode2 = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim_in=dim_in,
                    dim_out=dim_in,
                    intermediate_dim=bottleneck_dim,
                    style_dim=style_dim,
                    dilation=[1],
                    activation=True,
                ),
                ConvNeXtBlock(
                    dim_in=dim_in,
                    dim_out=dim_in,
                    intermediate_dim=bottleneck_dim,
                    style_dim=style_dim,
                    dilation=[1],
                    activation=True,
                ),
            ]
        )

        # self.F0_block = ConvNeXtBlock(
        #     dim_in=1,
        #     dim_out=residual_dim,
        #     intermediate_dim=bottleneck_dim,
        #     style_dim=style_dim,
        #     dilation=[1],
        #     activation=True,
        # )

        self.F0_conv = nn.Sequential(
            ResBlk1d(1, residual_dim, normalize=True, downsample="none"),
            weight_norm(nn.Conv1d(residual_dim, 1, kernel_size=1)),
            nn.InstanceNorm1d(1, affine=True),
        )

        # self.N_block = ConvNeXtBlock(
        #     dim_in=1,
        #     dim_out=residual_dim,
        #     intermediate_dim=bottleneck_dim,
        #     style_dim=style_dim,
        #     dilation=[1],
        #     activation=True,
        # )

        self.N_conv = nn.Sequential(
            ResBlk1d(1, residual_dim, normalize=True, downsample="none"),
            weight_norm(nn.Conv1d(residual_dim, 1, kernel_size=1)),
            nn.InstanceNorm1d(1, affine=True),
        )

        self.asr_res = nn.Sequential(
            weight_norm(nn.Conv1d(dim_in, residual_dim, kernel_size=1)),
            nn.InstanceNorm1d(residual_dim, affine=True),
        )

        self.to_out = nn.Sequential(weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0)))

        self.generator = Generator()

    def forward(self, asr, F0_curve, N_curve, s, pretrain=False, probing=False):
        asr = F.interpolate(asr, scale_factor=2, mode="nearest")
        # F0 = self.F0_block(F0_curve.unsqueeze(1), s)
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        # N = self.N_block(N_curve.unsqueeze(1), s)
        N = self.N_conv(N_curve.unsqueeze(1))

        x = torch.cat([asr, F0, N], axis=1)
        for block in self.encode:
            x = block(x, s)

        asr_res = self.asr_res(asr)

        for block in self.decode1:
            x = torch.cat([x, asr_res, F0, N], axis=1)
            x = block(x, s)

        for block in self.decode2:
            x = block(x, s)

        x = self.to_out(x)
        audio, logamp, phase, real, imaginary = self.generator(
            mel=x, style=s, pitch=F0_curve, energy=N_curve
        )
        audio = torch.tanh(audio)
        return DecoderPrediction(
            audio=audio,
            log_amplitude=logamp,
            phase=phase,
            real=real,
            imaginary=imaginary,
        )
