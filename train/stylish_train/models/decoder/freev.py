import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
import numpy as np
from munch import Munch
from librosa.filters import mel as librosa_mel_fn
from torchaudio.functional import melscale_fbanks
import random
from scipy.signal import get_window

import math

from torch.nn.utils.parametrizations import weight_norm
from utils import DecoderPrediction
from .harmonics import HarmonicGenerator
from ..conv_next import ConvNeXtBlock, BasicConvNeXtBlock
from ..common import get_padding


class InverseMel:
    def __init__(
        self,
        *,
        n_fft,
        num_mels,
        sampling_rate,
        hop_size,
        win_size,
        fmin,
        fmax,
        # device
    ):
        # mel_np = librosa_mel_fn(
        #     sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        # )
        mel_basis = melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0,
            f_max=12000,
            n_mels=num_mels,
            sample_rate=sampling_rate,
        )
        mel_basis = mel_basis.transpose(0, 1)
        # mel_basis = torch.from_numpy(mel_np).float() # .to(device)
        # hann_window = torch.hann_window(win_size).to(device)
        self.inv_basis = mel_basis.pinverse()  # .to(device)

    def execute(self, mel):
        return self.inv_basis.to(mel.device) @ spectral_de_normalize_torch(mel)


# mel_window = {}
# inv_mel_window = {}
#
#
# def inverse_mel(
#     mel,
#     n_fft,
#     num_mels,
#     sampling_rate,
#     hop_size,
#     win_size,
#     fmin,
#     fmax,
#     in_dataset=False,
# ):
#     global inv_mel_window, mel_window
#     device = torch.device("cpu") if in_dataset else mel.device
#     ps = param_string(sampling_rate, n_fft, num_mels, fmin, fmax, win_size, device)
#     if ps in inv_mel_window:
#         inv_basis = inv_mel_window[ps]
#     else:
#         if ps in mel_window:
#             mel_basis, _ = mel_window[ps]
#         else:
#             mel_np = librosa_mel_fn(
#                 sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
#             )
#             mel_basis = torch.from_numpy(mel_np).float().to(device)
#             hann_window = torch.hann_window(win_size).to(device)
#             mel_window[ps] = (mel_basis.clone(), hann_window.clone())
#         inv_basis = mel_basis.pinverse()
#         inv_mel_window[ps] = inv_basis.clone()
#     return inv_basis.to(device) @ spectral_de_normalize_torch(mel.to(device))


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
        "ASP_channel": 512,
        "ASP_input_conv_kernel_size": 7,
        "ASP_output_conv_kernel_size": 7,
        "PSP_channel": 512,
        "PSP_input_conv_kernel_size": 7,
        "PSP_output_R_conv_kernel_size": 7,
        "PSP_output_I_conv_kernel_size": 7,
        "num_mels": 512,
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


class FreevGenerator(torch.nn.Module):
    def __init__(self):
        super(FreevGenerator, self).__init__()
        h = config_h
        self.h = config_h
        self.dim = 512
        self.num_layers = 8
        window = torch.hann_window(h.win_size)
        self.register_buffer("window", window, persistent=False)

        # self.harmonic = HarmonicGenerator(
        #     sample_rate=h.sampling_rate,
        #     dim_out=self.dim * 2,
        #     win_length=h.win_size,
        #     hop_length=h.hop_size,
        #     divisor=2,
        # )

        # self.ASP_harmonic_conv = torch.nn.Linear(
        #     self.dim,
        #     h.ASP_channel,
        #     bias=False,
        #     # h.ASP_input_conv_kernel_size,
        #     # 1,
        #     # padding=get_padding(h.ASP_input_conv_kernel_size, 1),
        # )

        self.ASP_input_conv = Conv1d(
            h.num_mels,
            h.ASP_channel,
            h.ASP_input_conv_kernel_size,
            1,
            padding=get_padding(h.ASP_input_conv_kernel_size, 1),
        )

        self.PSP_input_conv = Conv1d(
            h.num_mels,
            h.PSP_channel,
            h.PSP_input_conv_kernel_size,
            1,
            padding=get_padding(h.PSP_input_conv_kernel_size, 1),
        )

        self.ASP_output_conv = Conv1d(
            h.ASP_channel,
            h.n_fft // 2 + 1,
            h.ASP_output_conv_kernel_size,
            1,
            padding=get_padding(h.ASP_output_conv_kernel_size, 1),
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

        self.amp_norm = nn.LayerNorm(self.dim, eps=1e-6)
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
                for _ in range(self.num_layers)
            ]
        )
        self.amp_final_layer_norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.phase_final_layer_norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.apply(self._init_weights)
        # self.inverse_mel = InverseMel(
        #     n_fft=self.h.n_fft,
        #     num_mels=self.h.num_mels,
        #     sampling_rate=self.h.sampling_rate,
        #     hop_size=self.h.hop_size,
        #     win_size=self.h.win_size,
        #     fmin=self.h.fmin,
        #     fmax=self.h.fmax,
        #     # device=self.device
        # )

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):  # (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, *, mel, style, pitch, energy):
        # har_spec, har_phase = self.harmonic(pitch, energy)
        # har_spec = har_spec.transpose(1, 2)
        # har_spec = self.ASP_harmonic_conv(har_spec)
        # har_spec = har_spec.transpose(1, 2)
        # inv_amp = self.inverse_mel.execute(mel).abs().clamp_min(1e-5)
        # logamp = inv_amp.log()

        logamp = self.ASP_input_conv(mel)
        logamp = self.amp_norm(logamp.transpose(1, 2))
        logamp = logamp.transpose(1, 2)
        for conv_block in self.amp_convnext:
            logamp = conv_block(logamp, style)
        logamp = self.amp_final_layer_norm(logamp.transpose(1, 2))
        logamp = logamp.transpose(1, 2)
        logamp = self.ASP_output_conv(logamp)

        pha = self.PSP_input_conv(mel)
        pha = self.phase_norm(pha.transpose(1, 2))
        pha = pha.transpose(1, 2)
        for conv_block in self.phase_convnext:
            pha = conv_block(pha, style)
        pha = self.phase_final_layer_norm(pha.transpose(1, 2))
        pha = pha.transpose(1, 2)
        R = self.PSP_output_R_conv(pha)
        I = self.PSP_output_I_conv(pha)

        pha = torch.atan2(I, R)

        logamp = F.pad(logamp, pad=(0, 1), mode="replicate")
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

        audio = torch.tanh(audio)
        return DecoderPrediction(
            audio=audio.unsqueeze(1),
            log_amplitude=logamp,
            phase=pha,
            real=rea,
            imaginary=imag,
        )
