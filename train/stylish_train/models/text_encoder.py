from typing import Optional
import torch
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
)
from .common import LinearNorm, get_padding
from .conv_next import BasicConvNeXtBlock


class TextEncoder(torch.nn.Module):
    def __init__(
        self, channels, kernel_size, depth, n_symbols, actv=torch.nn.LeakyReLU(0.2)
    ):
        super().__init__()
        padding = get_padding(kernel_size)
        self.embedding = torch.nn.Embedding(n_symbols, channels)

        self.cnn = torch.nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(
                BasicConvNeXtBlock(channels, channels * 2)
                # torch.nn.Sequential(
                #     weight_norm(
                #         torch.nn.Conv1d(
                #             channels, channels, kernel_size=kernel_size, padding=padding
                #         )
                #     ),
                #     LayerNorm(channels),
                #     actv,
                #     torch.nn.Dropout(0.2),
                # )
            )

        # self.prepare_projection = LinearNorm(channels, channels // 2)
        # self.post_projection = LinearNorm(channels // 2, channels)

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            context_length=channels,
            num_blocks=8,
            # embedding_dim=channels // 2,
            embedding_dim=channels,
        )

        self.lstm = xLSTMBlockStack(cfg)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)

        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)

        x = x.transpose(1, 2)  # [B, T, chn]

        # x = self.prepare_projection(x)
        x = self.lstm(x)
        # x = self.post_projection(x)

        x = x.transpose(1, 2)

        x.masked_fill_(m, 0.0)

        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        x, _ = self.lstm(x)
        return x

    def length_to_mask(self, lengths):
        mask = (
            torch.arange(lengths.max())
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
            .type_as(lengths)
        )

        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask


class LayerNorm(torch.nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)
