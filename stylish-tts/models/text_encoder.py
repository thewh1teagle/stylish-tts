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
                # BasicConvNeXtBlock(channels, channels * 2)
                torch.nn.Sequential(
                    weight_norm(
                        torch.nn.Conv1d(
                            channels, channels, kernel_size=kernel_size, padding=padding
                        )
                    ),
                    LayerNorm(channels),
                    actv,
                    torch.nn.Dropout(0.2),
                )
            )

        self.prepare_projection = LinearNorm(channels, channels // 2)
        self.post_projection = LinearNorm(channels // 2, channels)

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            context_length=channels,
            num_blocks=8,
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

        input_lengths = input_lengths.cpu().numpy()

        x = self.prepare_projection(x)
        x = self.lstm(x)
        x = self.post_projection(x)

        x = x.transpose(-1, -2)

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


class BasicConvNeXtBlock(torch.nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
    ):
        super().__init__()
        self.dwconv = torch.nn.Conv1d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = torch.nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = torch.nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = torch.nn.GELU()
        self.pwconv2 = torch.nn.Linear(intermediate_dim, dim)

    def forward(
        self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x
