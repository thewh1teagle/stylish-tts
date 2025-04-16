from typing import List, Optional
import torch
from torch.nn.utils.parametrizations import weight_norm

from .common import get_padding, init_weights, leaky_clamp


class ConvNeXtBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        dim_in: int,
        dim_out: int,
        intermediate_dim: int,
        style_dim: int,
        dilation: List[int],
        activation: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim_out = dim_out
        self.dwconv = weight_norm(
            torch.nn.Conv1d(dim_in, dim_in, kernel_size=7, padding=3, groups=dim_in)
        )  # depthwise conv

        self.norm = AdaResBlock(
            channels=dim_in,
            kernel_size=7,
            style_dim=style_dim,
            dilation=dilation,
            activation=activation,
            dropout_p=dropout,
        )
        self.pwconv1 = torch.nn.Linear(
            dim_in, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = torch.nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = torch.nn.Linear(intermediate_dim, dim_out)
        self.shortcut = None
        if dim_in != dim_out:
            self.shortcut = weight_norm(
                torch.nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False)
            )

    def forward(
        self, x: torch.Tensor, s: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x_in = x
        if self.shortcut is not None:
            x_in = self.shortcut(x_in)
        x = self.dwconv(x)
        x = self.norm(x, s, h)

        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        # x = x + self.shortcut(x_in.transpose(1, 2)).transpose(1, 2)
        return x + x_in


class AdaNorm1d(torch.nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = torch.nn.InstanceNorm1d(num_features, affine=False)
        self.fc = torch.nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        result = (1 + gamma) * self.norm(leaky_clamp(x, -1e10, 1e10)) + beta
        return result


class AdaResBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        channels,
        kernel_size,
        style_dim,
        dilation=[1, 3, 5],
        activation=False,
        dropout_p=0.0,
    ):
        super(AdaResBlock, self).__init__()
        self.num_layers = len(dilation)
        self.convs1 = torch.nn.ModuleList(
            [
                weight_norm(
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[i],
                        padding=get_padding(kernel_size, dilation[i]),
                    )
                )
                for i in range(self.num_layers)
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = torch.nn.ModuleList(
            [
                weight_norm(
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in range(self.num_layers)
            ]
        )
        self.convs2.apply(init_weights)

        self.adain1 = torch.nn.ModuleList(
            [AdaNorm1d(style_dim, channels) for _ in range(self.num_layers)]
        )

        self.adain2 = torch.nn.ModuleList(
            [AdaNorm1d(style_dim, channels) for _ in range(self.num_layers)]
        )
        self.activation = None
        if activation:
            self.activation = torch.nn.LeakyReLU(0.2)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.conv1x1 = None

    def forward(self, x, style, harmonics=None):
        for i in range(self.num_layers):
            xt = self.adain1[i](x, style)
            if harmonics is not None:
                xt = xt + harmonics
            if self.activation is not None:
                xt = self.activation(xt)
            xt = self.dropout(xt)
            xt = self.convs1[i](xt)

            xt = self.adain2[i](xt, style)
            if harmonics is not None:
                xt = xt + harmonics
            if self.activation is not None:
                xt = self.activation(xt)
            xt = self.dropout(xt)
            xt = self.convs2[i](xt)

            x = x + xt

        return x


class GRN(torch.nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = torch.nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class BasicConvNeXtBlock(torch.nn.Module):
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
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = torch.nn.Linear(intermediate_dim, dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x
