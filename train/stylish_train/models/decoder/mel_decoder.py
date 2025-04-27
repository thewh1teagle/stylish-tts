import torch
from torch.nn.utils.parametrizations import weight_norm

from ..conv_next import ConvNeXtBlock, BasicConvNeXtBlock


class MelDecoder(torch.nn.Module):
    def __init__(
        self,
        dim_in=512,
        style_dim=128,
        residual_dim=64,
        dim_out=512,
        intermediate_dim=1536,
        num_layers=8,
    ):
        super().__init__()

        bottleneck_dim = dim_in * 2

        self.encode = torch.nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim_in=dim_in + 2 * residual_dim,
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

        self.decode1 = torch.nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim_in=bottleneck_dim + 3 * residual_dim,
                    dim_out=bottleneck_dim,
                    intermediate_dim=bottleneck_dim,
                    style_dim=style_dim,
                    dilation=[1],
                    activation=True,
                ),
                ConvNeXtBlock(
                    dim_in=bottleneck_dim + 3 * residual_dim,
                    dim_out=bottleneck_dim,
                    intermediate_dim=bottleneck_dim,
                    style_dim=style_dim,
                    dilation=[1],
                    activation=True,
                ),
                ConvNeXtBlock(
                    dim_in=bottleneck_dim + 3 * residual_dim,
                    dim_out=dim_in,
                    intermediate_dim=bottleneck_dim,
                    style_dim=style_dim,
                    dilation=[1],
                    activation=True,
                ),
            ]
        )
        self.decode2 = torch.nn.ModuleList(
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

        self.F0_conv = torch.nn.Sequential(
            weight_norm(torch.nn.Conv1d(1, residual_dim, kernel_size=1)),
            BasicConvNeXtBlock(residual_dim, residual_dim * 2),
            torch.nn.InstanceNorm1d(residual_dim, affine=True),
        )

        self.N_conv = torch.nn.Sequential(
            weight_norm(torch.nn.Conv1d(1, residual_dim, kernel_size=1)),
            BasicConvNeXtBlock(residual_dim, residual_dim * 2),
            torch.nn.InstanceNorm1d(residual_dim, affine=True),
        )

        self.asr_res = torch.nn.Sequential(
            weight_norm(torch.nn.Conv1d(dim_in, residual_dim, kernel_size=1)),
            torch.nn.InstanceNorm1d(residual_dim, affine=True),
        )

        self.to_out = torch.nn.Sequential(
            weight_norm(torch.nn.Conv1d(dim_in, dim_out, 1, 1, 0))
        )

    def forward(self, asr, F0_curve, N_curve, s, pretrain=False, probing=False):
        # asr = torch.nn.functional.interpolate(asr, scale_factor=2, mode="nearest")
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
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

        return self.to_out(x)
