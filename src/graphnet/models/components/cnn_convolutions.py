"""Convolutional building blocks for use in CNN models."""

from typing import Tuple, Union

import torch
from torch import nn
from pytorch_lightning import LightningModule


class Conv3dBN(LightningModule):
    """3D convolution with batch normalization from Theo Glauch's DNN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        padding: Union[str, Tuple[int, int, int]],
        bias: bool = False,
    ):
        """Create a Conv3dBN module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the kernel.
            padding: Padding of the kernel.
            bias: If True, bias is used in the Convolution.
        """
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Conv3dBN."""
        return self.activation(self.bn(self.conv(x)))
