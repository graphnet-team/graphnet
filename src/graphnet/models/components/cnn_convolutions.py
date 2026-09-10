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


class InceptionBlock4(LightningModule):
    """Inception block with 4 parallel towers from Theo Glauch's DNN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t0: int = 2,
        t1: int = 4,
        t2: int = 5,
        n_pool: int = 3,
    ):
        """Create a InceptionBlock4 module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            t0: Size of the first kernel sequence.
            t1: Size of the second kernel sequence.
            t2: Size of the third kernel sequence.
            n_pool: Size of the pooling kernel.
        """
        super().__init__()

        self.tower0 = nn.Sequential(
            Conv3dBN(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(t0, 1, 1),
                padding="same",
            ),
            Conv3dBN(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, t0, 1),
                padding="same",
            ),
            Conv3dBN(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, 1, t0),
                padding="same",
            ),
        )

        self.tower1 = nn.Sequential(
            Conv3dBN(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(t1, 1, 1),
                padding="same",
            ),
            Conv3dBN(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, t1, 1),
                padding="same",
            ),
            Conv3dBN(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, 1, t1),
                padding="same",
            ),
        )

        self.tower4 = nn.Sequential(
            Conv3dBN(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1, t2),
                padding="same",
            ),
        )

        self.tower3 = nn.Sequential(
            nn.MaxPool3d(
                kernel_size=(n_pool, n_pool, n_pool),
                stride=(1, 1, 1),
                padding=(n_pool // 2, n_pool // 2, n_pool // 2),
            ),
            Conv3dBN(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1, 1),
                padding="same",
            ),
        )
        self.out_channels = out_channels * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the InceptionBlock4."""
        ret = torch.cat(
            [
                self.tower0(x),
                self.tower1(x),
                self.tower3(x),
                self.tower4(x),
            ],
            dim=1,
        )
        return ret


class InceptionResnet(LightningModule):
    """Inception block with residual connections from Theo Glauch's DNN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t1: int = 2,
        t2: int = 4,
        n_pool: int = 3,
        scale: float = 0.1,
    ):
        """Create a InceptionResnet module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            t1: Size of the first kernel sequence.
            t2: Size of the second kernel sequence.
            n_pool: Size of the pooling kernel.
            scale: Scaling factor for the residual connection.
        """
        super().__init__()
        self._scale = scale
        self.tower1 = nn.Sequential(
            Conv3dBN(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1, 1),
                padding="same",
            ),
            Conv3dBN(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(t1, 1, 1),
                padding="same",
            ),
            Conv3dBN(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, t1, 1),
                padding="same",
            ),
            Conv3dBN(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, 1, t1),
                padding="same",
            ),
        )
        self.tower2 = nn.Sequential(
            Conv3dBN(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1, 1),
                padding="same",
            ),
            Conv3dBN(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(t2, 1, 1),
                padding="same",
            ),
            Conv3dBN(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, t2, 1),
                padding="same",
            ),
            Conv3dBN(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, 1, t2),
                padding="same",
            ),
        )
        self.tower3 = nn.Sequential(
            nn.MaxPool3d(
                kernel_size=(n_pool, n_pool, n_pool),
                stride=(1, 1, 1),
                padding=(n_pool // 2, n_pool // 2, n_pool // 2),
            ),
            Conv3dBN(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1, 1),
                padding="same",
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the InceptionResnet block."""
        tmp = torch.cat(
            [
                self.tower1(x),
                self.tower2(x),
                self.tower3(x),
            ],
            dim=1,
        )
        return x + self._scale * tmp
