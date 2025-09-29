"""CNN used for muon energy reconstruction in IceCube.

Copy of `upgoing_muon_energy` model from
https://github.com/IceCubeOpenSource/i3deepice/tree/master

Class and variable names are kept for
compatibility with the original code.
"""

from typing import Tuple, Union

import torch
from torch import nn
from pytorch_lightning import LightningModule
from torch_geometric.data import Data
from .cnn import CNN


class Conv3dBN(LightningModule):
    """The Conv3dBN module from Theos CNN model."""

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
    """The inception_block4 module from Theos CNN model."""

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
    """The inception_resnet module from Theos CNN model."""

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
        """Forward pass of the Conv"."""
        tmp = torch.cat(
            [
                self.tower1(x),
                self.tower2(x),
                self.tower3(x),
            ],
            dim=1,
        )
        return x + self._scale * tmp


class TheosMuonEUpgoing(CNN):
    """The TheosMuonEUpgoing module."""

    def __init__(self, nb_inputs: int = 15, nb_outputs: int = 16) -> None:
        """Construct `TheosMuonEUpgoing`.

        Args:
            nb_inputs: Number of input features.
            nb_outputs: Number of output features.
        """
        super().__init__(nb_inputs, nb_outputs)
        self.inceptionblocks4 = nn.Sequential(
            InceptionBlock4(
                in_channels=nb_inputs,
                out_channels=18,
                t0=2,
                t1=5,
                t2=8,
            ),
            InceptionBlock4(
                in_channels=18 * 4,
                out_channels=18,
                t0=2,
                t1=3,
                t2=7,
            ),
            InceptionBlock4(
                in_channels=18 * 4,
                out_channels=18,
                t0=2,
                t1=4,
                t2=8,
            ),
            InceptionBlock4(
                in_channels=18 * 4,
                out_channels=18,
                t0=3,
                t1=5,
                t2=9,
            ),
            InceptionBlock4(
                in_channels=18 * 4,
                out_channels=18,
                t0=2,
                t1=8,
                t2=9,
            ),
        )
        self.avgpool1 = nn.AvgPool3d((2, 2, 3))
        self.bn1 = nn.BatchNorm3d(18 * 4)
        tmp = [
            InceptionResnet(
                in_channels=18 * 4,
                out_channels=24,
                t2=3,
            ),
            InceptionResnet(
                in_channels=24 * 3,
                out_channels=24,
                t2=4,
            ),
            InceptionResnet(
                in_channels=24 * 3,
                out_channels=24,
                t2=5,
            ),
        ]
        for _ in range(5):
            tmp = tmp + [
                InceptionResnet(
                    in_channels=24 * 3,
                    out_channels=24,
                    t2=3,
                ),
                InceptionResnet(
                    in_channels=24 * 3,
                    out_channels=24,
                    t2=4,
                ),
                InceptionResnet(
                    in_channels=24 * 3,
                    out_channels=24,
                    t2=5,
                ),
            ]

        self.resblocks1 = nn.Sequential(*tmp)
        self.avgpool2 = nn.AvgPool3d((1, 1, 2))
        self.bn2 = nn.BatchNorm3d(24 * 3)
        tmp = []
        for _ in range(6):
            tmp = tmp + [
                InceptionResnet(
                    in_channels=24 * 3,
                    out_channels=24,
                    t2=3,
                ),
                InceptionResnet(
                    in_channels=24 * 3,
                    out_channels=24,
                    t2=4,
                ),
                InceptionResnet(
                    in_channels=24 * 3,
                    out_channels=24,
                    t2=5,
                ),
            ]
        self.resblocks2 = nn.Sequential(*tmp)
        self.convs111 = nn.Sequential(
            nn.Conv3d(
                in_channels=24 * 3,
                out_channels=64,
                kernel_size=(1, 1, 1),
                padding=(0, 0, 0),
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=64,
                out_channels=4,
                kernel_size=(1, 1, 1),
                padding=(0, 0, 0),
            ),
            nn.ReLU(),
        )
        self.avgpool3 = nn.AvgPool3d((1, 1, 2))
        self.mlps = nn.Sequential(
            nn.Linear(500, 120),
            nn.Linear(120, 64),
            nn.Linear(64, 16),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Apply learnable forward pass in model."""
        assert len(data.x) == 1, "Only one image expected"
        x = data.x[0]
        x = self.inceptionblocks4(x)
        x = self.avgpool1(x)
        x = self.bn1(x)
        x = self.resblocks1(x)
        x = self.avgpool2(x)
        x = self.bn2(x)
        x = self.resblocks2(x)
        x = self.convs111(x)
        x = self.avgpool3(x)
        x = nn.Flatten()(x)
        x = self.mlps(x)
        return x
