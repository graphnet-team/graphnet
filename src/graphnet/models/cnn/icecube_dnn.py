"""Implementation of the IceCube DNN image convolution model by Theo Glauch.

Based on the `upgoing_muon_energy` model from
https://github.com/IceCubeOpenSource/i3deepice/tree/master
"""

from typing import List, Tuple

import torch
from torch import nn
from torch_geometric.data import Data

from graphnet.models.components.cnn_convolutions import (
    InceptionBlock4,
    InceptionResnet,
)
from .cnn import CNN


class IceCubeDNN(CNN):
    """Implementation of the IceCube DNN by Theo Glauch.

    An inception-based 3D CNN originally used within IceCube. Based on
    the model from
    https://github.com/IceCubeOpenSource/i3deepice/tree/master
    """

    def __init__(
        self,
        nb_inputs: int = 15,
        nb_outputs: int = 16,
        image_size: Tuple[int, int, int] = (10, 10, 60),
        inception_out_channels: int = 18,
        inception_configs: List[Tuple[int, int, int]] = [
            (2, 5, 8),
            (2, 3, 7),
            (2, 4, 8),
            (3, 5, 9),
            (2, 8, 9),
        ],
        resnet_out_channels: int = 24,
        resnet_t2_pattern: List[int] = [3, 4, 5],
        num_resblocks1_repeats: int = 6,
        num_resblocks2_repeats: int = 6,
        avgpool1_size: Tuple[int, int, int] = (2, 2, 3),
        avgpool2_size: Tuple[int, int, int] = (1, 1, 2),
        avgpool3_size: Tuple[int, int, int] = (1, 1, 2),
        pointwise_channels: List[int] = [64, 4],
        mlp_hidden_sizes: List[int] = [120, 64],
    ) -> None:
        """Construct `IceCubeDNN`.

        Args:
            nb_inputs: Number of input features.
            nb_outputs: Number of output features.
            image_size: Spatial dimensions of the input image
                (height, width, depth).
            inception_out_channels: Output channels per tower in each
                inception block.
            inception_configs: List of (t0, t1, t2) kernel size tuples
                for each InceptionBlock4 layer.
            resnet_out_channels: Output channels per tower in each
                inception-resnet block.
            resnet_t2_pattern: Pattern of t2 kernel sizes repeated in
                each group of resnet blocks.
            num_resblocks1_repeats: Number of times to repeat the
                resnet_t2_pattern in the first resnet stage.
            num_resblocks2_repeats: Number of times to repeat the
                resnet_t2_pattern in the second resnet stage.
            avgpool1_size: Kernel size for the first average pooling.
            avgpool2_size: Kernel size for the second average pooling.
            avgpool3_size: Kernel size for the third average pooling.
            pointwise_channels: Output channels for each 1x1x1
                convolution layer.
            mlp_hidden_sizes: Hidden layer sizes for the final MLP.
                The input size is computed from the preceding layers
                and the output size is nb_outputs.
        """
        super().__init__(nb_inputs, nb_outputs)

        # Inception blocks
        inception_blocks = []
        in_ch = nb_inputs
        for t0, t1, t2 in inception_configs:
            inception_blocks.append(
                InceptionBlock4(
                    in_channels=in_ch,
                    out_channels=inception_out_channels,
                    t0=t0,
                    t1=t1,
                    t2=t2,
                )
            )
            in_ch = inception_out_channels * 4
        self.inceptionblocks4 = nn.Sequential(*inception_blocks)

        # All inception/resnet blocks use "same" padding, so spatial
        # dimensions only change at pooling layers.
        spatial = list(image_size)

        self.avgpool1 = nn.AvgPool3d(avgpool1_size)
        spatial = [s // p for s, p in zip(spatial, avgpool1_size)]
        self.bn1 = nn.BatchNorm3d(in_ch)

        # First resnet stage
        resnet_in_ch = in_ch
        tmp = []
        for _ in range(num_resblocks1_repeats):
            for t2 in resnet_t2_pattern:
                tmp.append(
                    InceptionResnet(
                        in_channels=resnet_in_ch,
                        out_channels=resnet_out_channels,
                        t2=t2,
                    )
                )
                resnet_in_ch = resnet_out_channels * 3
        self.resblocks1 = nn.Sequential(*tmp)

        self.avgpool2 = nn.AvgPool3d(avgpool2_size)
        spatial = [s // p for s, p in zip(spatial, avgpool2_size)]
        self.bn2 = nn.BatchNorm3d(resnet_in_ch)

        # Second resnet stage
        tmp = []
        for _ in range(num_resblocks2_repeats):
            for t2 in resnet_t2_pattern:
                tmp.append(
                    InceptionResnet(
                        in_channels=resnet_in_ch,
                        out_channels=resnet_out_channels,
                        t2=t2,
                    )
                )
                resnet_in_ch = resnet_out_channels * 3
        self.resblocks2 = nn.Sequential(*tmp)

        # Pointwise 1x1x1 convolutions
        pointwise_layers: List[nn.Module] = []
        pw_in = resnet_in_ch
        for pw_out in pointwise_channels:
            pointwise_layers.append(
                nn.Conv3d(
                    in_channels=pw_in,
                    out_channels=pw_out,
                    kernel_size=(1, 1, 1),
                    padding=(0, 0, 0),
                )
            )
            pointwise_layers.append(nn.ReLU())
            pw_in = pw_out
        self.convs111 = nn.Sequential(*pointwise_layers)

        self.avgpool3 = nn.AvgPool3d(avgpool3_size)
        spatial = [s // p for s, p in zip(spatial, avgpool3_size)]

        # MLP head
        latent_dim = pw_in * spatial[0] * spatial[1] * spatial[2]
        mlp_sizes = [latent_dim] + mlp_hidden_sizes + [nb_outputs]
        mlp_layers: List[nn.Module] = []
        for i in range(len(mlp_sizes) - 1):
            mlp_layers.append(nn.Linear(mlp_sizes[i], mlp_sizes[i + 1]))
        self.mlps = nn.Sequential(*mlp_layers)

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
