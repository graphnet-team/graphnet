"""Module for the Lightning CNN signal classifier (LCSC).

All credits go to Alexander Harnisch (https://github.com/AlexHarn)
"""

from .cnn import CNN
import torch
from torch_geometric.data import Data
from typing import List, Union


class LCSC(CNN):
    """Lightning CNN Signal Classifier (LCSC).

    All credits go to Alexander Harnisch (https://github.com/AlexHarn)
    """

    def __init__(
        self,
        num_input_features: int,
        out_put_dim: int = 2,
        input_norm: bool = True,
        num_conv_layers: int = 8,
        conv_filters_list: List[int] = [50, 50, 50, 50, 50, 50, 50, 10],
        kernel_size_list: Union[int, List[Union[int, List[int]]]] = 3,
        padding_list: str = "Same",
        pooling_type_list: List[Union[None, str]] = [
            None,
            "Avg",
            None,
            "Avg",
            None,
            "Avg",
            None,
            "Avg",
        ],
        pooling_kernel_size_list: List[Union[None, int, List[int]]] = [
            None,
            [1, 1, 2],
            None,
            [2, 2, 2],
            None,
            [2, 2, 2],
            None,
            [2, 2, 2],
        ],
        pooling_stride_list: List[Union[None, int, List[int]]] = [
            None,
            [1, 1, 2],
            None,
            [2, 2, 2],
            None,
            [2, 2, 2],
            None,
            [2, 2, 2],
        ],
        num_fc_neurons: int = 50,
        norm_list: bool = True,
        norm_type: str = "Batch",
    ) -> None:
        """Initialize the Lightning CNN signal classifier (LCSC).

        Args:
            num_input_features (int): Number of input features.
            out_put_dim (int): Number of output dimensions of final MLP.
                Defaults to 2.
            input_norm (bool): Whether to apply normalization to the input.
                Defaults to True.
            num_conv_layers (int): Number of convolutional layers.
                Defaults to 8.
            conv_filters_list (List[int]): List of number ofconvolutional
                filters to use in hidden layers.
                Defaults to [50, 50, 50, 50, 50, 50, 50, 50, 10].
            kernel_size_list (int, List[int], or List[List[int]]):
                Size of the convolutional kernels.
                Options are:
                    int: single integer for all dimensions
                        and all layers,
                        e.g. 3 would equal [3, 3, 3].
                    list: list of integers specifying the kernel size,
                        for each layer for all dimensions equally,
                        e.g. [3, 5, 6].
                    If a list of lists is provided, each list will be used
                        for the corresponding layer as kernel size.
                If an integer is provided, it will be used for all layers.
                Defaults to 3.
            padding_list (str or int): Padding for the convolutional layers.
                Either 'Same' or an integer which will be used for all layers.
                Defaults to 'Same'.
            pooling_type_list (List[str]): List of pooling types for layers.
                Options are
                    'None'  : No pooling is used,
                    'Avg'   : Average pooling is used,
                    'Max'   : Max pooling is used
                    Defaults to [
                        None, 'Avg',
                        None, 'Avg',
                        None, 'Avg',
                        None, 'Avg'
                    ].
            pooling_kernel_size_list (List[Union[int,List[int]]]):
                List of pooling kernel sizes for each layer.
                If an integer is provided, it will be used for all layers.
                Options of list elements are:
                    list: list of integers for each dimension, e.g. [1, 1, 2].
                    int: single integer for all dimensions,
                        e.g. 2 would equal [2, 2, 2].
                    If None, no pooling is applied.
                Defaults to [
                    None, [1, 1, 2],
                    None, [2, 2, 2],
                    None, [2, 2, 2],
                    None, [2, 2, 2]
                ].
            pooling_stride_list (List[List[int]]): List of pooling strides
                for each layer.
                If an integer is provided, it will be used for all layers.
                Defaults to [
                    None, [1, 1, 2],
                    None, [2, 2, 2],
                    None, [2, 2, 2],
                    None, [2, 2, 2]
                ].
            num_fc_neurons (int): Number of neurons in the
                fully connected layers.
                Defaults to 50.
            norm_list (bool or List[bool]): Whether to apply normalization
                for each convolutional layer.
                If a boolean is provided, it will be used for all layers.
                Defaults to True.
            norm_type (str): Type of normalization to use.
                Options are 'Batch' or 'Instance'.
                Defaults to 'Batch'.
        """
        super().__init__(nb_inputs=num_input_features, nb_outputs=out_put_dim)

        # Check input parameters
        if isinstance(conv_filters_list, int):
            conv_filters_list = [
                conv_filters_list for _ in range(num_conv_layers)
            ]
        else:
            if not isinstance(conv_filters_list, list):
                raise TypeError(
                    (
                        f"`conv_filters_list` must be a "
                        f"list or an integer, not {type(conv_filters_list)}!"
                    )
                )
            if len(conv_filters_list) != num_conv_layers:
                raise ValueError(
                    f"`conv_filters_list` must have {num_conv_layers} "
                    f"elements, not {len(conv_filters_list)}!"
                )

        if isinstance(kernel_size_list, int):
            kernel_size_list = [  # type: ignore[assignment]
                [kernel_size_list, kernel_size_list, kernel_size_list]
                for _ in range(num_conv_layers)
            ]
        else:
            if not isinstance(kernel_size_list, list):
                raise TypeError(
                    (
                        "`kernel_size_list` must be a list or an "
                        f"integer, not {type(kernel_size_list)}!"
                    )
                )
            if len(kernel_size_list) != num_conv_layers:
                raise ValueError(
                    (
                        f"`kernel_size_list` must have {num_conv_layers} "
                        f"elements, not {len(kernel_size_list)}!"
                    )
                )

        if isinstance(padding_list, int):
            padding_list = [padding_list for _ in range(num_conv_layers)]
        elif padding_list.lower() == "same":
            self._padding_list = ["same" for i in range(num_conv_layers)]
        else:
            if not isinstance(padding_list, list):
                raise TypeError(
                    (
                        f"`padding_list` must be a list or "
                        f"an integer, not {type(padding_list)}!"
                    )
                )
            if len(padding_list) != num_conv_layers:
                raise ValueError(
                    f"`padding_list` must have {num_conv_layers} "
                    f"elements, not {len(padding_list)}!"
                )
            self._padding_list = padding_list

        if isinstance(pooling_kernel_size_list, int):
            pooling_kernel_size_list = [
                pooling_kernel_size_list for i in range(num_conv_layers)
            ]
        else:
            if not isinstance(pooling_kernel_size_list, list):
                raise TypeError(
                    (
                        "`pooling_kernel_size_list` must be a list or "
                        f"an integer, not {type(pooling_kernel_size_list)}!"
                    )
                )
            if len(pooling_kernel_size_list) != num_conv_layers:
                raise ValueError(
                    (
                        f"`pooling_kernel_size_list` must have "
                        f"{num_conv_layers} elements, not "
                        f"{len(pooling_kernel_size_list)}!"
                    )
                )

        if isinstance(pooling_stride_list, int):
            pooling_stride_list = [
                pooling_stride_list for i in range(num_conv_layers)
            ]
        else:
            if not isinstance(pooling_stride_list, list):
                raise TypeError(
                    (
                        "`pooling_stride_list` must be a list or an integer, "
                        f"not {type(pooling_stride_list)}!"
                    )
                )
            if len(pooling_stride_list) != num_conv_layers:
                raise ValueError(
                    (
                        f"`pooling_stride_list` must have {num_conv_layers} "
                        f"elements, not {len(pooling_stride_list)}!"
                    )
                )

        if isinstance(norm_list, bool):
            self._norm_list = [norm_list for i in range(num_conv_layers)]
        else:
            if not isinstance(norm_list, list):
                raise TypeError(
                    (
                        "`norm_list` must be a list or a boolean, "
                        f"not {type(norm_list)}!"
                    )
                )
            if len(norm_list) != num_conv_layers:
                raise ValueError(
                    (
                        f"`norm_list` must have {num_conv_layers} "
                        f"elements, not {len(norm_list)}!"
                    )
                )
            self._norm_list = norm_list

        if norm_type.lower() == "instance":
            norm_class = torch.nn.InstanceNorm3d
            if input_norm:
                self.input_normal = torch.nn.InstanceNorm3d(num_input_features)
        elif norm_type.lower() == "batch":
            norm_class = torch.nn.BatchNorm3d
            if input_norm:
                # No momentum or learnable parameters for input normalization,
                # just use the average
                self.input_normal = torch.nn.BatchNorm3d(
                    num_input_features, momentum=None, affine=False
                )
        else:
            raise ValueError(
                (
                    "`norm_type` has to be 'instance' or "
                    f"'batch, not '{norm_type}'!"
                )
            )

        # Initialize layers
        self.conv = torch.nn.ModuleList()
        self.pool = torch.nn.ModuleList()
        self.input_norm = input_norm

        self.normal = torch.nn.ModuleList()
        dimensions: List[int] = [
            num_input_features,
            10,
            10,
            60,
        ]  # (nb_features per pixel, height, width, depth)
        for i in range(num_conv_layers):
            self.conv.append(
                torch.nn.Conv3d(
                    dimensions[0],
                    conv_filters_list[i],
                    kernel_size=kernel_size_list[i],
                    padding=self._padding_list[i],
                )
            )
            dimensions = self._calc_output_dimension(
                dimensions,
                conv_filters_list[i],
                kernel_size_list[i],
                self._padding_list[i],
            )
            if pooling_type_list[i] is None or pooling_type_list[i] == "None":
                self.pool.append(None)
            elif pooling_type_list[i] == "Avg":
                self.pool.append(
                    torch.nn.AvgPool3d(
                        kernel_size=pooling_kernel_size_list[i],
                        stride=pooling_stride_list[i],
                    )
                )
                dimensions = self._calc_output_dimension(
                    dimensions,
                    out_channels=dimensions[
                        0
                    ],  # same out channels as input channels for pooling
                    kernel_size=pooling_kernel_size_list[i],
                    stride=pooling_stride_list[i],
                )
            elif pooling_type_list[i] == "Max":
                self.pool.append(
                    torch.nn.MaxPool3d(
                        kernel_size=pooling_kernel_size_list[i],
                        stride=pooling_stride_list[i],
                    )
                )
                dimensions = self._calc_output_dimension(
                    dimensions,
                    out_channels=dimensions[
                        0
                    ],  # same out channels as input channels for pooling
                    kernel_size=pooling_kernel_size_list[i],
                    stride=pooling_stride_list[i],
                )
            else:
                raise ValueError(
                    "Pooling type must be 'None', 'Avg' or 'Max'!"
                )
            if self._norm_list[i]:
                self.normal.append(norm_class(dimensions[0]))
            else:
                self.normal.append(None)

        latent_dim = (
            dimensions[0] * dimensions[1] * dimensions[2] * dimensions[3]
        )

        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(latent_dim, num_fc_neurons)
        self.fc2 = torch.nn.Linear(num_fc_neurons, out_put_dim)

    def _calc_output_dimension(
        self,
        dimensions: List[int],
        out_channels: int,
        kernel_size: Union[None, int, List[int]],
        padding: Union[str, int, List[int]] = 0,
        stride: Union[None, int, List[int]] = 1,
    ) -> List[int]:
        """Calculate the output dimension after a CNN layers.

        Works for Conv3D, MaxPool3D and AvgPool3D layers.

        Args:
            dimensions (Tuple[int]): Current dimensions of the input tensor.
                (C,H,W,D) where C is the number of channels,
                H is the height, W is the width and D is the depth.
            out_channels (int): Number of output channels.
            kernel_size (Union[int,List[int]]): Size of the kernel.
                If an integer is provided, it will be used for all dimensions.
            padding (Union[int,List[int]]): Padding size.
                If an integer is provided, it will be used for all dimensions.
                If 'Same', the padding will be calculated to keep the
                output size the same as the input size.
                Defaults to 0.
            stride (Union[int,List[int]]): Stride size.
                If an integer is provided, it will be used for all dimensions.
                Defaults to 1.

        Returns:
            Tuple[int]: New dimensions after the layer.

        NOTE: For the pooling layers, set out_channels equal to the
            input channels. Since they do not change the number of channels.
        """
        krnl_sz: int
        if isinstance(padding, str):
            if not padding.lower() == "same":
                raise ValueError(
                    f"`padding` must be 'Same' or an integer, not {padding}!"
                )
            dimensions[0] = out_channels
        else:
            for i in range(1, 4):
                if isinstance(kernel_size, list):
                    krnl_sz = kernel_size[i - 1]
                else:
                    assert isinstance(kernel_size, int)
                    krnl_sz = kernel_size
                if isinstance(padding, list):
                    pad = padding[i - 1]
                else:
                    pad = padding
                if isinstance(stride, list):
                    strd = stride[i - 1]
                else:
                    assert isinstance(stride, int)
                    strd = stride
                dimensions[i] = (dimensions[i] + 2 * pad - krnl_sz) // strd + 1

        return dimensions

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the LCSC."""
        assert len(data.x) == 1, "Only Main Array image is supported for LCSC"
        x = data.x[0]
        if self.input_norm:
            x = self.input_normal(x)
        for i in range(len(self.conv)):
            x = self.conv[i](x)
            if self.pool[i] is not None:
                x = self.pool[i](x)
            x = torch.nn.functional.elu(x)
            if self.normal[i] is not None:
                x = self.normal[i](x)

        x = self.flatten(x)
        x = torch.nn.functional.elu(self.fc1(x))
        x = torch.nn.functional.elu(self.fc2(x))
        return x
