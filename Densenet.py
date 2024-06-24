# Import necessary modules and packages
from __future__ import annotations
import re
from collections import OrderedDict
from collections.abc import Callable, Sequence
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from monai.networks.layers.factories import Conv, Dropout, Pool
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils.module import look_up_option

# Define exported symbols
__all__ = [
    "DenseNet",
    "DenseNetModel",
    "DenseNet121",
    "create_densenet121",
    "DenseNet121Model",
]

class DenseLayer(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        dropout_probability: float,
        act: str | tuple = ("relu", {"inplace": True}),
        normalization: str | tuple = "batch",
    ) -> None:
        super().__init__()

        out_channels = bn_size * growth_rate
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, spatial_dims]

        self.layers = nn.Sequential()

        self.layers.add_module("norm1", get_norm_layer(name=normalization, spatial_dims=spatial_dims, channels=in_channels))
        self.layers.add_module("relu1", get_act_layer(name=act))
        self.layers.add_module("conv1", conv_type(in_channels, out_channels, kernel_size=1, bias=False))

        self.layers.add_module("norm2", get_norm_layer(name=normalization, spatial_dims=spatial_dims, channels=out_channels))
        self.layers.add_module("relu2", get_act_layer(name=act))
        self.layers.add_module("conv2", conv_type(out_channels, growth_rate, kernel_size=3, padding=1, bias=False))
        
        if dropout_probability > 0:
            self.layers.add_module("dropout", dropout_type(dropout_probability))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layers(x)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        layers: int,
        in_channels: int,
        bn_size: int,
        growth_rate: int,
        dropout_probability: float,
        act: str | tuple = ("relu", {"inplace": True}),
        normalization: str | tuple = "batch",
    ) -> None:
        super().__init__()
        for i in range(layers):
            layer = DenseLayer(spatial_dims, in_channels, growth_rate, bn_size, dropout_probability, act=act, normalization=normalization)
            in_channels += growth_rate
            self.add_module("dense_layer%d" % (i + 1), layer)

class TransitionLayer(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: str | tuple = ("relu", {"inplace": True}),
        normalization: str | tuple = "batch",
    ) -> None:
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        pool_type: Callable = Pool[Pool.AVG, spatial_dims]

        self.add_module("norm", get_norm_layer(name=normalization, spatial_dims=spatial_dims, channels=in_channels))
        self.add_module("relu", get_act_layer(name=act))
        self.add_module("conv", conv_type(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module("pool", pool_type(kernel_size=2, stride=2))

class DenseNetEncoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        bn_size: int = 4,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 32, 16),
        act: str | tuple = ("relu", {"inplace": True}),
        normalization: str | tuple = "batch",
        dropout_probability: float = 0.0,
    ) -> None:
        super().__init__()

        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        pooling_type: type[nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d] = Pool[Pool.MAX, spatial_dims]
        adaptive_avg_pooling_type: type[nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", conv_type(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", get_norm_layer(name=normalization, spatial_dims=spatial_dims, channels=init_features)),
                    ("relu0", get_act_layer(name=act)),
                    ("pool0", pooling_type(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        in_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=in_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_probability=dropout_probability,
                act=act,
                normalization=normalization,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features.add_module(
                    "norm5", get_norm_layer(name=normalization, spatial_dims=spatial_dims, channels=in_channels)
                )
            else:
                _out_channels  = in_channels // 2
                transition = TransitionLayer(
                    spatial_dims, in_channels=in_channels, out_channels=_out_channels , act=act, normalization=normalization
                )
                self.features.add_module(f"transition{i + 1}", transition)
                in_channels = _out_channels 

        # pooling and classification
        self.classifier_layers = nn.Sequential(
            OrderedDict(
                [
                    ("relu", get_act_layer(name=act)),
                    ("pool", adaptive_avg_pooling_type(1)),
                    ("flatten", nn.Flatten(1)),
                    ("out", nn.Linear(in_channels, out_channels)),
                ]
            )
        )

        for module in self.modules():
            if isinstance(module, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(module.weight))
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(torch.as_tensor(module.weight), 1)
                nn.init.constant_(torch.as_tensor(module.bias), 0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(torch.as_tensor(module.bias), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier_layers(x)
        return x


class CustomDenseNet121(DenseNetEncoder):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 32, 16),
        pretrained: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            **kwargs,
        )
        if pretrained:
            if spatial_dims > 2:
                raise NotImplementedError(
                    "Pretrained models for more than two spatial dimensions are not available on PyTorch Hub."
                )

Densenetencoder=DenseNetEncoder
CustomDenseNet121_dis = CustomDenseNet121
