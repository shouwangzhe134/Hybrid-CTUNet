from functools import partial
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer

def get_inplanes():
    return [32, 64, 128, 256]

def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Union[Tuple, str] = Norm.INSTANCE,
    dropout: Optional[Union[Tuple, str, float]] = None,
    groups: int = 1,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        groups=groups,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )

def get_padding(
    kernel_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
) -> Union[Tuple[int, ...], int]:

    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]

def get_output_padding(
    kernel_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
    padding: Union[Sequence[int], int],
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, 
                 in_planes: int,
                 planes: int,
                 spatial_dims: int = 3,
                 stride: Union[Sequence[int], int] = 1,
                 norm_name: Union[Tuple, str] = Norm.INSTANCE,
                 dropout: Optional[Union[Tuple, str, float]] = None,
                 downsample: Optional[nn.Module] = None,
                 ):
        super().__init__()

        self.conv1 = get_conv_layer(spatial_dims, in_planes, planes, kernel_size=1, stride=1, dropout=dropout, conv_only=True)
        self.gn1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=planes)
        self.conv2 = get_conv_layer(spatial_dims, planes, planes, kernel_size=3, stride=stride, dropout=dropout, conv_only=True)
        self.gn2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=planes)
        self.conv3 = get_conv_layer(spatial_dims, planes, planes * self.expansion, kernel_size=1, stride=1, dropout=dropout, conv_only=True)
        self.gn3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=planes * self.expansion)
        self.lrelu = get_act_layer(("leakyrelu", {"inplace": True, "negative_slope": 0.01}))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.lrelu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.lrelu(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.lrelu(out)

        return out

class ResNet(nn.Module):

    def __init__(self,
                 block: nn.Module,
                 layers: Sequence[int],
                 block_inplanes: Sequence[int],
                 shortcut_type: str = 'B',
                 n_input_channels: int = 1,
                 conv1_t_size: int = 7,
                 DS_stride: tuple = ((2,2,1), (2,2,2), (2,2,2), (2,2,2)),
                 no_max_pool: bool = True,
                 width_factor: float = 1.0,
                 spatial_dims: int = 3,
                 norm_name: Union[Tuple, str] = Norm.INSTANCE,):
        super().__init__()

        block_inplanes = [int(x * width_factor) for x in block_inplanes]

        # self.in_planes = block_inplanes[0]
        self.in_planes = 64
        self.no_max_pool = no_max_pool

        self.conv1 = get_conv_layer(spatial_dims, 
                                    n_input_channels,
                                    self.in_planes,
                                    kernel_size=(7, 7, conv1_t_size),
                                    stride=DS_stride[0],
                                    conv_only=True,)
        self.gn1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=self.in_planes)
        self.lrelu = get_act_layer(("leakyrelu", {"inplace": True, "negative_slope": 0.01}))
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=DS_stride[1])
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=DS_stride[2])
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=DS_stride[3])

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    get_conv_layer(3, self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, conv_only=True,),
                    get_norm_layer(name=Norm.INSTANCE, spatial_dims=3, channels=planes * block.expansion)
                )

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        features = []

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.lrelu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return features


def generate_model(model_depth, **kwargs):
    assert model_depth in [50, 101, 152, 200]

    if model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [8, 9, 13, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [8, 9, 30, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [8, 25, 30, 3], get_inplanes(), **kwargs)

    return model
