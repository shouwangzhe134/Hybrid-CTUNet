import math
from functools import partial, partialmethod
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import get_conv_layer
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.layers.factories import Act, Norm

def get_inplanes():
    return [32, 64, 128, 256]

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, 
                 in_planes: int,
                 planes: int,
                 spatial_dims: int = 3,
                 stride: Union[Sequence[int], int] = 1,
                 norm_name: Union[Tuple, str] = Norm.INSTANCE,
                 dropout: Optional[Union[Tuple, str, float]] = None,
                 downsample: Optional[nn.Module] = None,):
        super().__init__()

        self.conv1 = get_conv_layer(3, in_planes, planes, kernel_size=1, stride=1, conv_only=True)
        self.gn1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=planes)

        n_3d_parameters = planes * planes * 3 * 3 * 3
        n_2d1d_parameters = planes * 3 + 3 * planes
        mid_planes = n_3d_parameters // n_2d1d_parameters
        self.conv2_s = get_conv_layer(3, planes, mid_planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), conv_only=True,)
        self.gn2_s = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=mid_planes)
        self.conv2_t = get_conv_layer(3, mid_planes, planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1), conv_only=True)
        self.gn2_t = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=planes)

        self.conv3 = get_conv_layer(3, planes, planes * self.expansion, kernel_size=1, stride=1, conv_only=True)
        self.gn3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=planes * self.expansion)
        self.lrelu = get_act_layer(("leakyrelu", {"inplace": True, "negative_slope": 0.01}))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.lrelu(self.gn1(self.conv1(x)))
        out = self.lrelu(self.gn2_s(self.conv2_s(out)))
        out = self.lrelu(self.gn2_t(self.conv2_t(out)))
        out = self.gn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.lrelu(residual + out)

        return out
    

class ResNet(nn.Module):
    def __init__(self,
                 block: nn.Module,
                 layers: Sequence[int],
                 block_inplanes: Sequence[int],
                 shortcut_type: str = 'B',
                 n_input_channels: int = 1,
                 conv1_t_size: int = 7,
                 conv1_t_stride: int = 1,
                 no_max_pool: bool = True,
                 width_factor: float = 1.0,
                 spatial_dims: int = 3,
                 norm_name: Union[Tuple, str] = Norm.INSTANCE,):
        super().__init__()

        block_inplanes = [int(x * width_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        n_3d_parameters = 1 * self.in_planes * conv1_t_size * 7 * 7
        n_2p1d_parameters = 1 * 7 * 7 + conv1_t_size * self.in_planes
        mid_planes = n_3d_parameters // n_2p1d_parameters
        self.conv1_s = get_conv_layer(3, n_input_channels, mid_planes, kernel_size=(1, 7, 7), stride=(1, 2, 2), conv_only=True,)
        self.gn1_s = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=mid_planes)
        self.conv1_t = get_conv_layer(3, mid_planes, self.in_planes, kernel_size=(conv1_t_size, 1, 1), stride=(conv1_t_stride, 1, 1), conv_only=True)
        self.gn1_t = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=self.in_planes)
        self.lrelu = get_act_layer(("leakyrelu", {"inplace": True, "negative_slope": 0.01}))

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
    
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

        x = self.lrelu(self.gn1_s(self.conv1_s(x)))
        x = self.lrelu(self.gn1_t(self.conv1_t(x)))

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
        # model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
        model = ResNet(Bottleneck, [8, 9, 13, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        # model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
        model = ResNet(Bottleneck, [8, 9, 30, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        # model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)
        model = ResNet(Bottleneck, [8, 25, 30, 3], get_inplanes(), **kwargs)

    return model






