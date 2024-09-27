import math
from functools import partial, partialmethod
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import get_conv_layer, Bottleneck, ResNet

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer

def partialclass(cls, *args, **kwargs):

    class PartialClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return PartialClass

def get_inplanes():
    return [128, 256, 512, 1024]

class ResNeXtBottleneck(Bottleneck):
    expansion = 2

    def __init__(self, 
                 in_planes: int,
                 planes: int,
                 cardinality: int = 32,
                 spatial_dims: int = 3,
                 stride: Union[Sequence[int], int] = 1,
                 norm_name: Union[Tuple, str] = Norm.INSTANCE,
                 dropout: Optional[Union[Tuple, str, float]] = None,
                 downsample: Optional[nn.Module] = None,
                 ):
        super().__init__(in_planes, planes, spatial_dims, stride, norm_name, dropout, downsample)

        mid_planes = cardinality * planes // 32
        self.conv1 = get_conv_layer(spatial_dims, in_planes, mid_planes, kernel_size=1, stride=1, dropout=dropout, conv_only=True,)
        self.gn1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=mid_planes)
        self.conv2 = get_conv_layer(spatial_dims, mid_planes, mid_planes, kernel_size=3, stride=stride, dropout=dropout, groups=cardinality, conv_only=True,)
        self.gn2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=mid_planes)
        self.conv3 = get_conv_layer(spatial_dims, mid_planes, planes * self.expansion, kernel_size=1, stride=1, dropout=dropout, conv_only=True,)


class ResNeXt(ResNet):

    def __init__(self,
                block: Union[nn.Module, partial],
                layers: Sequence[int],
                block_inplanes: Sequence[int],
                shortcut_type: str = 'B',
                n_input_channels: int = 1,
                conv1_t_size: int = 7,
                conv1_t_stride: int = 2,
                no_max_pool: bool = True,
                cardinality: int = 32,
                width_factor: float = 1.0,
                spatial_dims: int = 3,
                norm_name: Union[Tuple, str] = Norm.INSTANCE,):
        block = partialclass(block, cardinality=cardinality)
        super().__init__(block, layers, block_inplanes, shortcut_type, 
                        n_input_channels, conv1_t_size, conv1_t_stride, no_max_pool, width_factor, spatial_dims, norm_name)
            
def generate_model(model_depth, **kwargs):
    assert model_depth in [50, 101, 152, 200]

    if model_depth == 50:
        model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], get_inplanes(),
                        **kwargs)
    elif model_depth == 101:
        model = ResNeXt(ResNeXtBottleneck, [8, 9, 13, 3], get_inplanes(),
                        **kwargs)
    elif model_depth == 152:
        model = ResNeXt(ResNeXtBottleneck, [8, 9, 30, 3], get_inplanes(),
                        **kwargs)
    elif model_depth == 200:
        model = ResNeXt(ResNeXtBottleneck, [8, 25, 30, 3], get_inplanes(),
                        **kwargs)

    return model        
        