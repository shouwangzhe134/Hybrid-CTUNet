import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import Optional, Sequence, Tuple, Union

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer

from .resnet import get_conv_layer, get_inplanes, ResNet

class PreActivationBottleneck(nn.Module):
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
        
        self.gn1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=in_planes)
        self.conv1 = get_conv_layer(spatial_dims, in_planes, planes, kernel_size=1, stride=1, dropout=dropout, conv_only=True,)
        self.gn2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=planes)
        self.conv2 = get_conv_layer(spatial_dims, planes, planes, kernel_size=3, stride=stride, dropout=dropout, conv_only=True)
        self.gn3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=planes)
        self.conv3 = get_conv_layer(spatial_dims, planes, planes * self.expansion, kernel_size=1, stride=1, dropout=dropout, conv_only=True,)
        self.lrelu = get_act_layer(("leakyrelu", {"inplace": True, "negative_slope": 0.01}))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.gn1(x)
        out = self.lrelu(out)
        out = self.conv1(out)

        out = self.gn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        out = self.gn3(out)
        out = self.lrelu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out
    
def generate_model(model_depth, **kwargs):
    assert model_depth in [50, 101, 152, 200]

    if model_depth == 50:
        model = ResNet(PreActivationBottleneck, [3, 4, 6, 3], get_inplanes(),
                       **kwargs)
    elif model_depth == 101:
        # model = ResNet(PreActivationBottleneck, [3, 4, 23, 3], get_inplanes(), 
        #                 **kwargs)
        model = ResNet(PreActivationBottleneck, [8, 9, 13, 3], get_inplanes(), 
                       **kwargs)
    elif model_depth == 152:
        # model = ResNet(PreActivationBottleneck, [3, 8, 36, 3], get_inplanes(), 
        #                **kwargs)
        model = ResNet(PreActivationBottleneck, [8, 9, 30, 3], get_inplanes(), 
                       **kwargs)
    elif model_depth == 200:
        # model = ResNet(PreActivationBottleneck, [3, 24, 36, 3], get_inplanes(),
        #                **kwargs)
        model = ResNet(PreActivationBottleneck, [8, 25, 30, 3], get_inplanes(),
                       **kwargs)

    return model