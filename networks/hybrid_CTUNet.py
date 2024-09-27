# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import nn, einsum 

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.layers.utils import get_act_layer, get_norm_layer 

from .vit import ViT 
from .resnet import generate_model as resnet
from .resnet import get_conv_layer 

from einops.layers.torch import Rearrange 
from einops import rearrange, repeat

class ResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        dropout: dropout probability

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super(ResBlock, self).__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            conv_only=True,
        )
        self.conv3 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.lrelu = get_act_layer(("leakyrelu", {"inplace": True, "negative_slope": 0.01}))
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample:
            residual = self.conv3(residual)
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out
    
class BasicConvBlock(nn.Module):
    """
    A CNN module that can be used for UNETR, based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()

        self.layer = ResBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm_name=norm_name,
        )

    def forward(self, inp):
        return self.layer(inp)
    
class UpCatConvBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super(UpCatConvBlock, self).__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.conv_block = ResBlock(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out
    
class UpConvBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super(UpConvBlock, self).__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.conv_block = ResBlock(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
        )

    def forward(self, inp):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = self.conv_block(out)
        return out
    
class Up_2Fusion_Block(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super(Up_2Fusion_Block, self).__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.pixelweight_attention1 = pixelweight_attention(out_channels)
        self.pixelweight_attention2 = pixelweight_attention(out_channels)

        self.up_addconv_block1 = ResBlock(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
        )
        self.up_addconv_block2 = ResBlock(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
        )
    
    def forward_(self, inp, skip_conv=None, skip_vit=None): 
        # number of channels for skip should equals to out_channels
        # fusion1
        out = self.transp_conv(inp) 

        out = self.pixelweight_attention1(out, skip_conv) 
        out = self.up_addconv_block1(out) 

        if skip_vit is not None: 
            out = self.pixelweight_attention2(out, skip_vit) 
            out = self.up_addconv_block2(out) 
        return out 
    
    def forward(self, inp, skip_conv=None, skip_vit=None):
        # number of channels for skip should equals to out_channels 
        # fusion2
        if skip_vit is not None: 
            skip = self.pixelweight_attention1(skip_conv, skip_vit) 
            skip = self.up_addconv_block1(skip) 

        out = self.transp_conv(inp) 

        out = self.pixelweight_attention2(out, skip)
        out = self.up_addconv_block2(out)

        return out 
    
class PixelweightConvBlock(nn.Module): 
    """
    A projection upsampling module for UNETR which are completely free of convolutions.
    pixel shuffle following nn.Linear and FeedForward.
    FeedForward following nn.Linear with out_channels dimension. 
    """
    def __init__(
        self, 
        spatial_dims: int, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
    ) -> None:
        super().__init__() 
        
        self.downsample = in_channels != out_channels # In downsample, the kernel_size of convolution is 1.
        self.conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=1, # In downsample, the kernel_size of convolution is 1.
            stride=1,
            conv_only=True,
        )
        self.norm = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

        self.pixelweight_attention = pixelweight_attention(out_channels)

        self.conv_block = ResBlock(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
        )
    
    def forward(self, x, skip): 
        if self.downsample: 
            skip = self.norm(self.conv(skip))
        out = self.pixelweight_attention(x, skip) 
        out = self.conv_block(out)
        return out
    
class PixelShuffle(nn.Module): 
    """
    Up-sampling using pixel shuffle.
    """
    def __init__(
        self, 
        spatial_dims: int,
        scale_factor: Union[Sequence[int], int],
        in_channels: int, 
        out_channels: int, 
    ):
        super().__init__()
        self.spatial_dims = spatial_dims 
        self.scale_factor = scale_factor 
        self.to_out = nn.Linear(in_channels // (scale_factor[0] * scale_factor[1] * scale_factor[2]), out_channels)

    def forward(self, x): 
        dim, factor = self.spatial_dims, self.scale_factor
        input_size = list(x.size()) 
        batch_size, channels = input_size[:2] 
        scale_divisor = factor[0] * factor[1] * factor[2]

        if channels % scale_divisor != 0: 
            raise ValueError(
                f"Number of input channels ({channels}) must be evenly"
                f"divisibel by scale_factor ** dimensions ({factor}**{dim}={scale_divisor})."
            )
    
        org_channels = int(channels // scale_divisor) 
        output_size = [batch_size, org_channels] + [d * factor for d in input_size[2:]]
        output_size = [batch_size, org_channels] + [d * k for d, k in zip(input_size[2:], factor)]

        indices = list(range(2, 2 + 2 * dim)) 
        indices = indices[dim:] + indices[:dim] 
        permute_indices = [0, 1]
        for idx in range(dim): 
            permute_indices.extend(indices[idx::dim]) # [0, 1, 5, 2, 6, 3, 7, 4] 
        x = x.reshape([batch_size, org_channels] + list(factor) + list(input_size[2:]))
        x = x.permute(permute_indices).reshape(output_size) 

        x = rearrange(x, 'b c h w f -> b h w f c')
        x = self.to_out(x) 
        x = rearrange(x, 'b h w f c -> b c h w f')

        return x 
    
class Residual(nn.Module): 
    def __init__(self, fn): 
        super().__init__() 
        self.fn = fn 
    
    def forward(self, x): 
        return self.fn(x) + x 
    
class MultiAxisAttention(nn.Module): 
    def __init__(
        self,
        dim, 
        dim_head = 32, 
        dropout = 0., 
        window_size = 7
    ):
        super().__init__() 
        assert (dim % dim_head) == 0, 'dimension must be divisible by the head dimension' 

        self.heads = dim // dim_head 
        self.scale = dim_head ** -0.5 

        self.norm = nn.LayerNorm(dim) 
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False) 

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1), 
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False), 
            nn.Dropout(dropout)
        )

        # relative positional bias 
        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 3, self.heads) 

        pos = torch.arange(window_size) 
        grid = torch.stack(torch.meshgrid(pos, pos, pos, indexing = 'ij')) 
        grid = rearrange(grid, 'c h w f -> (h w f) c') 
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...') 
        rel_pos += window_size - 1 
        rel_pos_indices = (rel_pos * torch.tensor([(2 * window_size -1)**2, 2 * window_size -1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x): 
        batch, height, width, frame, window_height, window_width, window_frame, _, device, h = *x.shape, x.device, self.heads 

        x = self.norm(x) 

        # flatten 
        x = rearrange(x, 'b h w f h1 w1 f1 d -> (b h w f) (h1 w1 f1) d')

        # project for querie, keys, values 
        q, k, v = self.to_qkv(x).chunk(3, dim = -1) 
        # split heads 
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v)) 
        # scale 
        q = q * self.scale 
        # sim 
        sim = einsum('b h i d, b h j d -> b h i j', q, k) 

        # add positional bias 
        bias = self.rel_pos_bias(self.rel_pos_indices) 
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention 
        attn = self.attend(sim) 
        # aggregate 
        out = einsum('b h i j, b h j d -> b h i d', attn, v) 

        # merge heads 
        out = rearrange(out, 'b h (h1 w1 f1) d -> b h1 w1 f1 (h d)', f1 = window_frame, h1 = window_height, w1 = window_width)
        # combine heads out 
        out = self.to_out(out)
        return rearrange(out, '(b h w f) ... -> b h w f ...', h = height, w = width, f = frame)
    
class FeedForward(nn.Module): 
    def __init__(self, dim, mult = 4, dropout = 0.): 
        super().__init__() 
        inner_dim = int(dim * mult) 
        self.net = nn.Sequential(
            nn.LayerNorm(dim), 
            nn.Linear(dim, inner_dim), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(inner_dim, dim), 
            nn.Dropout(dropout)
        )
    def forward(self, x): 
        return self.net(x) 
    
class UpAttentionBlock(nn.Module): 
    """
    A projection upsampling module for UNETR, which are completely free of convolutions. 
    Up-sampling using pixel shuffle, following nn.Linear, MultiAxisAttention and FeedForward.
    """ 
    def __init__(
        self, 
        spatial_dims: int, 
        in_channels: int, 
        dims: tuple = (512, 256, 128, 64), 
        DS_stride: tuple = ((2,2,1), (2,2,2), (2,2,2), (2,2,2)),
        depth: tuple = (1, 1, 1, 1), 
        dropout: float = 0.0, 
    ):
        super().__init__() 

        # dims = (in_channels, *(512, 256, 128)) 
        # dims = (in_channels, *(512, 256, 128, 64)) 
        dims = (in_channels, *dims[::-1][1:], 64)
        dim_pairs = tuple(zip(dims[:-1], dims[1:])) 

        self.layers = nn.ModuleList([]) 

        window_size = 6
        w = window_size

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)): 
            blocks = [] 
            if ind <= 2: 
                for _ in range(layer_depth): 
                    block = nn.Sequential(
                        Rearrange('b c (h h1) (w w1) (f f1) -> b h w f h1 w1 f1 c', h1 = w, w1 = w, f1 = w),
                        Residual(MultiAxisAttention(dim = layer_dim_in, dim_head=32, dropout = dropout, window_size = w)), 
                        Residual(FeedForward(layer_dim_in, dropout=dropout)),
                        Rearrange('b h w f h1 w1 f1 c -> b c (h h1) (w w1) (f f1)'),

                        Rearrange('b c (h1 h) (w1 w) (f1 f) -> b h w f h1 w1 f1 c', h1 = w, w1 = w, f1 = w), 
                        Residual(MultiAxisAttention(dim = layer_dim_in, dim_head=32, dropout = dropout, window_size = w)), 
                        Residual(FeedForward(layer_dim_in, dropout=dropout)),
                        Rearrange('b h w f h1 w1 f1 c -> b c (h1 h) (w1 w) (f1 f)'), 
                        PixelShuffle(spatial_dims, DS_stride[::-1][ind], layer_dim_in, layer_dim), 
                    ) 
                    blocks.append(block)
            else: 
                for _ in range(layer_depth): 
                    block = nn.Sequential(
                        Rearrange('b c h w f -> b h w f c'), 
                        Residual(FeedForward(layer_dim_in, dropout = dropout)), 
                        Residual(FeedForward(layer_dim_in, dropout=dropout)),
                        Rearrange('b h w f c -> b c h w f'),
                        PixelShuffle(spatial_dims, DS_stride[::-1][ind], layer_dim_in, layer_dim), 
                    ) 

                    blocks.append(block)

            self.layers.append(nn.Sequential(*blocks)) 

    def forward(self, x): 
        features = [] 
        features.append(x) 
        for stage in self.layers:
            x = stage(x) 
            features.append(x)
        return features
    
class CatConvBlock(nn.Module): 
    """
    A projection upsampling module for UNETR which are completely free of convolutions.
    pixel shuffle following nn.Linear and FeedForward.
    FeedForward following nn.Linear with out_channels dimension. 
    """
    def __init__(
        self, 
        spatial_dims: int, 
        in_channels: int, 
        kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
    ) -> None:
        super().__init__() 

        self.conv_block = ResBlock(
            spatial_dims,
            in_channels + in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
        )
    
    def forward(self, x, skip): 
        out = torch.cat((x, skip), dim=1)
        out = self.conv_block(out)
        return out
    
class pixelweight_attention(nn.Module):
    def __init__(self, dim, dim_head = 32, dropout = 0.0):
        super().__init__() 

        self.dim_head = dim_head
        self.heads = dim // dim_head 
        self.scale = dim_head ** -0.5 

        self.norm1 = nn.LayerNorm(dim) 
        self.norm2 = nn.LayerNorm(dim)
        self.to_qkv1 = nn.Linear(dim, dim * 3, bias = False) 
        self.to_qkv2 = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1), 
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False), 
            nn.Dropout(dropout)
        )

    def forward(self, x1, x2): 
        b, c, f, h, w = x1.size()

        x1 = rearrange(x1, 'b c f h w -> b (f h w) c')
        x1 = self.norm1(x1) 
        qkv1 = self.to_qkv1(x1).chunk(3, dim = -1) 
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv1)

        x2 = rearrange(x2, 'b c f h w -> b (f h w) c') 
        x2 = self.norm2(x2) 
        qkv2 = self.to_qkv2(x2).chunk(3, dim = -1) 
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv2) 

        dots1 = torch.sum(q2 * k1, dim=-1).unsqueeze(-1) * self.scale 
        dots2 = torch.sum(q1 * k2, dim=-1).unsqueeze(-1) * self.scale 
        dots = torch.cat((dots1, dots2), dim=-1) 
        attn = self.attend(dots) 
        attn1 = attn[:, :, :, 0].unsqueeze(-1).repeat(1, 1, 1, self.dim_head) 
        attn2 = attn[:, :, :, 1].unsqueeze(-1).repeat(1, 1, 1, self.dim_head) 

        out = rearrange((attn1 * v1 + attn2 * v2), 'b h n d -> b n (h d)') 
        out = self.to_out(out)
        out = rearrange(out, 'b (f h w) c -> b c f h w', f = f, h = h, w = w)

        return out
    
class DecoderLinear(nn.Module): 
    def __init__(self, n_cls, patch_size, d_encoder): 
        super().__init__() 

        self.d_encoder = d_encoder 
        self.patch_size = patch_size 
        self.n_cls = n_cls 

        self.head = nn.Linear(self.d_encoder, n_cls) 

    @torch.jit.ignore
    def no_weight_decay(self): 
        return set() 
    
    def forward(self, x, im_size): 
        F, H, W = im_size 
        GS = (F // self.patch_size, H // self.patch_size, W // self.patch_size) 
        x = self.head(x) 
        x = rearrange(x, "b (f h w) c -> b c f h w", f = GS[0], h = GS[1], w = GS[2])

        return x 


class CTUNet(nn.Module): 
    """
    UNETR based on resnet and vit.
    """
    def __init__(
        self, 
        in_channels: int, 
        dim_conv_stem: int, 
        out_channels: int, 
        model_depth: int, # for resnet 
        img_size: Tuple[int, int],
        frames: int, 
        patch_frame: int, 
        hidden_size: int = 768, 
        num_depths: int = 12, 
        mlp_dim: int = 3072, 
        num_heads: int = 12, 
        norm_name: Union[Tuple, str] = "instance", 
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:

        """
        super().__init__() 

        self.patch_size = (16, 16, patch_frame) 
        self.feat_size = (
            img_size[0] // self.patch_size[0], 
            img_size[1] // self.patch_size[1], 
            frames // self.patch_size[2]
        )
        self.hidden_size = hidden_size 
        dims = [int(4 * item) for item in [32, 64, 128, 256]]
        DS_stride = ((2,2,1), (2,2,2), (2,2,2), (2,2,2))

        # the backbone of encoder 
        self.convnet = resnet(model_depth, DS_stride=DS_stride) 
        self.vit = ViT(
            image_size = img_size, 
            image_patch_size = 16, 
            frames = frames, 
            frame_patch_size = patch_frame, 
            dim = hidden_size, 
            depth = num_depths, 
            heads = num_heads, 
            mlp_dim = mlp_dim, 
            dropout = dropout_rate, 
            emb_dropout = dropout_rate,
            drop_path = dropout_rate, 
        )
        
        # the decoder of resnet 
        self.res_decoder3 = Up_2Fusion_Block(
            spatial_dims = 3, 
            in_channels = dims[3], 
            out_channels = dims[2],
            kernel_size = 3, 
            upsample_kernel_size = DS_stride[3], 
            norm_name = norm_name, 
        )
        self.res_decoder2 = Up_2Fusion_Block(
            spatial_dims = 3, 
            in_channels = dims[2], 
            out_channels = dims[1],
            kernel_size = 3, 
            upsample_kernel_size = DS_stride[2], 
            norm_name = norm_name, 
        )
        self.res_decoder1 = Up_2Fusion_Block(
            spatial_dims = 3, 
            in_channels = dims[1], 
            out_channels = dims[0],
            kernel_size = 3, 
            upsample_kernel_size = DS_stride[1], 
            norm_name = norm_name, 
        )
        self.res_decoder0 = UpConvBlock(
            spatial_dims = 3, 
            in_channels = dims[0], 
            out_channels = 64, # dim_conv_stem
            kernel_size = 3, 
            upsample_kernel_size = DS_stride[0], 
            norm_name = norm_name,
        )

        # self.res_out = UnetOutBlock(spatial_dims=3, in_channels=dims[0], out_channels=out_channels)
        self.res_out = UnetOutBlock(spatial_dims=3, in_channels=64, out_channels=out_channels)
        self.res_out_48x48 = UnetOutBlock(spatial_dims=3, in_channels=dims[0], out_channels=out_channels)
        self.res_out_24x24 = UnetOutBlock(spatial_dims=3, in_channels=dims[1], out_channels=out_channels)

        # the decoder of vit 
        self.vit_encoder0 = BasicConvBlock(
            spatial_dims = 3, 
            in_channels = in_channels, 
            out_channels = dim_conv_stem, 
            kernel_size = 3, 
            stride = 1, 
            norm_name = norm_name,
        )
        self.vit_encoder = UpAttentionBlock(
            spatial_dims = 3, 
            in_channels = hidden_size, 
            dims = dims,
            DS_stride = DS_stride, 
            depth = (1, 1, 1, 1),
            dropout = dropout_rate, 
        )
        self.vit_decoder0 = CatConvBlock(
            spatial_dims = 3, 
            in_channels = dim_conv_stem, 
            kernel_size = 3, 
            norm_name = norm_name,
        )

        self.decoder_linear_96x96 = DecoderLinear(out_channels, 1, 64)
        self.vit_out = UnetOutBlock(spatial_dims=3, in_channels=dim_conv_stem, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x
    
    def forward(self, x_in): # with feature fusion(wFF) 
        # vit output with the encoder0
        F, H, W = x_in.size(2), x_in.size(3), x_in.size(4)

        vit_features = self.vit(x_in) 
        vit_enc0 = self.vit_encoder0(x_in) 

        vit_enc = self.vit_encoder(self.proj_feat(vit_features, self. hidden_size, self.feat_size))
        vit_enc_6x6 = vit_enc[0]
        vit_enc_12x12 = vit_enc[1]
        vit_enc_24x24 = vit_enc[2]
        vit_enc_48x48 = vit_enc[3]
        vit_enc_96x96 = vit_enc[4]

        vit_out = self.vit_decoder0(vit_enc_96x96, vit_enc0)
        vit_logits = self.vit_out(vit_out) # output 2 

        vit_enc_96x96 = rearrange(vit_enc_96x96, 'b c f h w -> b (f h w) c')
        vit_96x96 = self.decoder_linear_96x96(vit_enc_96x96, (F, H, W))  # output 3

        # resnet output without the encoder0 
        res_features = self.convnet(x_in) 
        res_enc1 = res_features[0]
        res_enc2 = res_features[1]
        res_enc3 = res_features[2]
        res_enc4 = res_features[3]

        # res_dec4 = self.res_decoder4(res_enc4, vit_enc_6x6)
        res_dec4 = res_enc4
        res_dec3 = self.res_decoder3(res_dec4, res_enc3, vit_enc_12x12)
        res_dec2 = self.res_decoder2(res_dec3, res_enc2, vit_enc_24x24) 
        res_dec1 = self.res_decoder1(res_dec2, res_enc1, vit_enc_48x48)

        res_out = self.res_decoder0(res_dec1)

        # res_logits = self.res_out(res_out) # output1 
        res_logits = self.res_out(res_out) # output1 
        res_logits_48x48 = self.res_out_48x48(res_dec1) 
        res_logits_24x24 = self.res_out_24x24(res_dec2)

        return ((res_logits, res_logits_48x48, res_logits_24x24), (vit_logits, vit_96x96))
    
class CUNet(nn.Module): 
    """
    UNETR based on resnet and vit.
    """
    def __init__(
        self, 
        out_channels: int, 
        model_depth: int, # for resnet 
        norm_name: Union[Tuple, str] = "instance", 
    ) -> None:
        """
        Args:

        """
        super().__init__() 

        dims = [int(4 * item) for item in [32, 64, 128, 256]]
        DS_stride = ((2,2,1), (2,2,2), (2,2,2), (2,2,2))

        # the backbone of encoder 
        self.convnet = resnet(model_depth, DS_stride = DS_stride) 
        
        # the decoder of resnet 
        self.res_decoder3 = UpCatConvBlock(
            spatial_dims = 3, 
            in_channels = dims[3], 
            out_channels = dims[2],
            kernel_size = 3, 
            upsample_kernel_size = DS_stride[3], 
            norm_name = norm_name, 
        )
        self.res_decoder2 = UpCatConvBlock(
            spatial_dims = 3, 
            in_channels = dims[2], 
            out_channels = dims[1],
            kernel_size = 3, 
            upsample_kernel_size = DS_stride[2], 
            norm_name = norm_name, 
        )
        self.res_decoder1 = UpCatConvBlock(
            spatial_dims = 3, 
            in_channels = dims[1], 
            out_channels = dims[0],
            kernel_size = 3, 
            upsample_kernel_size = DS_stride[1], 
            norm_name = norm_name, 
        )
        self.res_decoder0 = UpConvBlock(
            spatial_dims = 3, 
            in_channels = dims[0], 
            out_channels = 64, # dim_conv_stem
            kernel_size = 3, 
            upsample_kernel_size = DS_stride[0], 
            norm_name = norm_name, 
        )

        self.res_out = UnetOutBlock(spatial_dims=3, in_channels=64, out_channels=out_channels)
        self.res_out_48x48 = UnetOutBlock(spatial_dims=3, in_channels=dims[0], out_channels=out_channels)
        self.res_out_24x24 = UnetOutBlock(spatial_dims=3, in_channels=dims[1], out_channels=out_channels)

    def forward(self, x_in): # without feature fusion(woFF) 
        # resnet output without the encoder0 
        res_features = self.convnet(x_in) 
        res_enc1 = res_features[0]
        res_enc2 = res_features[1]
        res_enc3 = res_features[2]
        res_enc4 = res_features[3] # 6x6 dim[3]

        res_dec3 = self.res_decoder3(res_enc4, res_enc3) # 12x12 dim[2]
        res_dec2 = self.res_decoder2(res_dec3, res_enc2) # 24x24 dim[1]
        res_dec1 = self.res_decoder1(res_dec2, res_enc1) # 48x48 dim[0]

        res_out = self.res_decoder0(res_dec1) # 96x96 dim[0]

        res_logits = self.res_out(res_out) # output1 
        res_logits_48x48 = self.res_out_48x48(res_dec1) 
        res_logits_24x24 = self.res_out_24x24(res_dec2)

        return (res_logits, res_logits_48x48, res_logits_24x24)

class TUNet(nn.Module): 
    """
    UNETR based on resnet and vit.
    """
    def __init__(
        self, 
        in_channels: int, 
        dim_conv_stem: int, 
        out_channels: int, 
        img_size: Tuple[int, int],
        frames: int, 
        patch_frame: int, 
        hidden_size: int = 768, 
        num_depths: int = 12, 
        mlp_dim: int = 3072, 
        num_heads: int = 12, 
        norm_name: Union[Tuple, str] = "instance", 
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:

        """
        super().__init__() 

        self.patch_size = (16, 16, patch_frame)
        self.feat_size = (
            img_size[0] // self.patch_size[0], 
            img_size[1] // self.patch_size[1], 
            frames // self.patch_size[2]
        )
        self.hidden_size = hidden_size 
        dims = [int(4 * item) for item in [32, 64, 128, 256]]
        DS_stride = ((2,2,1), (2,2,2), (2,2,2), (2,2,2))

        # the backbone of encoder 
        self.vit = ViT(
            image_size = img_size, 
            image_patch_size = 16, 
            frames = frames, 
            frame_patch_size = patch_frame, 
            dim = hidden_size, 
            depth = num_depths, 
            heads = num_heads, 
            mlp_dim = mlp_dim, 
            dropout = dropout_rate, 
            emb_dropout = dropout_rate,
            drop_path = dropout_rate, 
        )

        # the decoder of vit 
        self.vit_encoder0 = BasicConvBlock(
            spatial_dims = 3, 
            in_channels = in_channels, 
            out_channels = dim_conv_stem, 
            kernel_size = 3, 
            stride = 1, 
            norm_name = norm_name,
        )
        self.vit_encoder = UpAttentionBlock(
            spatial_dims = 3, 
            in_channels = hidden_size, 
            dims = dims,
            DS_stride = DS_stride, 
            depth = (1, 1, 1, 1),
            dropout = dropout_rate, 
        )
        self.vit_decoder0 = CatConvBlock(
            spatial_dims = 3, 
            in_channels = dim_conv_stem, 
            kernel_size = 3, 
            norm_name = norm_name,
        )

        self.decoder_linear_96x96 = DecoderLinear(out_channels, 1, 64)
        self.vit_out = UnetOutBlock(spatial_dims=3, in_channels=dim_conv_stem, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in): # without feature fusion(woFF) 
        # vit output with the encoder0
        H, W, F= x_in.size(2), x_in.size(3), x_in.size(4)

        vit_features = self.vit(x_in) 
        vit_enc0 = self.vit_encoder0(x_in) 

        vit_enc = self.vit_encoder(self.proj_feat(vit_features, self.hidden_size, self.feat_size))

        vit_out = self.vit_decoder0(vit_enc[-1], vit_enc0)
        vit_logits = self.vit_out(vit_out) # output 2 

        vit_enc_96x96 = rearrange(vit_enc[::-1][0], 'b c h w f -> b (h w f) c')
        vit_96x96 = self.decoder_linear_96x96(vit_enc_96x96, (H, W, F))  # output 3

        return (vit_logits, vit_96x96)

