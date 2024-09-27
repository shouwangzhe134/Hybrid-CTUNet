import torch
from torch import nn 

from einops import rearrange, repeat
from einops.layers.torch import Rearrange 

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class DropPath(nn.Module): 
    def __init__(self, drop_prob = 0.): 
        super().__init__() 
        self.drop_prob = drop_prob 

    def forward(self, x): 
        batch, drop_prob, device, dtype = x.shape[0], self.drop_prob, x.device, x.dtype 

        if drop_prob <= 0. or not self.training: 
            return x 
        
        keep_prob = 1 - drop_prob 
        shape = (batch, *((1, ) * (x.ndim - 1))) 

        keep_mask = torch.zeros(shape, device = device).float().uniform_(0, 1) < keep_prob 
        output = x.div(keep_prob) * keep_mask.float() 

        return output 

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads 
        project_out = not (heads == 1 and dim_head == dim) 

        self.heads = heads 
        self.scale = dim_head ** -0.5 

        self.norm = nn.LayerNorm(dim) 
        self.attend = nn.Softmax(dim = -1) 
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) 

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), 
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x) 
        qkv = self.to_qkv(x).chunk(3, dim = -1) 
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) 

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale 
        
        attn = self.attend(dots) 
        attn = self.dropout(attn) 

        out = torch.matmul(attn, v) 
        out = rearrange(out, 'b h n d -> b n (h d)') 
        return self.to_out(out)
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0., drop_path = 0.):
        super().__init__()

        self.attn = Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout = dropout)
        self.drop_path = DropPath(dropout) if drop_path > 0. else nn.Identity()
    
    def forward_(self, x):
        x = self.drop_path(self.attn(x)) + x 
        x = self.drop_path(self.ff(x)) + x 
        return x 
    
    def forward(self, x):
        x = self.attn(x) + x 
        x = self.ff(x) + x 
        return x 
    
    

class ViT(nn.Module):
    def __init__(self, image_size, image_patch_size, frames, frame_patch_size, 
                 dim, depth, heads, mlp_dim, channels = 1, dim_head = 64, 
                 dropout = 0., emb_dropout = 0., drop_path = 0.):
        super().__init__()
        image_height, image_width = pair(image_size) 
        patch_height, patch_width = pair(image_patch_size) 

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by the frame patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size 

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (f pf) -> b (h w f) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim), 
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.ModuleList(
            [
                TransformerBlock(dim, heads, dim_head, mlp_dim, dropout, drop_path) for i in range(depth)
            ]
        )

    def forward(self, img):
        x = self.to_patch_embedding(img) 

        x += self.pos_embedding
        x = self.dropout(x) 

        for transformer in self.transformer:
            x = transformer(x)

        return x
    






        



