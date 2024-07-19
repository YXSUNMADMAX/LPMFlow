import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath
from models.encoder.layers.attn import Attention

class SpatialAttention_heat(nn.Module):
    def __init__(self):
        super(SpatialAttention_heat, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.weight = nn.Parameter(torch.ones(1, 1, 768), requires_grad=True)

    def forward(self, attm, x):
        attm = self.sigmoid(attm)
        x1 = x * self.weight * attm
        x = x + x1
        return x
class Block_nmask(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.spatial_attention = SpatialAttention_heat()

    def forward(self, x, mask=None):
        tmp, attn = self.attn(self.norm1(x), mask, return_attention=True)
        x = x + self.drop_path(tmp)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, torch.mean(attn, dim=1)
