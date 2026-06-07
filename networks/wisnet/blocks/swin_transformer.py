import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath, trunc_normal_

class WindowAttention3D(nn.Module):
    """ 3D Window-based Self-Attention """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size # (Wd, Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """ 3D Swin Transformer Block """
    def __init__(self, in_channels, embed_dim, patch_size, window_size, mlp_ratio, depths, num_heads, feature_size, spatial_dims=3):
        super().__init__()
        self.dim = in_channels
        self.num_heads = num_heads[0] if isinstance(num_heads, list) else num_heads
        self.window_size = window_size
        
        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = WindowAttention3D(self.dim, window_size, self.num_heads)
        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, int(self.dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(self.dim * mlp_ratio), self.dim)
        )

    def forward(self, x):
        # Input x: (B, C, D, H, W)
        b, c, d, h, w = x.shape
        
        # Flatten for Transformer: (B, D*H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Attention
        shortcut = x_flat
        x_flat = self.norm1(x_flat)
        x_flat = self.attn(x_flat)
        x_flat = shortcut + x_flat
        
        # MLP
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        
        # Reshape back to 3D
        x_out = x_flat.transpose(1, 2).view(b, c, d, h, w)
        return x_out