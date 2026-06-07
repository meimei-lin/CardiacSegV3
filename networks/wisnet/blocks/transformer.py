import torch
import torch.nn as nn
from monai.networks.blocks import TransformerBlock

class Transformer(nn.Module):
    """
    3D Bottleneck Transformer using MONAI's TransformerBlock.
    Clean version: No DropPath, using MONAI's internal residuals.
    """
    def __init__(self, in_channels, mlp_ratio=4.0, num_heads=8, dropout=0.0):
        super().__init__()
        
        # 參數檢查與設定
        hidden_size = in_channels
        mlp_dim = int(in_channels * mlp_ratio)
        
        # MONAI TransformerBlock
        # 內部結構包含: Norm -> Attention -> Residual -> Norm -> MLP -> Residual
        self.transformer = TransformerBlock(
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            dropout_rate=dropout, # 使用標準 Dropout 即可
            qkv_bias=True, 
            save_attn=False
        )

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape

        # 1. Flatten: (B, C, D, H, W) -> (B, Tokens, C)
        # 必須轉置為 (Batch, Tokens, Channels) 以符合 MONAI 輸入格式
        x_flat = x.flatten(2).transpose(1, 2)

        # 2. Transformer Forward
        # 輸出 shape: (B, Tokens, C)
        x_out_flat = self.transformer(x_flat)

        # 3. Reshape back: (B, Tokens, C) -> (B, C, D, H, W)
        x_out = x_out_flat.transpose(1, 2).view(B, C, D, H, W)
        
        return x_out