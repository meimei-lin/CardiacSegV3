# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# ... (License header omitted for brevity)

from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from .dynunet_block import get_conv_layer
from .cbam import CBAM
from networks.uwtnet_3d.uwtnet_3d import WTConv3d 

# ==========================================================================
# 1. FSRBlock: 頻率-空間殘差塊 (Frequency-Spatial Residual Block)
# ==========================================================================
class FSRBlock(nn.Module):
    """
    [創新解碼單元] Frequency-Spatial Residual Block
    
    修正重點：
    將 CBAM 移到 Residual 相加「之前」。
    這樣做可以讓 CBAM 專注於過濾特徵，而不會阻斷 Residual (Identity) 的梯度流動。
    這對深層網路的訓練穩定性至關重要。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        norm_name: str,
        wt_levels: int = 1
    ) -> None:
        super().__init__()
        
        # 1. Spatial Stream (標準 ResBlock 路徑)
        self.spatial_conv1 = get_conv_layer(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            conv_only=True,
        )
        self.spatial_conv2 = get_conv_layer(
            spatial_dims=3,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            conv_only=True,
        )
        
        # 2. Frequency Stream (創新路徑 - Injection)
        inter_dim = out_channels // 2
        self.freq_path = nn.Sequential(
            nn.Conv3d(in_channels, inter_dim, 1), # 降維
            WTConv3d(inter_dim, inter_dim, kernel_size=5, wt_levels=wt_levels),
            nn.Conv3d(inter_dim, out_channels, 1) # 升維
        )
        
        # 3. 標準化與激活
        self.lrelu = get_act_layer(name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}))
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=3, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=3, channels=out_channels)
        
        # 4. 下採樣
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = get_conv_layer(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                conv_only=True
            )
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=3, channels=out_channels)

        # 5. Gatekeeper: CBAM
        self.cbam = CBAM(out_channels, reduction=16, kernel_size=7)

    def forward(self, inp):
        residual = inp
        
        # A. 空間路徑
        out_s = self.spatial_conv1(inp)
        out_s = self.norm1(out_s)
        out_s = self.lrelu(out_s)
        out_s = self.spatial_conv2(out_s)
        out_s = self.norm2(out_s)
        
        # B. 頻率路徑 (Injection)
        out_f = self.freq_path(inp)
        
        # C. 融合
        out = out_s + out_f
        
        # [關鍵修正] D. CBAM 過濾 (在 Residual 相加之前！)
        # 我們只過濾 "變化的部分" (Delta)，保留原始資訊 (Identity) 不受干擾
        out = self.cbam(out)
        
        # E. 殘差連接
        if self.downsample is not None:
            residual = self.downsample(residual)
            if hasattr(self, "norm3"):
                residual = self.norm3(residual)
        
        out += residual # 加法在 CBAM 之後
        
        out = self.lrelu(out)
        return out

# ==========================================================================
# 3. UnetrUpBlock: 上採樣模組
# ==========================================================================
class UnetrUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        upsample_kernel_size: int,
        norm_name: str,
        res_block: bool = True,
        use_fsr: bool = False, 
        wt_levels: int = 1
    ) -> None:
        super().__init__()
        
        # 1. 上採樣
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size,
            conv_only=True,
            is_transposed=True,
        )

        # 2. 特徵融合塊 (Conv Block)
        self.use_fsr = use_fsr
        
        if use_fsr:
            # === A. 創新模式: FSR Block ===
            # FSRBlock 內部已有 CBAM，所以外部不需要
            print(f"Using FSR-Block (WT Levels: {wt_levels})")
            self.conv_block = FSRBlock(
                in_channels=out_channels + out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
                wt_levels=wt_levels
            )
            self.use_cbam_external = False 
            
        elif res_block:
            # === B. 傳統模式: ResBlock ===
            # 注意：您的 UnetResBlock 已經內建 CBAM 了！
            # 所以這裡外部也必須設為 False，否則會跑兩次 CBAM (Double Filtering)
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
            self.use_cbam_external = False # [修正] 內部已有，外部不用再加
            
        else:
            # === C. 基礎模式 (無 Res) ===
            # 只有這種情況才需要外部 CBAM
            self.conv_block = UnetBasicBlock( 
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
            self.use_cbam_external = True # BasicBlock 沒有內建，所以要外掛

        # 外部 CBAM (只給 BasicBlock 用)
        if self.use_cbam_external:
            self.cbam = CBAM(out_channels, reduction=16, kernel_size=7)

    def forward(self, inp, skip):
        # 1. 上採樣
        out = self.transp_conv(inp)
        # 2. 拼接
        out = torch.cat((out, skip), dim=1)
        # 3. 融合
        out = self.conv_block(out)
        
        # 4. 外部 CBAM (僅在 BasicBlock 模式下執行)
        if self.use_cbam_external:
            out = self.cbam(out)
            
        return out
    
# ==========================================================================
# 2. UnetResBlock: 標準殘差塊 (含 CBAM)
# ==========================================================================
class UnetResBlock(nn.Module):
    """
    已經內建 CBAM 的 ResBlock (對應 UNetIRC 的架構)
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
    ):
        super().__init__()
        # ... (Conv1, Conv2 定義保持不變) ...
        self.conv1 = get_conv_layer(
            spatial_dims, in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            dropout=dropout, act=None, norm=None, conv_only=False,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
            dropout=dropout, act=None, norm=None, conv_only=False,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        
        # 下採樣邏輯
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims, in_channels, out_channels, kernel_size=1, stride=stride,
                dropout=dropout, act=None, norm=None, conv_only=False,
            )
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
            
        # 內建 CBAM
        self.cbam = CBAM(out_channels, reduction=16, kernel_size=7)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
            
        # CBAM 在 Residual 之前 (正確位置)
        out = self.cbam(out)
        
        out += residual
        out = self.lrelu(out)
        return out

