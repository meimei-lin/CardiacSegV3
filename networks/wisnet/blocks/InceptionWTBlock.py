import torch
import torch.nn as nn
from timm.models.layers import DropPath
from .utils import LayerNorm, GRN
from networks.uwtnet_3d.uwtnet_3d import WTConv3d 

class InceptionWTBlock(nn.Module):
    """
    針對心臟 CT 分割 (AOV + Myo) 的三大改良：
    1. Channel Rebalancing: 提升 WTConv 通道佔比至 33% (抓 AOV 細節)。
    2. Spatial Gating: 使用 3x3 卷積生成門控，取代全域池化 (保留 AOV 空間位置)。
    3. Frequency Injection: 將銳化後的邊界注入空間分支 (強化 Myo 邊界感知)。
    """
    def __init__(
        self, 
        dim, 
        kernel_size=11, 
        exp_rate=4, 
        drop_path=0., 
        wt_levels=2, 
        wt_type='db1',
        freq_ratio=1.0 # 控制頻率分支的強度 (淺層設 1.0, 深層設 0.1)
    ):
        super().__init__()
        
        self.freq_ratio = freq_ratio
        
        # ============================================================
        # 1. 通道分配 (Channel Rebalancing)
        # ============================================================
        # 原本是 1/5 (0.2)，現在提升到 1/3 (0.33)
        # 這樣有更多的 Filter 可以去抓主動脈瓣的微細邊界
        gc = int(dim // 3) 
        
        # split_indexes: (Identity, WT, Spatial_HW, Spatial_D)
        # 剩下的通道留給 Identity，確保梯度傳遞順暢
        self.split_indexes = (dim - 3 * gc, gc, gc, gc)

        # ============================================================
        # 2. 頻率分支 (Frequency Branch)
        # ============================================================
        self.wt_branch = WTConv3d(
            in_channels=gc, 
            out_channels=gc, 
            kernel_size=5, 
            wt_levels=wt_levels, 
            wt_type=wt_type
        )
        
        # [核心改良 A] Spatial Gating (空間門控)
        # 取代 AdaptiveAvgPool3d(1)。
        # 使用 Depthwise Conv 3x3 來感知局部資訊，讓 Gate 知道 "哪裡" 是邊界。
        self.spatial_gate = nn.Sequential(
            nn.Conv3d(gc, gc, kernel_size=3, padding=1, groups=gc), # Depthwise
            nn.Conv3d(gc, gc, kernel_size=1), # Pointwise
            nn.Sigmoid()
        )

        # ============================================================
        # 3. 空間分支 (Spatial Branches)
        # ============================================================
        # 根據 kernel_size 動態計算 padding
        k = kernel_size
        p = k // 2 
        
        # Branch 1: H-W Plane (針對切片面的形狀)
        self.dwconv_hw = nn.Conv3d(gc, gc, kernel_size=(1, k, k), padding=(0, p, p), groups=gc)
        
        # Branch 2: D-Axis (針對深度/Z軸的連續性)
        self.dwconv_d = nn.Conv3d(gc, gc, kernel_size=(k, 1, 1), padding=(p, 0, 0), groups=gc)
        
        # ============================================================
        # 4. 通道混合 (Mixing)
        # ============================================================
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, exp_rate * dim)
        self.act = nn.GELU()
        self.grn = GRN(exp_rate * dim)
        self.pwconv2 = nn.Linear(exp_rate * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        # 1. Split: 切成 4 份
        x_id, x_wt, x_hw, x_d = torch.split(x, self.split_indexes, dim=1) 
        
        # ----------------------------------------------------------
        # Step 2: 頻率提取與門控
        # ----------------------------------------------------------
        wt_feat = self.wt_branch(x_wt)
        
        # 計算 Pixel-wise Gate (每個像素都有自己的權重)
        gate_map = self.spatial_gate(wt_feat) 
        
        # 應用門控並調節強度
        out_wt = wt_feat * gate_map * self.freq_ratio
        
        # ----------------------------------------------------------
        # Step 3: [核心改良 B] 頻率注入 (Frequency Injection)
        # ----------------------------------------------------------
        # 將銳化後的邊界特徵 (out_wt) 加到空間分支的輸入 (x_hw)
        # 這樣大核卷積在掃描形狀時，會同時看到 WTConv 找到的邊界
        out_hw = self.dwconv_hw(x_hw + out_wt) 
        
        # 深度軸通常比較模糊，保持獨立處理即可，避免雜訊擴散
        out_d = self.dwconv_d(x_d)
        
        # ----------------------------------------------------------
        # Step 4: 融合與混合
        # ----------------------------------------------------------
        # Concat
        x = torch.cat((x_id, out_wt, out_hw, out_d), dim=1) 
        
        # MLP Mixing (Inverted Bottleneck)
        x = x.permute(0, 2, 3, 4, 1) # Channel Last
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3) # Channel First

        # Residual Connection
        x = input + self.drop_path(x)
        return x