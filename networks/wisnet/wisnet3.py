import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock
from .blocks.inceptionnext_v2 import InceptionNeXtBlock_V2
from .blocks.utils import LayerNorm
from .blocks.cbam import CBAM 
from .blocks.transformer import ViTBlock3D
from networks.uwtnet_3d.uwtnet_3d import WTConv3d 
from .blocks.eca import ECA
from .blocks.conv2former import ConvMod
from .blocks.cst import WideFocusBlock, ConvAttnWideFocusBlock

# ==============================================================================
# 1. WT-CBAM Block (頻域修復 + 抗噪過濾) - 關鍵組件
# ==============================================================================
class WT_CBAM_Block(nn.Module):
    """
    Sequential Refinement: 
    1. WTConv3d (Recover High-freq Edges) 
    2. CBAM (Filter Background Noise)
    """
    def __init__(self, dim, wave_level=1):
        super().__init__()
        # 1. 小波卷積：提取高頻邊界
        self.wt_conv = WTConv3d(dim, dim, kernel_size=5, wt_levels=wave_level)
        self.norm = nn.BatchNorm3d(dim)
        self.act = nn.LeakyReLU(inplace=True)
        
        # 2. CBAM：過濾背景雜訊 (針對 HD95)
        self.attention = CBAM(channel=dim, reduction=16, kernel_size=7)

    def forward(self, x):
        residual = x
        
        out = self.wt_conv(x)
        out = self.act(self.norm(out))
        
        out = self.attention(out)
        
        return out + residual

# ==============================================================================
# 2. W-Inception Block (Encoder 雙分支)
# ==============================================================================
class W_InceptionBlock(nn.Module):
    """
    Dual-Branch Block:
    - Branch A: InceptionNeXt (Spatial Context)
    - Branch B: WTConv3d (Frequency Details)
    """
    def __init__(self, dim, kernel_size, exp_rate, drop_path, wave_level=1):
        super().__init__()
        
        # Branch A: InceptionNeXt (空間語意)
        self.spatial_branch = InceptionNeXtBlock_V2(
            dim=dim, 
            kernel_size=kernel_size, 
            exp_rate=exp_rate, 
            drop_path=drop_path
        )
        
        # Branch B: WTConv (頻域邊界)
        self.freq_branch = WTConv3d(dim, dim, kernel_size=5, wt_levels=wave_level)
        
        # Fusion Layer (1x1 Conv)
        self.fusion = nn.Conv3d(dim * 2, dim, kernel_size=1)
        self.norm = nn.BatchNorm3d(dim)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        feat_spatial = self.spatial_branch(x)
        feat_freq = self.freq_branch(x)
        
        # 融合
        out = torch.cat([feat_spatial, feat_freq], dim=1)
        out = self.fusion(out)
        out = self.act(self.norm(out))
        
        # Residual
        return x + out

# ==============================================================================
# 3. WISNET (Optimized Version)
# ==============================================================================
class WISNET(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=2,
            patch_size=2, # 建議維持 2 以獲得最高精度，若訓練仍太慢可改 4
            kernel_size=7,
            exp_rate=4,
            feature_size=48,
            depths=[3, 3, 9, 3],
            drop_path_rate=0.0, 
            use_init_weights=False,
            is_conv_stem=False,
            skip_encoder_name='cbam', # 建議開啟，針對 AOV 優化
            deep_sup=False,
            first_feature_size_half=False,
            wave_level=2, # 小波層數
            **kwargs,
    ) -> None:
        super().__init__()
        
        feature_sizes = [feature_size*(2**i) for i in range(len(depths))]
        
        if first_feature_size_half:
            first_feature_size = feature_sizes[0] // 2
        else:
            first_feature_size = feature_sizes[0]
        
        decoder_norm_name = 'instance'
        res_block = True
        spatial_dims = 3
        
        # --- Encoder 0 (Stem) ---
        self.encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=first_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )
        
        # --- Hybrid Encoder (Inception + WT) ---
        self.backbone = HybridBackbone(
            in_channels=in_channels,
            patch_size=patch_size,
            kernel_size=kernel_size,
            exp_rate=exp_rate,
            feature_sizes=feature_sizes,
            depths=depths,
            drop_path_rate=drop_path_rate,
            use_init_weights=use_init_weights,
            is_conv_stem=is_conv_stem,
            wave_level=wave_level
        )
        
        # --- Skip Connections ---
        self.skip_encoder_name = skip_encoder_name       
        if self.skip_encoder_name == 'cbam':
             self.skip_encoder0 = nn.Identity()
             self.skip_encoder1 = CBAM(feature_sizes[0], reduction=16, kernel_size=7)
             self.skip_encoder2 = CBAM(feature_sizes[1], reduction=16, kernel_size=7)
             self.skip_encoder3 = CBAM(feature_sizes[2], reduction=16, kernel_size=7)
             self.skip_encoder4 = CBAM(feature_sizes[3], reduction=16, kernel_size=7)
        # ... (省略其他 skip 選項以節省篇幅，邏輯同前) ...
        
        # --- Bottleneck (ViT Global Reasoning) ---
        # 1. 下採樣 + CBAM (進入 Stage 5)
        self.bottleneck_downsample = nn.Sequential(
            LayerNorm(feature_sizes[3], eps=1e-6, data_format="channels_first"),
            nn.Conv3d(feature_sizes[3], feature_sizes[3]*2, kernel_size=2, stride=2),
            CBAM(feature_sizes[3]*2, reduction=16, kernel_size=7)
        )

        # 2. 全域注意力 (ViT)
        self.bottleneck = ViTBlock3D(
            in_channels=feature_sizes[3]*2, 
            mlp_ratio=4.0,
            num_heads=8,
            drop=0.0, 
            attn_drop=0.0,
            drop_path=0.0
        )

        # --- Decoder (Selective Refinement Strategy) ---
        # 策略：深層只做上採樣 (加速)，淺層做 WT-CBAM (修復)

        # Level 5 (深層): 僅上採樣融合
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[3]*2, 
            out_channels=feature_sizes[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block
        )
        
        # Level 4 (深層): 僅上採樣融合
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[3],
            out_channels=feature_sizes[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block
        )

        # Level 3 (中層): 僅上採樣融合 (視顯存情況，若夠大可加 WT，但建議不加省時間)
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[2],
            out_channels=feature_sizes[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )

        # ★★★ Level 2 (淺層 - 關鍵戰場): 加上 WT-CBAM 修復 ★★★
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[1],
            out_channels=feature_sizes[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )
        self.wt_dec2 = WT_CBAM_Block(feature_sizes[0], wave_level=wave_level)
        
        # ★★★ Level 1 (最淺層 - 最終修復): 加上 WT-CBAM 修復 ★★★
        # 這是決定 HD95/ASSD 的最後一哩路，必須加強
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[0],
            out_channels=first_feature_size,
            kernel_size=3,
            upsample_kernel_size=patch_size,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )
        self.wt_dec1 = WT_CBAM_Block(first_feature_size, wave_level=wave_level)

        self.out_block = UnetOutBlock(spatial_dims=3, in_channels=first_feature_size, out_channels=out_channels)
    
        # Deep Supervision
        self.deep_sup = deep_sup
        if deep_sup:
            self.ds_block1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_sizes[0], out_channels=out_channels)
            self.ds_block2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_sizes[1], out_channels=out_channels)

    def forward(self, x):
        # Encoder
        enc0 = self.encoder0(x)
        hidden_states_out = self.backbone(x) # 跑 W_InceptionBlock
        enc1, enc2, enc3, enc4 = hidden_states_out

        # Skip Connection Refinement
        if self.skip_encoder_name:
            if hasattr(self, 'skip_encoder1'): enc1 = self.skip_encoder1(enc1)
            if hasattr(self, 'skip_encoder2'): enc2 = self.skip_encoder2(enc2)
            if hasattr(self, 'skip_encoder3'): enc3 = self.skip_encoder3(enc3)
            if hasattr(self, 'skip_encoder4'): enc4 = self.skip_encoder4(enc4)

        # Bottleneck
        bn_feat = self.bottleneck_downsample(enc4)
        bn = self.bottleneck(bn_feat)

        # Decoder 5 (無 WT)
        dec5 = self.decoder5(bn, enc4)
        
        # Decoder 4 (無 WT)
        dec4 = self.decoder4(dec5, enc3)
        
        # Decoder 3 (無 WT)
        dec3 = self.decoder3(dec4, enc2)
        
        # Decoder 2 (有 WT-CBAM!)
        dec2 = self.decoder2(dec3, enc1)
        dec2 = self.wt_dec2(dec2) 
        
        # Decoder 1 (有 WT-CBAM!)
        dec1 = self.decoder1(dec2, enc0)
        dec1 = self.wt_dec1(dec1)
        
        out = self.out_block(dec1)
        
        if self.deep_sup and self.training:
            out1 = self.ds_block1(dec2)
            out2 = self.ds_block2(dec3)
            return [out, out1, out2]
        else:
            return out


# ==============================================================================
# 4. Hybrid Backbone (修正後的版本)
# ==============================================================================
class HybridBackbone(nn.Module):
    def __init__(self, in_channels, patch_size, kernel_size, exp_rate, feature_sizes, depths, drop_path_rate, use_init_weights, is_conv_stem, wave_level):
        super().__init__()
        
        self.downsample_layers = nn.ModuleList()
        # Stem
        if is_conv_stem:
            stem = nn.Sequential(
                nn.Conv3d(in_channels, feature_sizes[0], kernel_size=7, stride=patch_size, padding=3),
                LayerNorm(feature_sizes[0], eps=1e-6, data_format="channels_first")
            )
        else:
             stem = nn.Sequential(
                nn.Conv3d(in_channels, feature_sizes[0], kernel_size=patch_size, stride=patch_size),
                LayerNorm(feature_sizes[0], eps=1e-6, data_format="channels_first")
            )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(feature_sizes[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv3d(feature_sizes[i], feature_sizes[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Stages (使用 W_InceptionBlock)
        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        
        for i in range(4):
            stage = nn.Sequential(
                *[
                    W_InceptionBlock(  # ★★★ 修正：確保使用雙分支 Block ★★★
                        dim=feature_sizes[i], 
                        kernel_size=kernel_size,
                        exp_rate=exp_rate,
                        drop_path=dp_rates[cur + j],
                        wave_level=wave_level
                    )
                for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        if use_init_weights:
            self.apply(self._init_weights)

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(x)
        return outs

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)