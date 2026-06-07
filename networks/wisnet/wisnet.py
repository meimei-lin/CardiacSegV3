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
# 1. W-Inception Block (Encoder 核心創新)
# ==============================================================================
class W_InceptionBlock(nn.Module):
    """
    Dual-Branch Block:
    - Branch A: InceptionNeXt (Spatial Context)
    - Branch B: WTConv3d (Frequency Details)
    """
    def __init__(self, dim, kernel_size, exp_rate, drop_path, wave_level=1):
        super().__init__()
        
        # Branch A: InceptionNeXt (現有)
        self.spatial_branch = InceptionNeXtBlock_V2(
            dim=dim, 
            kernel_size=kernel_size, 
            exp_rate=exp_rate, 
            drop_path=drop_path
        )
        
        # Branch B: WTConv (新增)
        # 為了不增加太多參數量，我們可以只用一層 WTConv
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
        
        # Residual Connection (如果輸入輸出維度允許)
        return x + out

# ==============================================================================
# 2. Proposed Full Model
# ==============================================================================
class WISNET(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=2,
            patch_size=2,
            kernel_size=7,
            exp_rate=4,
            feature_size=48,
            depths=[3, 3, 9, 3],
            drop_path_rate=0.0, 
            use_init_weights=False,
            is_conv_stem=False,
            skip_encoder_name=None,
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
            wave_level=wave_level # 傳入小波層數
        )
        
        # --- Skip Connections ---
        self.skip_encoder_name = skip_encoder_name       
        if self.skip_encoder_name == 'cbam':
             self.skip_encoder0 = nn.Identity()
             self.skip_encoder1 = CBAM(feature_sizes[0], reduction=16, kernel_size=7)
             self.skip_encoder2 = CBAM(feature_sizes[1], reduction=16, kernel_size=7)
             self.skip_encoder3 = CBAM(feature_sizes[2], reduction=16, kernel_size=7)
             self.skip_encoder4 = CBAM(feature_sizes[3], reduction=16, kernel_size=7)
        elif self.skip_encoder_name == 'eca':
            print('use skip encoder: eca')
            self.skip_encoder0 = ECA(feature_sizes[0], k_size=3)
            self.skip_encoder1 = ECA(feature_sizes[0], k_size=3)
            self.skip_encoder2 = ECA(feature_sizes[1], k_size=3)
            self.skip_encoder3 = ECA(feature_sizes[2], k_size=3)
            self.skip_encoder4 = ECA(feature_sizes[3], k_size=3)
        elif self.skip_encoder_name == 'convmod':
            print('use skip encoder: convmod')
            self.skip_encoder0 = nn.Identity()
            self.skip_encoder1 = nn.Identity()
            self.skip_encoder2 = ConvMod(feature_sizes[1])
            self.skip_encoder3 = ConvMod(feature_sizes[2])
            self.skip_encoder4 = ConvMod(feature_sizes[3])
        elif self.skip_encoder_name == 'res':
            print('use skip encoder: res')
            self.skip_encoder0 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_sizes[0],
                out_channels=feature_sizes[0],
                kernel_size=3,
                stride=1,
                norm_name=decoder_norm_name,
                res_block=res_block,
            )
            self.skip_encoder1 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_sizes[0],
                out_channels=feature_sizes[0],
                kernel_size=3,
                stride=1,
                norm_name=decoder_norm_name,
                res_block=res_block,
            )
            self.skip_encoder2 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_sizes[1],
                out_channels=feature_sizes[1],
                kernel_size=3,
                stride=1,
                norm_name=decoder_norm_name,
                res_block=res_block,
            )
            self.skip_encoder3 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_sizes[2],
                out_channels=feature_sizes[2],
                kernel_size=3,
                stride=1,
                norm_name=decoder_norm_name,
                res_block=res_block,
            )
            self.skip_encoder4 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_sizes[3],
                out_channels=feature_sizes[3],
                kernel_size=3,
                stride=1,
                norm_name=decoder_norm_name,
                res_block=res_block,
            )
        elif self.skip_encoder_name == 'wf':
            self.skip_encoder0 = nn.Identity()
            self.skip_encoder1 = nn.Identity()
            self.skip_encoder2 = nn.Identity()
            self.skip_encoder3 = nn.Identity()
            self.skip_encoder4 = WideFocusBlock(feature_sizes[3])
        elif self.skip_encoder_name == 'cawf':
            self.skip_encoder0 = ConvAttnWideFocusBlock(feature_sizes[0])
            self.skip_encoder1 = ConvAttnWideFocusBlock(feature_sizes[0])
            self.skip_encoder2 = ConvAttnWideFocusBlock(feature_sizes[1])
            self.skip_encoder3 = ConvAttnWideFocusBlock(feature_sizes[2])
            self.skip_encoder4 = ConvAttnWideFocusBlock(feature_sizes[3])
        # self.swin_bottleneck = ViTBlock3D(
        #     in_channels=feature_sizes[3],
        #     embed_dim=feature_sizes[3],
        #     patch_size=(2, 2, 2),
        #     window_size=(7, 7, 7), # 如果顯存不夠可改 (4,4,4)
        #     mlp_ratio=4.0,
        #     depths=(2, 2),
        #     num_heads=8,
        #     feature_size=12, # 需根據輸入大小調整，或讓 MONAI 自動推斷
        #     spatial_dims=3,
        # )
        
        # 修改 1: 增加下採樣層
        self.bottleneck_downsample = nn.Sequential(
            LayerNorm(feature_sizes[3], eps=1e-6, data_format="channels_first"),
            nn.Conv3d(feature_sizes[3], feature_sizes[3]*2, kernel_size=2, stride=2),
            CBAM(feature_sizes[3]*2, reduction=16, kernel_size=7)
        )

        # 修改 2: Transformer (輸入通道變 2 倍)
        self.bottleneck = ViTBlock3D(
            in_channels=feature_sizes[3]*2, # <--- 變 2 倍
            mlp_ratio=4.0,
            num_heads=8,
            drop=0.0,       # dropout
            attn_drop=0.0,  # attention dropout
            drop_path=0.0   # stochastic depth
        )

        # 修改 3: Decoder 5  
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[3]*2, 
            out_channels=feature_sizes[3],
            kernel_size=3,
            upsample_kernel_size=2, # <--- 恢復上採樣
            norm_name=decoder_norm_name,
            res_block=res_block
        )
             
        # Level 4
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[3],
            out_channels=feature_sizes[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block
        )
        self.wt_dec4 = WTConv3d(feature_sizes[2], feature_sizes[2], wt_levels=wave_level)

        # Level 3
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[2],
            out_channels=feature_sizes[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )
        self.wt_dec3 = WTConv3d(feature_sizes[1], feature_sizes[1], wt_levels=wave_level)

        # Level 2
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[1],
            out_channels=feature_sizes[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )
        self.wt_dec2 = WTConv3d(feature_sizes[0], feature_sizes[0], wt_levels=wave_level)
        
        # Level 1
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[0],
            out_channels=first_feature_size,
            kernel_size=3,
            upsample_kernel_size=patch_size,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )
        # 最後一層通常不做 WT，直接輸出

        self.out_block = UnetOutBlock(spatial_dims=3, in_channels=first_feature_size, out_channels=out_channels)
        
        # Deep Supervision
        self.deep_sup = deep_sup
        if deep_sup:
            self.ds_block1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_sizes[0], out_channels=out_channels)
            self.ds_block2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_sizes[1], out_channels=out_channels)

    def forward(self, x):
        # Encoder
        enc0 = self.encoder0(x)
        hidden_states_out = self.backbone(x) # Backbone 現在會跑 W-Inception Block
        enc1, enc2, enc3, enc4 = hidden_states_out

        # Skip Connection Refinement
        if self.skip_encoder_name:
            enc1 = self.skip_encoder1(enc1)
            enc2 = self.skip_encoder2(enc2)
            enc3 = self.skip_encoder3(enc3)
            enc4 = self.skip_encoder4(enc4)

        #bn = self.swin_bottleneck(enc4) 

        # 先下採樣
        bn_feat = self.bottleneck_downsample(enc4)
        
        bn = self.bottleneck(bn_feat)

        # 進 Decoder 5 (這時候 bn 是 Stage 5 的特徵)
        dec5 = self.decoder5(bn, enc4)
        # Decoder + WT Refinement
        dec4 = self.decoder4(dec5, enc3)
        dec4 = self.wt_dec4(dec4) # 加入 WT 修復

        dec3 = self.decoder3(dec4, enc2)
        dec3 = self.wt_dec3(dec3) # 加入 WT 修復

        dec2 = self.decoder2(dec3, enc1)
        dec2 = self.wt_dec2(dec2) # 加入 WT 修復

        dec1 = self.decoder1(dec2, enc0)
        
        out = self.out_block(dec1)
        
        if self.deep_sup and self.training:
            out1 = self.ds_block1(dec2)
            out2 = self.ds_block2(dec3)
            return [out, out1, out2]
        else:
            return out


# ==============================================================================
# 3. Hybrid Backbone (Inception + WT)
# ==============================================================================
class HybridBackbone(nn.Module):
    def __init__(self, in_channels, patch_size, kernel_size, exp_rate, feature_sizes, depths, drop_path_rate, use_init_weights, is_conv_stem, wave_level):
        super().__init__()
        
        self.downsample_layers = nn.ModuleList()
        
        # Stem (保持原樣)
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

        # Stages
        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        
        for i in range(4):
            # 使用 W_InceptionBlock 
            stage = nn.Sequential(
                *[
                    W_InceptionBlock( 
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
            
            # 檢查 m.bias 是否為 None 
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)