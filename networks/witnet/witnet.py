import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from monai.networks.blocks import UnetrBasicBlock, UnetOutBlock, UnetrUpBlock
from .blocks.test_block import UnetrUpCABlock
from .blocks.inceptionnext_v2 import InceptionNeXtBlock_V2
from .blocks.utils import LayerNorm
from .blocks.cbam import CBAM 
from .blocks.transformer import Transformer
from networks.uwtnet_3d.uwtnet_3d import WTConv3d 
from .blocks.eca import ECA
from .blocks.conv2former import ConvMod
from .blocks.cst import WideFocusBlock, ConvAttnWideFocusBlock
from .blocks.InceptionWTBlock import InceptionWTBlock

class WITNET(nn.Module):
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
            wave_level=1, 
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

         # Skip Connections
        self.skip_encoder_name = skip_encoder_name
        if self.skip_encoder_name == 'cbam': 
             self.skip_encoder0 = nn.Identity()
             self.skip_encoder1 = CBAM(feature_sizes[0], reduction=16, kernel_size=7)
             self.skip_encoder2 = CBAM(feature_sizes[1], reduction=16, kernel_size=7)
             self.skip_encoder3 = CBAM(feature_sizes[2], reduction=16, kernel_size=7)
             self.skip_encoder4 = CBAM(feature_sizes[3], reduction=16, kernel_size=7)
        
        elif self.skip_encoder_name == 'hybrid': 
             # Level 0 (Stem)
             self.skip_encoder0 = nn.Identity() 
             # Level 1 (Stage 0): CBAM 
             self.skip_encoder1 = CBAM(feature_sizes[0], reduction=16, kernel_size=7) 
             # Level 2 (Stage 1): CBAM 
             self.skip_encoder2 = CBAM(feature_sizes[1], reduction=16, kernel_size=7)
             # Level 3 (Stage 2): Identity 
             self.skip_encoder3 = nn.Identity()
             # Level 4 (Stage 3): Identity
             self.skip_encoder4 = nn.Identity()

        self.bottleneck = nn.Sequential(
            LayerNorm(feature_sizes[3], eps=1e-6, data_format="channels_first"),
            nn.Conv3d(feature_sizes[3], feature_sizes[3]*2, kernel_size=2, stride=2),
            CBAM(feature_sizes[3]*2, reduction=16, kernel_size=7)
        )

        # Decoder 5  
        self.decoder5 = UnetrUpCABlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[3]*2, 
            out_channels=feature_sizes[3],
            kernel_size=3,
            upsample_kernel_size=2, 
            norm_name=decoder_norm_name,
            res_block=res_block
        )
             
        # Level 4
        self.decoder4 = UnetrUpCABlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[3],
            out_channels=feature_sizes[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block
        )
        #self.wt_dec4 = WTConv3d(feature_sizes[2], feature_sizes[2], wt_levels=wave_level)

        # Level 3
        self.decoder3 = UnetrUpCABlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[2],
            out_channels=feature_sizes[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )
        #self.wt_dec3 = WTConv3d(feature_sizes[1], feature_sizes[1], wt_levels=wave_level)

        # Level 2
        self.decoder2 = UnetrUpCABlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[1],
            out_channels=feature_sizes[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )
        #self.wt_dec2 = WTConv3d(feature_sizes[0], feature_sizes[0], wt_levels=wave_level)
        
        # Level 1
        self.decoder1 = UnetrUpCABlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[0],
            out_channels=first_feature_size,
            kernel_size=3,
            upsample_kernel_size=patch_size,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )
        self.out_block = UnetOutBlock(spatial_dims=3, in_channels=first_feature_size, out_channels=out_channels)
        
        # Deep Supervision
        self.deep_sup = deep_sup
        if deep_sup:
            self.ds_block1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_sizes[0], out_channels=out_channels)
            self.ds_block2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_sizes[1], out_channels=out_channels)

    def forward(self, x):
        # Encoder
        enc0 = self.encoder0(x)
        hidden_states_out = self.backbone(x) 
        enc1, enc2, enc3, enc4 = hidden_states_out

        # Skip Connection Refinement
        if self.skip_encoder_name == 'cbam':
            enc1 = self.skip_encoder1(enc1)
            enc2 = self.skip_encoder2(enc2)
            enc3 = self.skip_encoder3(enc3)
            enc4 = self.skip_encoder4(enc4)
        elif self.skip_encoder_name == 'hybrid':
            enc1 = enc1 + self.skip_encoder1(enc1) 
            enc2 = enc2 + self.skip_encoder2(enc2)
            # Level 3 & 4: Identity
            enc3 = self.skip_encoder3(enc3)
            enc4 = self.skip_encoder4(enc4)

        # Bottleneck       
        bn = self.bottleneck(enc4)

        # Decoder
        dec5 = self.decoder5(bn, enc4)
        
        dec4 = self.decoder4(dec5, enc3)
        #dec4 = dec4 + self.wt_dec4(dec4)
        dec3 = self.decoder3(dec4, enc2)
        #dec3 = dec3 + self.wt_dec3(dec3) 

        dec2 = self.decoder2(dec3, enc1)
        #dec2 = dec2 + self.wt_dec2(dec2) 

        dec1 = self.decoder1(dec2, enc0)
        out = self.out_block(dec1)
        
        if self.deep_sup and self.training:
            out1 = self.ds_block1(dec2)
            out2 = self.ds_block2(dec3)
            return [out, out1, out2]
        else:
            return out

# Hybrid Backbone
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

        # Stages
        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        
        for i in range(4):
            #curr_wave_level = wave_level if i < 2 else 1
            #curr_wave_level = 1 if i < 2 else wave_level
            stage = nn.Sequential(
                *[
                    InceptionWTBlock(  
                        dim=feature_sizes[i], 
                        kernel_size=kernel_size,
                        exp_rate=exp_rate,
                        drop_path=dp_rates[cur + j],
                        wt_levels=wave_level 
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