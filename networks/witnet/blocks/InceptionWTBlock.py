import torch
import torch.nn as nn
from timm.models.layers import DropPath
from .utils import LayerNorm, GRN
from .wtconv3d import WTConv3d 

class InceptionWTBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        kernel_size=11, 
        exp_rate=4, 
        drop_path=0., 
        wt_levels=1, 
        wt_type='db1'
    ):
        super().__init__()
        self.gc = int(dim * 0.2)
        #self.alpha = nn.Parameter(torch.tensor(0.5))
        # 空間分支：4 路卷積
        self.dwconv_hwd = nn.Conv3d(self.gc, self.gc, kernel_size=5, padding=2, groups=self.gc)
        self.dwconv_h = nn.Conv3d(self.gc, self.gc, kernel_size=(kernel_size, kernel_size, 1), 
                                   padding=(kernel_size//2, kernel_size//2, 0), groups=self.gc)
        self.dwconv_w = nn.Conv3d(self.gc, self.gc, kernel_size=(1, kernel_size, kernel_size), 
                                   padding=(0, kernel_size//2, kernel_size//2), groups=self.gc)
        self.dwconv_d = nn.Conv3d(self.gc, self.gc, kernel_size=(kernel_size, 1, kernel_size), 
                                   padding=(kernel_size//2, 0, kernel_size//2), groups=self.gc)
        
        # 定義切分索引
        self.split_indexes = (dim - 4 * self.gc, self.gc, self.gc, self.gc, self.gc)

        inter_dim = max(8, dim // 4) 
        self.freq_path = nn.Sequential(
            nn.Conv3d(dim, inter_dim, 1),
            WTConv3d(inter_dim, inter_dim, kernel_size=5, wt_levels=wt_levels, wt_type=wt_type),
            nn.Conv3d(inter_dim, dim, 1),
            nn.Sigmoid() 
        )
        #self.beta = nn.Parameter(torch.ones(dim, 1, 1, 1) * 0.1 )

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, exp_rate * dim)
        self.act = nn.GELU()
        self.grn = GRN(exp_rate * dim)
        self.pwconv2 = nn.Linear(exp_rate * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x_id, x_hwd, x_w, x_h, x_d = torch.split(x, self.split_indexes, dim=1) 
        
        x_spatial = torch.cat((
            x_id,
            self.dwconv_hwd(x_hwd), 
            self.dwconv_w(x_w), 
            self.dwconv_h(x_h), 
            self.dwconv_d(x_d)
        ), dim=1) 
        
        freq_attn = self.freq_path(x)
        x_modulated = x_spatial  * (1 + freq_attn)
        
        x = x_modulated.permute(0, 2, 3, 4, 1) # Channel Last
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3) # Channel First

        x = input + self.drop_path(x)
        return x