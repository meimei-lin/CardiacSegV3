import torch
import torch.nn as nn
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt3D(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt3D, self).__init__()
        # 3D 空間的三個方向池化
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1)) 
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1)) 
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None)) 

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.InstanceNorm3d(mip) 
        self.act = h_swish()
        
        self.conv_d = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        n, c, d, h, w = x.size()

        # 提取三個維度的特徵並統一轉置到 dim=2 拼接
        x_d = self.pool_d(x)                            
        x_h = self.pool_h(x).permute(0, 1, 3, 2, 4)     
        x_w = self.pool_w(x).permute(0, 1, 4, 3, 2)     

        # 拼接 D, H, W 的特徵
        y = torch.cat([x_d, x_h, x_w], dim=2)           
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        # 依照原本的長度拆分
        res_d, res_h, res_w = torch.split(y, [d, h, w], dim=2)
        
        # 生成三路 Attention Map 並還原維度
        a_d = self.conv_d(res_d).sigmoid()      
        
        # 還原 H 的順序
        res_h = res_h.permute(0, 1, 3, 2, 4)  
        a_h = self.conv_h(res_h).sigmoid()
        
        # 還原 W 的順序
        res_w = res_w.permute(0, 1, 4, 3, 2)
        a_w = self.conv_w(res_w).sigmoid()

        #三向座標加權 
        out = identity * a_d * a_h * a_w

        return out