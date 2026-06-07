import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import pywt.data

# ==============================================================================
# 1. 3D Wavelet Filter Creation (Haar Wavelet for 3D)
# ==============================================================================
def create_wavelet_filter_3d(wave, in_size, out_size, type=torch.float):
    """
    建立 3D Haar 小波濾波器 (8個子頻帶)
    """
    w = pywt.Wavelet(wave)
    
    # 1D Filters
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])

    # 3D Filters Construction (Tensor Product of 1D filters)
    # Order: LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH
    # Shape: (8, 1, 2, 2, 2)
    
    filters = []
    rec_filters_list = []
    
    # Loop through all combinations of Low/High for Z, Y, X axes
    for z in [dec_lo, dec_hi]:
        for y in [dec_lo, dec_hi]:
            for x in [dec_lo, dec_hi]:
                # Outer product to make 3D kernel
                # z[k] * y[j] * x[i]
                kernel = z.view(-1, 1, 1) * y.view(1, -1, 1) * x.view(1, 1, -1)
                filters.append(kernel)

    for z in [rec_lo, rec_hi]:
        for y in [rec_lo, rec_hi]:
            for x in [rec_lo, rec_hi]:
                kernel = z.view(-1, 1, 1) * y.view(1, -1, 1) * x.view(1, 1, -1)
                rec_filters_list.append(kernel)

    dec_filters = torch.stack(filters, dim=0) # (8, 2, 2, 2)
    rec_filters = torch.stack(rec_filters_list, dim=0)

    # Repeat for input channels (Depthwise convolution style)
    # Output shape: (In * 8, 1, 2, 2, 2) -> Groups=In
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1, 1)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform_3d(x, filters):
    """
    3D DWT: Input (B, C, D, H, W) -> Output (B, C, 8, D/2, H/2, W/2)
    """
    b, c, d, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1, filters.shape[4] // 2 - 1)
    
    # 3D Convolution with stride 2
    x = F.conv3d(x, filters, stride=2, groups=c, padding=pad)
    
    # Reshape to separate the 8 subbands
    x = x.reshape(b, c, 8, d // 2, h // 2, w // 2)
    return x

def inverse_wavelet_transform_3d(x, filters):
    """
    3D IWT: Input (B, C, 8, D/2, H/2, W/2) -> Output (B, C, D, H, W)
    """
    b, c, _, d_half, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1, filters.shape[4] // 2 - 1)
    
    # Combine channels and subbands for transposed conv
    x = x.reshape(b, c * 8, d_half, h_half, w_half)
    
    # 3D Transposed Convolution with stride 2
    #x = F.conv_transpose3d(x, filters, stride=2, groups=c, padding=pad)
    with torch.backends.cudnn.flags(enabled=False):
        x = F.conv_transpose3d(x, filters, stride=2, groups=c, padding=pad)
    return x

# ==============================================================================
# 2. Scale Module (Learnable Weighting)
# ==============================================================================
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)

# ==============================================================================
# 3. WTConv3d Layer
# ==============================================================================
class WTConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=2, wt_type='db1'):
        super(WTConv3d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        
        # 建立 3D Filters
        self.wt_filter, self.iwt_filter = create_wavelet_filter_3d(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        # 基礎卷積 (Standard 3D Conv)
        self.base_conv = nn.Conv3d(in_channels, in_channels, kernel_size, padding='same', stride=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1, 1]) # 5D tensor for 3D data

        # 小波卷積層 (處理 8 個子頻帶)
        # 注意：通道數變為 in_channels * 8
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv3d(in_channels*8, in_channels*8, kernel_size, padding='same', stride=1, groups=in_channels*8, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels*8, 1, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool3d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        # === Wavelet Decomposition ===
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            
            # Padding handling for odd dimensions in 3D (D, H, W)
            d, h, w = curr_shape[2], curr_shape[3], curr_shape[4]
            if (d % 2 > 0) or (h % 2 > 0) or (w % 2 > 0):
                curr_pads = (0, w % 2, 0, h % 2, 0, d % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet_transform_3d(curr_x_ll, self.wt_filter)
            
            # 3D DWT Output shape: (B, C, 8, D/2, H/2, W/2)
            # LLL is index 0
            curr_x_ll = curr_x[:,:,0,:,:,:]
            
            shape_x = curr_x.shape
            # Flatten subbands into channels for group convolution
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 8, shape_x[3], shape_x[4], shape_x[5])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:,:])
            x_h_in_levels.append(curr_x_tag[:,:,1:8,:,:,:]) # Remaining 7 high-freq bands

        # === Inverse Reconstruction ===
        next_x_ll = 0

        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform_3d(curr_x, self.iwt_filter)

            # Crop to original shape
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3], :curr_shape[4]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x