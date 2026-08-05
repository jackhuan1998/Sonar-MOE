import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
# -------------------------------
# 频率索引映射
# -------------------------------
def get_freq_indices(method='top16'):
    """根据方法选择DCT频率索引"""
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']

    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError

    return mapper_x, mapper_y


# -------------------------------
# DCT 频域提取层
# -------------------------------
class MultiSpectralDCTLayer(nn.Module):
    """DCT频域特征提取"""
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super().__init__()
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        # x: [N, C, H, W]
        return torch.sum(x * self.weight, dim=[2,3])  # 输出 [N, C]

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, H, W, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, H, W)
        c_part = channel // len(mapper_x)
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(H):
                for t_y in range(W):
                    val = self.build_filter(t_x, u_x, H) * self.build_filter(t_y, v_y, W)
                    dct_filter[i*c_part:(i+1)*c_part, t_x, t_y] = val
        return dct_filter


class DCTMultiSpectralAttention(nn.Module):
    def __init__(self, channels, reduction=16, freq_sel_method='top16'):
        super().__init__()

        self.channels = channels
        self.dct_h = 7
        self.dct_w = 7

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)

        mapper_x = [x * (self.dct_h // 7) for x in mapper_x]
        mapper_y = [y * (self.dct_w // 7) for y in mapper_y]

        self.dct_layer = MultiSpectralDCTLayer(
            self.dct_h, self.dct_w, mapper_x, mapper_y, channels
        )

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # ✔ 强约束：通道必须一致
        assert C == self.channels, f"Expected {self.channels}, got {C}"

        # ✔ 自适应到 DCT尺寸
        if H != self.dct_h or W != self.dct_w:
            x_pooled = F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        else:
            x_pooled = x

        # ✔ 频域提取
        y = self.dct_layer(x_pooled)   # (B, C)

        # ✔ 通道注意力
        y = self.fc(y).view(B, C, 1, 1)

        return x * y

class LargeKernelSpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.dw1 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.dw2 = nn.Conv2d(channels, channels, kernel_size=11, padding=5, groups=channels)

        self.fuse = nn.Conv2d(channels, channels, 1)

        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x

        x1 = self.dw1(x)
        x2 = self.dw2(x)

        x = x1 + x2
        x = self.fuse(x)
        x = self.bn(x)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.spatial(attn)

        return identity * attn

class LocalDetailAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # ✔ 局部细节提取（3x3 depthwise）
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

        # ✔ 通道压缩
        self.pw = nn.Conv2d(channels, channels, 1)

        # ✔ 边缘增强（类似Laplacian思想）
        self.edge_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

        # ✔ 注意力生成
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x

        # ===== 局部纹理 =====
        local_feat = self.dw(x)
        local_feat = self.pw(local_feat)

        # ===== 边缘增强 =====
        edge = self.edge_conv(x)

        # ===== 融合 =====
        feat = local_feat + edge
        feat = self.bn(feat)

        # ===== 注意力 =====
        weight = self.attn(feat)

        return identity * weight
##########################################

class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim == 64: #96 for large model, 64 for small and base model
            self.h = 56 #H
            self.w = 29 #(W/2)+1            
            self.complex_weight = nn.Parameter(torch.randn(self.h, self.w, dim, 2, dtype=torch.float32) * 0.02)
        if dim ==128:
            self.h = 28 #H
            self.w = 15 #(W/2)+1, this is due to rfft2
            self.complex_weight = nn.Parameter(torch.randn(self.h, self.w, dim, 2, dtype=torch.float32) * 0.02)
        if dim == 96: #96 for large model, 64 for small and base model
            self.h = 56 #H
            self.w = 29 #(W/2)+1            
            self.complex_weight = nn.Parameter(torch.randn(self.h, self.w, dim, 2, dtype=torch.float32) * 0.02)
        if dim ==192:
            self.h = 28 #H
            self.w = 15 #(W/2)+1, this is due to rfft2
            self.complex_weight = nn.Parameter(torch.randn(self.h, self.w, dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x, H, W):
        B, N, C = x.shape 
        x = x.view(B, H, W, C)
        x = x.to(torch.float32) 
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)# permute is not same as reshape or view
        return x


class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x

class FFTSpectralAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class FFTSpectralBlockAttention(nn.Module):
    def __init__(self, 
        dim, 
        num_heads, 
        mlp_ratio,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        sr_ratio=1, 
        block_type = 'wave'
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        if block_type == 'std_att':
            self.attn = FFTSpectralAttention(dim, num_heads)
        else:
            self.attn = SpectralGatingNetwork (dim)
        self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class FFTSpectralBlockWrapper(nn.Module):
    def __init__(self, channels, num_heads=4, mlp_ratio=2.0):
        super().__init__()

        self.channels = channels

        self.block = FFTSpectralBlockAttention(
            dim=channels,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # ✔ (B,C,H,W) → (B,N,C)
        x_flat = x.flatten(2).transpose(1, 2)   # B, HW, C

        # ✔ Transformer block
        x_flat = self.block(x_flat, H, W)

        # ✔ (B,N,C) → (B,C,H,W)
        x_out = x_flat.transpose(1, 2).reshape(B, C, H, W)

        return x_out
##########################################
