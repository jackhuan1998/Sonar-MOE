# 🐧Please note that this file has been modified by Tencent on 2026/01/16. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Expert modules for Mixture-of-Experts models"""
import torch
import torch.nn as nn
import math
from .utils import FlopsUtils, get_safe_groups
import torch.fft as fft
import pywt  # PyWavelets for Wavelet Transform
from pytorch_wavelets import DWTForward, DWTInverse
import itertools
import torch.nn.functional as F
from einops import rearrange
from .fre_domain import DCTMultiSpectralAttention, FFTSpectralBlockAttention,LargeKernelSpatialAttention,LocalDetailAttention

# ==========================================
# Optimized expert modules
# ==========================================
class OptimizedSimpleExpert(nn.Module):
    

    def __init__(self, in_channels, out_channels, expand_ratio=2, num_groups=8):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.GroupNorm(get_safe_groups(hidden_dim, num_groups), hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.GroupNorm(get_safe_groups(out_channels, num_groups), out_channels)
        )
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return self.conv(x)

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        flops = FlopsUtils.count_conv2d(self.conv[0], (1, C, H, W))
        flops += FlopsUtils.count_conv2d(self.conv[3], (1, self.hidden_dim, H, W))
        return flops


class FusedGhostExpert(nn.Module):
    """Fused Ghost expert that reduces memory traffic by combining operations."""

    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2, num_groups=8):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        # Use GroupNorm to improve stability
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.GroupNorm(min(num_groups, init_channels), init_channels),
            nn.SiLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, padding=1, groups=init_channels, bias=False),
            nn.GroupNorm(min(num_groups, new_channels), new_channels),
            nn.SiLU(inplace=True)
        )
        self.init_channels = init_channels

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        flops = FlopsUtils.count_conv2d(self.primary_conv[0], (1, C, H, W))
        flops += FlopsUtils.count_conv2d(self.cheap_operation[0], (1, self.init_channels, H, W))
        return flops


class SimpleExpert(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=2):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x): return self.conv(x)

    def compute_flops(self, input_shape): return FlopsUtils.count_conv2d(self.conv, input_shape)


class GhostExpert(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)[:, :self.out_channels, :, :]

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        flops = FlopsUtils.count_conv2d(self.primary_conv, input_shape)
        # Compute input shape to cheap op (output of primary conv)
        p_out = self.primary_conv[0].out_channels
        flops += FlopsUtils.count_conv2d(self.cheap_operation, (B, p_out, H, W))
        return flops


class InvertedResidualExpert(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, expand_ratio=2):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_expand = expand_ratio != 1
        layers = []
        if self.use_expand:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=(kernel_size - 1) // 2, groups=hidden_dim,
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x): return self.conv(x)

    def compute_flops(self, input_shape): return FlopsUtils.count_conv2d(self.conv, input_shape)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x
#################################################3
"""
class GlobalContextBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwconv = nn.Conv2d(
            channels, channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=False
        )
        self.pwconv = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.pwconv(self.dwconv(x))))


class ContextRelationExpert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        C = channels

        self.global_context = GlobalContextBlock(C)

        self.mlp = nn.Sequential(
            nn.Conv2d(C, C * 2, 1, bias=False),
            nn.BatchNorm2d(C * 2),
            nn.GELU(),
            nn.Conv2d(C * 2, C, 1, bias=False),
        )

        self.context_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, C, 1, bias=False),
            nn.Sigmoid()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.GELU()
        )

    def forward(self, x):
        gc = self.global_context(x)
        gc = self.mlp(gc)
        gc = gc * self.context_gate(gc)
        out = self.fuse(x + gc)
        return out
######################
class PhaseAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, phase):
        delta = self.conv1(phase)
        delta = F.relu(delta, inplace=True)
        delta = self.conv2(delta)
        delta = torch.tanh(delta)
        return phase + self.alpha * delta


class AmplitudeEnhance(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.conv2 = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, amp):
        weight = F.adaptive_avg_pool2d(amp, 1)
        weight = F.relu(self.conv1(weight), inplace=True)
        weight = torch.sigmoid(self.conv2(weight))
        return amp * weight


class EchoFrequencyExpert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')
        self.idwt = DWTInverse(wave='haar', mode='zero')

        self.amp_enhance = AmplitudeEnhance(channels)
        self.phase_attn = PhaseAttention(channels)

        # 只做轻量投影，不额外做强空间增强
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x

        yl, yh = self.dwt(x)
        fft = torch.fft.fft2(yl, norm='ortho')

        amp = torch.abs(fft)
        phase = torch.angle(fft)

        amp = self.amp_enhance(amp)
        phase = self.phase_attn(phase)

        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        fft_new = torch.complex(real, imag)

        yl_new = torch.fft.ifft2(fft_new, norm='ortho').real
        x_recon = self.idwt((yl_new, yh))

        feat = self.proj(x_recon)
        feat = feat * self.gate(feat)

        return feat + identity
"""
#######################333
"""
class FixedSobel2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32)

        kx = sobel_x.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        ky = sobel_y.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)

        self.conv_x = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.conv_y = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

        self.conv_x.weight.data.copy_(kx)
        self.conv_y.weight.data.copy_(ky)
        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False

    def forward(self, x):
        gx = self.conv_x(x)
        gy = self.conv_y(x)
        return gx, gy


class StructureGeometricExpert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        C = channels

        # 只建模方向几何
        self.dir_h = nn.Conv2d(C, C, kernel_size=(1, 3), padding=(0, 1), groups=C, bias=False)
        self.dir_v = nn.Conv2d(C, C, kernel_size=(3, 1), padding=(1, 0), groups=C, bias=False)

        # 轻量局部形状
        self.shape_conv = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C)
        )

        self.edge = FixedSobel2D(C)

        self.fuse = nn.Sequential(
            nn.Conv2d(C * 2, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.GELU()
        )

        self.gate = nn.Sequential(
            nn.Conv2d(C, C, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x

        dir_feat = self.dir_h(x) + self.dir_v(x)

        gx, gy = self.edge(x)
        edge_mag = torch.abs(gx) + torch.abs(gy)

        shape_feat = self.shape_conv(x)

        feat = self.fuse(torch.cat([dir_feat + shape_feat, edge_mag], dim=1))
        feat = feat * self.gate(feat)

        return feat + identity
"""
##############################

class FixedSobel2D(nn.Module):
    def __init__(self, channels):
        super().__init__()

        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32)

        kx = sobel_x.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        ky = sobel_y.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)

        self.conv_x = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.conv_y = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

        self.conv_x.weight.data.copy_(kx)
        self.conv_y.weight.data.copy_(ky)
        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False

    def forward(self, x):
        gx = self.conv_x(x)
        gy = self.conv_y(x)
        return gx, gy


class StructureGeometricExpert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        C = channels

        # 方向结构感知
        self.dir_h = nn.Conv2d(C, C, kernel_size=(1, 3), padding=(0, 1), groups=C, bias=False)
        self.dir_v = nn.Conv2d(C, C, kernel_size=(3, 1), padding=(1, 0), groups=C, bias=False)

        # 局部形状提取
        self.local_shape = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.GELU()
        )

        # Sobel先验
        self.edge = FixedSobel2D(C)

        # 结构融合
        self.fuse = nn.Sequential(
            nn.Conv2d(C * 3, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.GELU()
        )

        # 轻量门控
        self.gate = nn.Sequential(
            nn.Conv2d(C, C, 1, bias=False),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C)
        )

        self.act = nn.GELU()

    def forward(self, x):
        identity = x

        h_feat = self.dir_h(x)
        v_feat = self.dir_v(x)

        gx, gy = self.edge(x)
        edge_mag = torch.abs(gx) + torch.abs(gy)

        shape_feat = self.local_shape(x)

        feat = self.fuse(torch.cat([h_feat + v_feat, edge_mag, shape_feat], dim=1))
        feat = feat * self.gate(feat)
        feat = self.refine(feat)

        return self.act(feat + identity)
        
class EchoTextureExpert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        C = channels

        # 多尺度纹理建模
        self.tex_small = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
        self.tex_large = nn.Conv2d(C, C, 3, padding=2, dilation=2, groups=C, bias=False)

        self.tex_fuse = nn.Sequential(
            nn.Conv2d(C * 2, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.GELU()
        )

        # 高频/低频分解
        self.low_proj = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C)
        )

        self.high_proj = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C)
        )

        # 通道重标定
        hidden = max(C // 4, 8)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, C, 1, bias=False),
            nn.Sigmoid()
        )

        # 纹理稳定门控
        self.texture_gate = nn.Sequential(
            nn.Conv2d(C, C, 1, bias=False),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C)
        )

        self.act = nn.GELU()

    def forward(self, x):
        identity = x

        # 多尺度回波纹理
        t1 = self.tex_small(x)
        t2 = self.tex_large(x)
        tex = self.tex_fuse(torch.cat([t1, t2], dim=1))

        # 高频低频
        low = self.low_proj(x)
        high = x - low
        high = self.high_proj(high)

        feat = tex + high

        # 通道重标定
        feat = feat * self.channel_attn(feat)

        # 纹理门控
        feat = feat * self.texture_gate(feat)

        feat = self.refine(feat)
        return self.act(feat + identity)

class ContextRelationExpert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        C = channels

        # 大感受野上下文
        self.context_dw1 = nn.Conv2d(C, C, 5, padding=2, groups=C, bias=False)
        self.context_dw2 = nn.Conv2d(C, C, 3, padding=3, dilation=3, groups=C, bias=False)

        self.context_fuse = nn.Sequential(
            nn.Conv2d(C * 2, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.GELU()
        )

        # 横向/纵向关系池化
        self.row_proj = nn.Conv2d(C, C, kernel_size=(1, 3), padding=(0, 1), groups=C, bias=False)
        self.col_proj = nn.Conv2d(C, C, kernel_size=(3, 1), padding=(1, 0), groups=C, bias=False)

        # 全局上下文门控
        hidden = max(C // 4, 8)
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, C, 1, bias=False),
            nn.Sigmoid()
        )

        # 关系融合
        self.rel_fuse = nn.Sequential(
            nn.Conv2d(C * 2, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.GELU()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C)
        )

        self.act = nn.GELU()

    def forward(self, x):
        identity = x

        c1 = self.context_dw1(x)
        c2 = self.context_dw2(x)
        context = self.context_fuse(torch.cat([c1, c2], dim=1))

        relation = self.row_proj(x) + self.col_proj(x)

        feat = self.rel_fuse(torch.cat([context, relation], dim=1))
        feat = feat * self.global_gate(feat)

        feat = self.refine(feat)
        return self.act(feat + identity)



class PhaseAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 控制相位扰动幅度
    def forward(self, phase):
        delta = self.conv1(phase)
        delta = F.relu(delta)
        delta = self.conv2(delta)
        delta = torch.tanh(delta)
        return phase + self.alpha * delta


class AmplitudeEnhance(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 4, 1)
        self.conv2 = nn.Conv2d(channels // 4, channels, 1)
    def forward(self, amp):
        weight = F.adaptive_avg_pool2d(amp, 1)
        weight = F.relu(self.conv1(weight))
        weight = torch.sigmoid(self.conv2(weight))
        return amp * weight

class SpatialEnhance(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, 1)
        self.bn = nn.BatchNorm2d(channels)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.bn(x)


        
class WaveletFourierBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 小波
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')
        self.idwt = DWTInverse(wave='haar', mode='zero')
        # 幅度增强
        self.amp_enhance = AmplitudeEnhance(channels)
        # 相位注意力
        self.phase_attn = PhaseAttention(channels)
        # 空间增强
        self.spatial_enhance = SpatialEnhance(channels)
    def forward(self, x):
        yl, yh = self.dwt(x)   # yl: 低频, yh: 高频list
        fft = torch.fft.fft2(yl, norm='ortho')
        amp = torch.abs(fft)
        phase = torch.angle(fft)
        amp = self.amp_enhance(amp)
        phase = self.phase_attn(phase)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        fft_new = torch.complex(real, imag)
        yl_new = torch.fft.ifft2(fft_new, norm='ortho').real
        x_recon = self.idwt((yl_new, yh))
        out = self.spatial_enhance(x_recon)
        return out

class StructureFrequencyExpert(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        kernel_size = 7
        padding = (kernel_size-1)//2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )
        self.wavelet_block = WaveletFourierBlock(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        x = self.depthwise(x)
        x = self.wavelet_block(x)
        x = self.pointwise(x)

        # 如果输出尺寸不对，就补齐
        if x.shape[2] != H or x.shape[3] != W:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x


class DepthwiseSeparableConv_Sonar(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()

        if kernel_size == "StructureGeometricExpert":
            self.expert = StructureGeometricExpert(in_channels)             

        elif kernel_size == "EchoTextureExpert":
            self.expert = EchoTextureExpert(in_channels)

        elif kernel_size == "LargeKernelSpatialAttention":
            self.expert = LargeKernelSpatialAttention(in_channels)

        elif kernel_size == "LocalDetailAttention":
            self.expert = LocalDetailAttention(in_channels)
        
        elif kernel_size == "DCTMultiSpectralAttention":
            self.expert = DCTMultiSpectralAttention(in_channels)

        
        elif kernel_size == "ContextRelationExpert":
            self.expert = ContextRelationExpert(in_channels)

        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}")
#LocalDetailAttention
    def forward(self, x):
        return self.expert(x)



class EfficientExpertGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(EfficientExpertGroup, self).__init__()
        self.conv = DepthwiseSeparableConv_Sonar(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if not hasattr(self, "conv"):
            out_c = x.shape[1]
            self.conv = DepthwiseSeparableConv_Sonar(x.shape[1], out_c, 3, 1)
        return self.conv(x)
