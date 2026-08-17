#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Official RT-Focuser-V2 model definition.

This variant intentionally follows the clean RT-Focuser-v1 data flow:

    input -> encoder pyramid -> four cross-scale fusion maps -> U-shaped decoder

The main change is the fusion block. Instead of transformer-style MHA, E-GAM
uses convolutional projection plus MobileViTv2-style linear self-attention:
Q is a one-channel spatial score map, K/V are feature maps, and global context is
formed with element-wise multiply and reduction.  There is no QK^T attention
matrix, no multi-head matmul, and no LayerNorm.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def make_activation(name: str = "relu", inplace: bool = False) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "relu6":
        return nn.ReLU6(inplace=inplace)
    if name == "hardswish":
        return nn.Hardswish(inplace=inplace)
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    if name == "gelu":
        return nn.GELU()
    if name in ("identity", "none", "linear"):
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


class conv_block(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, act: str = "relu"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            make_activation(act),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, act: str = "relu", mode: str = "nearest"):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            make_activation(act),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.up(x)


class SN_Module(nn.Module):
    """Depthwise learnable sharpen branch from RT-Focuser-v1."""

    def __init__(self, channels: int, kernel_size: int = 3, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.kernel = nn.Parameter(torch.zeros(self.channels, 1, self.kernel_size, self.kernel_size))
        laplacian = torch.tensor(
            [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
            dtype=torch.float32,
        )
        self.kernel.data = laplacian.repeat(self.channels, 1, 1, 1) / float(self.kernel_size**2)
        self.bn = nn.BatchNorm2d(self.channels, eps, momentum, affine)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.bn.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        sharpened = F.conv2d(x, self.kernel, padding=self.padding, groups=self.channels)
        return self.bn(sharpened)


class Residual(nn.Module):
    def __init__(self, fn: nn.Module, ch_in: int, use_sharpen: bool = True):
        super().__init__()
        self.fn = fn
        self.denoiser = SN_Module(channels=ch_in) if use_sharpen else None

    def forward(self, x: Tensor) -> Tensor:
        y = self.fn(x) + x
        if self.denoiser is not None:
            y = y + self.denoiser(x)
        return y


class LD_Block(nn.Module):
    """RT-Focuser-v1 LD block with configurable activation.

    ReLU is the default for MobileGAM because it maps to a very simple Core ML
    activation and avoids GELU's extra approximation work.
    """

    def __init__(self, ch_in: int, ch_out: int, depth: int = 1, k: int = 3, act: str = "relu", use_sharpen: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                            make_activation(act),
                            nn.BatchNorm2d(ch_in),
                        ),
                        ch_in=ch_in,
                        use_sharpen=use_sharpen,
                    ),
                    nn.Conv2d(ch_in, ch_in * 4, kernel_size=1),
                    make_activation(act),
                    nn.BatchNorm2d(ch_in * 4),
                    nn.Conv2d(ch_in * 4, ch_in, kernel_size=1),
                    make_activation(act),
                    nn.BatchNorm2d(ch_in),
                )
                for _ in range(depth)
            ]
        )
        self.up = conv_block(ch_in, ch_out, act=act)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.block(x))


class MobileLinearAttention2d(nn.Module):
    """MobileViTv2-style linear attention for 2D feature maps.

    This keeps QKV as 1x1 convolutions but avoids QK^T.  Q is a single-channel
    spatial score map, K/V stay channels-first feature maps, and the global
    context vector is produced with element-wise multiply plus spatial sum.
    """

    def __init__(self, channels: int, act: str = "relu"):
        super().__init__()
        self.channels = int(channels)
        self.qkv_proj = nn.Conv2d(channels, 1 + 2 * channels, kernel_size=1, bias=True)
        self.value_act = make_activation(act)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, [1, self.channels, self.channels], dim=1)
        score = F.softmax(q.reshape(b, 1, h * w), dim=-1).reshape(b, 1, h, w)
        context = torch.sum(k * score, dim=(2, 3), keepdim=True)
        out = self.value_act(v) * context
        return self.out_proj(out)


class MobileGAMFusion(nn.Module):
    """V1-style multi-scale fusion plus mobile-friendly global modulation."""

    def __init__(self, in_channels_list: Sequence[int], out_channels: int, act: str = "relu", use_attention: bool = True):
        super().__init__()
        self.in_channels_list = tuple(int(c) for c in in_channels_list)
        self.out_channels = int(out_channels)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, self.out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.out_channels),
                    make_activation(act),
                )
                for in_ch in self.in_channels_list
            ]
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.out_channels * len(self.in_channels_list), self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            make_activation(act),
        )
        hidden = max(self.out_channels // 4, 1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channels, hidden, kernel_size=1, bias=False),
            make_activation(act),
            nn.Conv2d(hidden, self.out_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.use_attention = bool(use_attention)
        self.mobile_attention = MobileLinearAttention2d(self.out_channels, act=act) if self.use_attention else nn.Identity()
        self.attn_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, inputs: Sequence[Tensor]) -> Tensor:
        target_size = inputs[-1].shape[-2:]
        projected: List[Tensor] = []
        for feat, branch in zip(inputs, self.branches):
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="nearest")
            projected.append(branch(feat))
        fused = self.fusion_conv(torch.cat(projected, dim=1))
        fused = fused * self.channel_attention(fused)
        if self.use_attention:
            fused = fused + self.attn_scale * self.mobile_attention(fused)
        return fused


class ProgressiveMobileGAMFuse(nn.Module):
    """Fuse a lower-resolution state with the next higher-resolution skip.

    The output resolution follows the skip feature.  This creates a clear
    progressive path: x4+x3 -> m3, m3+x2 -> m2, and m2+x1 -> m1.
    """

    def __init__(self, low_channels: int, skip_channels: int, out_channels: int, act: str = "relu", use_attention: bool = True):
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            make_activation(act),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            make_activation(act),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            make_activation(act),
        )
        hidden = max(out_channels // 4, 1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, hidden, kernel_size=1, bias=False),
            make_activation(act),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.use_attention = bool(use_attention)
        self.mobile_attention = MobileLinearAttention2d(out_channels, act=act) if self.use_attention else nn.Identity()
        self.attn_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, low: Tensor, skip: Tensor) -> Tensor:
        if low.shape[-2:] != skip.shape[-2:]:
            low = F.interpolate(low, size=skip.shape[-2:], mode="nearest")
        fused = self.mix(torch.cat([self.low_proj(low), self.skip_proj(skip)], dim=1))
        fused = fused * self.channel_attention(fused)
        if self.use_attention:
            fused = fused + self.attn_scale * self.mobile_attention(fused)
        return fused


class PyramidExitHead(nn.Module):
    """Predict a full-size residual image from a pyramid feature."""

    def __init__(self, channels: int, act: str = "relu", residual_scale: float = 1.0):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True),
            make_activation(act),
            nn.Conv2d(channels, 3, kernel_size=1, bias=True),
        )

    def forward(self, feat: Tensor, image: Tensor) -> Tensor:
        image_low = F.interpolate(image, size=feat.shape[-2:], mode="nearest")
        residual = self.residual_scale * torch.tanh(self.head(feat))
        y_low = torch.sigmoid(image_low + residual)
        if y_low.shape[-2:] != image.shape[-2:]:
            y_low = F.interpolate(y_low, size=image.shape[-2:], mode="nearest")
        return y_low


class SPPF(nn.Module):
    """YOLO-style Spatial Pyramid Pooling Fast.

    Placed at the x4/f4 bottleneck so every exit can receive the additional
    multi-scale context, including exit1.
    """

    def __init__(self, ch_in: int, ch_out: int, k: int = 5, act: str = "silu"):
        super().__init__()
        hidden = max(int(ch_in) // 2, 1)
        self.cv1 = nn.Sequential(
            nn.Conv2d(ch_in, hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden),
            make_activation(act),
        )
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = nn.Sequential(
            nn.Conv2d(hidden * 4, ch_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_out),
            make_activation(act),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class XFuse_Block(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, act: str = "relu"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            make_activation(act),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=1),
            make_activation(act),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=1),
            make_activation(act),
            nn.BatchNorm2d(ch_out),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_out + 3, ch_out, kernel_size=3, stride=1, padding=1, groups=1, bias=True),
            make_activation(act),
            nn.BatchNorm2d(ch_out),
            nn.Conv2d(ch_out, ch_out * 4, kernel_size=1),
            make_activation(act),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=1),
            make_activation(act),
            nn.BatchNorm2d(ch_out),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.conv(x)
        x = torch.cat((x, skip), dim=1)
        return self.conv2(x)


class RT_Focuser_MobileGAM(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        dims: Sequence[int] = (16, 32, 128, 160, 256),
        depths: Sequence[int] = (3, 4, 4, 3, 2),
        kernels: Sequence[int] = (3, 3, 7, 7, 7),
        act: str = "relu",
        upsample_mode: str = "nearest",
        use_sharpen: bool = True,
        use_mobile_attention: bool = True,
        exit_level: int = 4,
        use_sppf: bool = False,
        sppf_activation: str = "silu",
    ):
        super().__init__()
        if int(exit_level) not in (1, 2, 3, 4):
            raise ValueError(f"exit_level must be 1, 2, 3, or 4; got {exit_level}")
        if len(dims) != 5 or len(depths) != 5 or len(kernels) != 5:
            raise ValueError("dims, depths, and kernels must each have length 5")
        d = tuple(int(x) for x in dims)
        dep = tuple(int(x) for x in depths)
        ker = tuple(int(x) for x in kernels)
        self.dims = d
        self.act = act
        self.exit_level = int(exit_level)
        self.use_sppf = bool(use_sppf)

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=d[0], act=act)
        self.encoder1 = LD_Block(ch_in=d[0], ch_out=d[0], depth=dep[0], k=ker[0], act=act, use_sharpen=use_sharpen)
        self.encoder2 = LD_Block(ch_in=d[0], ch_out=d[1], depth=dep[1], k=ker[1], act=act, use_sharpen=use_sharpen)
        self.encoder3 = LD_Block(ch_in=d[1], ch_out=d[2], depth=dep[2], k=ker[2], act=act, use_sharpen=use_sharpen)
        self.encoder4 = LD_Block(ch_in=d[2], ch_out=d[3], depth=dep[3], k=ker[3], act=act, use_sharpen=use_sharpen)
        self.encoder5 = LD_Block(ch_in=d[3], ch_out=d[4], depth=dep[4], k=ker[4], act=act, use_sharpen=use_sharpen)

        self.sppf = SPPF(d[3], d[3], k=5, act=sppf_activation) if self.use_sppf else nn.Identity()
        self.f4_refine = nn.Sequential(
            nn.Conv2d(d[3], d[3], kernel_size=3, padding=1, groups=2, bias=True),
            make_activation(act),
            nn.BatchNorm2d(d[3]),
            nn.Conv2d(d[3], d[3], kernel_size=1, bias=True),
            nn.BatchNorm2d(d[3]),
        )
        self.fuse3 = ProgressiveMobileGAMFuse(d[3], d[2], d[2], act=act, use_attention=use_mobile_attention)
        self.fuse2 = ProgressiveMobileGAMFuse(d[2], d[1], d[1], act=act, use_attention=use_mobile_attention)
        self.fuse1 = ProgressiveMobileGAMFuse(d[1], d[0], d[0], act=act, use_attention=use_mobile_attention)

        self.exit1_head = PyramidExitHead(d[2], act=act, residual_scale=1.0)
        self.exit2_head = PyramidExitHead(d[1], act=act, residual_scale=1.0)
        self.exit3_head = PyramidExitHead(d[0], act=act, residual_scale=1.0)

        self.Up5 = up_conv(ch_in=d[4], ch_out=d[3], act=act, mode=upsample_mode)
        self.Up_conv5 = XFuse_Block(ch_in=d[3] * 2, ch_out=d[3], act=act)
        self.Up4 = up_conv(ch_in=d[3], ch_out=d[2], act=act, mode=upsample_mode)
        self.Up_conv4 = XFuse_Block(ch_in=d[2] * 2, ch_out=d[2], act=act)
        self.Up3 = up_conv(ch_in=d[2], ch_out=d[1], act=act, mode=upsample_mode)
        self.Up_conv3 = XFuse_Block(ch_in=d[1] * 2, ch_out=d[1], act=act)
        self.Up2 = up_conv(ch_in=d[1], ch_out=d[0], act=act, mode=upsample_mode)
        self.Up_conv2 = XFuse_Block(ch_in=d[0] * 2, ch_out=d[0], act=act)
        self.Conv_1x1 = nn.Conv2d(d[0], 3, kernel_size=1, stride=1, padding=0)

    def set_exit_level(self, exit_level: int) -> None:
        if int(exit_level) not in (1, 2, 3, 4):
            raise ValueError(f"exit_level must be 1, 2, 3, or 4; got {exit_level}")
        self.exit_level = int(exit_level)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x1 = self.encoder1(self.stem(x))
        x2 = self.encoder2(self.Maxpool(x1))
        x3 = self.encoder3(self.Maxpool(x2))
        x4 = self.encoder4(self.Maxpool(x3))
        x5 = self.encoder5(self.Maxpool(x4))
        return x1, x2, x3, x4, x5

    def fuse(self, x1: Tensor, x2: Tensor, x3: Tensor, x4: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        z4 = self.sppf(x4)
        f4 = x4 + self.f4_refine(z4)
        m3 = self.fuse3(f4, x3)
        m2 = self.fuse2(m3, x2)
        m1 = self.fuse1(m2, x1)
        return m1, m2, m3, f4

    def decode(self, x: Tensor, x5: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor) -> Tensor:
        o1 = x
        o2 = F.interpolate(o1, scale_factor=0.5, mode="nearest")
        o3 = F.interpolate(o2, scale_factor=0.5, mode="nearest")
        o4 = F.interpolate(o3, scale_factor=0.5, mode="nearest")

        d5 = self.Up_conv5(torch.cat((f4, self.Up5(x5)), dim=1), o4)
        d4 = self.Up_conv4(torch.cat((f3, self.Up4(d5)), dim=1), o3)
        d3 = self.Up_conv3(torch.cat((f2, self.Up3(d4)), dim=1), o2)
        d2 = self.Up_conv2(torch.cat((f1, self.Up2(d3)), dim=1), o1)
        return torch.sigmoid(self.Conv_1x1(d2) + o1)

    def forward_exit(self, x: Tensor, exit_level: int) -> Tensor:
        x1 = self.encoder1(self.stem(x))
        x2 = self.encoder2(self.Maxpool(x1))
        x3 = self.encoder3(self.Maxpool(x2))
        x4 = self.encoder4(self.Maxpool(x3))

        z4 = self.sppf(x4)
        f4 = x4 + self.f4_refine(z4)
        m3 = self.fuse3(f4, x3)
        if int(exit_level) == 1:
            return self.exit1_head(m3, x)

        m2 = self.fuse2(m3, x2)
        if int(exit_level) == 2:
            return self.exit2_head(m2, x)

        m1 = self.fuse1(m2, x1)
        if int(exit_level) == 3:
            return self.exit3_head(m1, x)

        x5 = self.encoder5(self.Maxpool(x4))
        return self.decode(x, x5, m1, m2, m3, f4)

    def forward_all(self, x: Tensor) -> Dict[str, Tensor]:
        x1 = self.encoder1(self.stem(x))
        x2 = self.encoder2(self.Maxpool(x1))
        x3 = self.encoder3(self.Maxpool(x2))
        x4 = self.encoder4(self.Maxpool(x3))
        z4 = self.sppf(x4)
        f4 = x4 + self.f4_refine(z4)
        m3 = self.fuse3(f4, x3)
        y1 = self.exit1_head(m3, x)
        m2 = self.fuse2(m3, x2)
        y2 = self.exit2_head(m2, x)
        m1 = self.fuse1(m2, x1)
        y3 = self.exit3_head(m1, x)
        x5 = self.encoder5(self.Maxpool(x4))
        y4 = self.decode(x, x5, m1, m2, m3, f4)
        return {"exit1": y1, "exit2": y2, "exit3": y3, "exit4": y4, "out": y4}

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_exit(x, self.exit_level)


def RT_Focuser_MobileGAM_Standard(
    dims: Sequence[int] = (16, 32, 128, 160, 256),
    depths: Sequence[int] = (3, 4, 4, 3, 2),
    kernels: Sequence[int] = (3, 3, 7, 7, 7),
    act: str = "relu",
    upsample_mode: str = "nearest",
    use_sharpen: bool = True,
    use_mobile_attention: bool = True,
    exit_level: int = 4,
    use_sppf: bool = False,
    sppf_activation: str = "silu",
) -> RT_Focuser_MobileGAM:
    return RT_Focuser_MobileGAM(
        dims=dims,
        depths=depths,
        kernels=kernels,
        act=act,
        upsample_mode=upsample_mode,
        use_sharpen=use_sharpen,
        use_mobile_attention=use_mobile_attention,
        exit_level=exit_level,
        use_sppf=use_sppf,
        sppf_activation=sppf_activation,
    )


def load_model(checkpoint: Union[str, Path], device: Union[str, torch.device] = "cpu", strict: bool = True) -> RT_Focuser_MobileGAM:
    """Load the standard model from a state dict or training checkpoint."""
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must contain a model state dictionary")
    state = {
        key.removeprefix("module.").removeprefix("model."): value
        for key, value in state.items()
    }
    model = RT_Focuser_MobileGAM_Standard().to(device)
    model.load_state_dict(state, strict=strict)
    model.eval()
    return model


# Public names used by the release examples.
RTFocuserV2 = RT_Focuser_MobileGAM
build_model = RT_Focuser_MobileGAM_Standard


@torch.no_grad()
def benchmark_torch(model: nn.Module, x: Tensor, warmup: int = 10, runs: int = 50) -> Dict[str, float]:
    for _ in range(warmup):
        _ = model(x)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = model(x)
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def export_onnx(model: nn.Module, x: Tensor, path: Path, opset: int = 18) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    external_data = Path(str(path) + ".data")
    if external_data.exists():
        external_data.unlink()
    torch.onnx.export(
        model,
        x,
        str(path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
        external_data=False,
    )


def benchmark_onnx(path: Path, x: Tensor, provider: str, warmup: int, runs: int) -> Dict[str, object]:
    import numpy as np
    import onnxruntime as ort

    available = ort.get_available_providers()
    if provider not in available:
        return {"error": f"{provider} is not available", "available_providers": available}
    try:
        sess = ort.InferenceSession(str(path), providers=[provider, "CPUExecutionProvider"])
        inp = {sess.get_inputs()[0].name: x.detach().cpu().numpy().astype(np.float32)}
        for _ in range(warmup):
            sess.run(None, inp)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            sess.run(None, inp)
            times.append((time.perf_counter() - start) * 1000.0)
        return {
            "providers": sess.get_providers(),
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "available_providers": available}


def export_coreml(model: nn.Module, x: Tensor, path: Path, compute_unit: str = "ALL") -> None:
    import coremltools as ct

    path.parent.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(model, x)
    try:
        traced = torch.jit.freeze(traced.eval())
    except Exception:
        pass
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=x.shape)],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",
        compute_units=getattr(ct.ComputeUnit, compute_unit),
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.save(str(path))


def _main() -> None:
    parser = argparse.ArgumentParser(description="RT-Focuser-V2 export and benchmark")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out-dir", default="exports/rt-focuser-mobilegam")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--act", default="relu", choices=["relu", "relu6", "hardswish", "silu", "gelu"])
    parser.add_argument("--exit", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument("--upsample-mode", default="nearest", choices=["nearest", "bilinear"])
    parser.add_argument("--no-sharpen", action="store_true")
    parser.add_argument("--no-mobile-attention", action="store_true")
    parser.add_argument("--use-sppf", action="store_true")
    parser.add_argument("--sppf-act", default="silu", choices=["relu", "relu6", "hardswish", "silu", "gelu"])
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--skip-coreml", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    out_dir = Path(args.out_dir)
    x = torch.rand(1, 3, args.height, args.width, dtype=torch.float32)
    if args.checkpoint:
        model = load_model(args.checkpoint, device="cpu", strict=True)
    else:
        model = RT_Focuser_MobileGAM_Standard(
            act=args.act,
            upsample_mode=args.upsample_mode,
            use_sharpen=not args.no_sharpen,
            use_mobile_attention=not args.no_mobile_attention,
            exit_level=args.exit,
            use_sppf=args.use_sppf,
            sppf_activation=args.sppf_act,
        )
    model.set_exit_level(args.exit)
    model.eval()

    report: Dict[str, object] = {
        "variant": "RT-Focuser-V2",
        "checkpoint": args.checkpoint,
        "input_shape": list(x.shape),
        "torch_num_threads": torch.get_num_threads(),
        "params": count_parameters(model),
        "activation": args.act,
        "exit_level": args.exit,
        "upsample_mode": args.upsample_mode,
        "use_sharpen": not args.no_sharpen,
        "use_mobile_attention": not args.no_mobile_attention,
        "use_sppf": args.use_sppf,
        "sppf_activation": args.sppf_act if args.use_sppf else None,
        "pytorch_cpu": benchmark_torch(model, x, warmup=args.warmup, runs=args.runs),
    }

    onnx_path = out_dir / "rt-focuser-mobilegam.onnx"
    mlpackage_path = out_dir / "rt-focuser-mobilegam.mlpackage"
    if not args.skip_onnx:
        if not args.benchmark_only or not onnx_path.exists():
            export_onnx(model, x, onnx_path, opset=args.opset)
        report["onnx_path"] = str(onnx_path)
        report["onnx_cpu"] = benchmark_onnx(onnx_path, x, "CPUExecutionProvider", args.warmup, args.runs)
        report["onnx_coreml_ep"] = benchmark_onnx(onnx_path, x, "CoreMLExecutionProvider", args.warmup, args.runs)

    if not args.skip_coreml:
        try:
            if not args.benchmark_only or not mlpackage_path.exists():
                export_coreml(model, x, mlpackage_path)
            report["coreml_path"] = str(mlpackage_path)
        except Exception as exc:
            report["coreml_error"] = f"{type(exc).__name__}: {exc}"

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    _main()
