"""ACL-style multi-exit restoration loss with progressive self-distillation.

This module intentionally keeps the recipe narrow:
- direct GT supervision is L1 + FFT-L1, following ACLNet/MIMO-style training;
- exit4 is the detached self-teacher for exit1/2/3;
- KD terms are enabled progressively so shallow exits are not forced to mimic a
  deep teacher before the final exit has become useful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def compute_psnr(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Return per-image PSNR for NCHW tensors in [0, 1]."""
    mse = (pred.clamp(0, 1) - target.clamp(0, 1)).pow(2).flatten(1).mean(1)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(eps))


def compute_ssim(pred: Tensor, target: Tensor, window_size: int = 11, eps: float = 1e-8) -> Tensor:
    """Return per-image SSIM using an 11x11 uniform window."""
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    pad = window_size // 2
    c1, c2 = 0.01**2, 0.03**2
    mu_x = F.avg_pool2d(pred, window_size, 1, pad)
    mu_y = F.avg_pool2d(target, window_size, 1, pad)
    sigma_x = F.avg_pool2d(pred * pred, window_size, 1, pad) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(target * target, window_size, 1, pad) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(pred * target, window_size, 1, pad) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2) + eps
    )
    return score.flatten(1).mean(1)


def fft_l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """ACLNet-style FFT-L1 on real/imaginary pairs."""
    pred_fft = torch.fft.fft2(pred.float(), dim=(-2, -1))
    target_fft = torch.fft.fft2(target.float(), dim=(-2, -1))
    pred_pair = torch.stack((pred_fft.real, pred_fft.imag), dim=-1)
    target_pair = torch.stack((target_fft.real, target_fft.imag), dim=-1)
    return F.l1_loss(pred_pair, target_pair)


def resize_image(x: Tensor, size: Tuple[int, int]) -> Tensor:
    if x.shape[-2:] == size:
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def scale_image(x: Tensor, scale: float) -> Tensor:
    if float(scale) == 1.0:
        return x
    h, w = x.shape[-2:]
    size = (max(int(round(h * float(scale))), 1), max(int(round(w * float(scale))), 1))
    return resize_image(x, size)


def restoration_l1_fft(pred: Tensor, target: Tensor, fft_weight: float = 0.1) -> Tuple[Tensor, Tensor, Tensor]:
    l1 = F.l1_loss(pred, target)
    fft = fft_l1_loss(pred, target)
    return l1 + float(fft_weight) * fft, l1, fft


def cosine_ramp(progress: float, start: float, end: float) -> float:
    """0 before start, 1 after end, smooth cosine in between."""
    progress = float(progress)
    start = float(start)
    end = float(end)
    if progress <= start:
        return 0.0
    if progress >= end:
        return 1.0
    span = max(end - start, 1e-8)
    p = (progress - start) / span
    return 0.5 * (1.0 - math.cos(math.pi * p))


@dataclass(frozen=True)
class ProgressiveSelfDistillConfig:
    """Loss and schedule defaults for the final clean recipe."""

    fft_weight: float = 0.1
    kd_max_weight: float = 0.2
    # Match the ACLNet hierarchy even when the model returns full-size exits:
    # exit1 is judged at 1/4, exit2 at 1/2, exit3/4 at full resolution.
    exit_scales: Tuple[float, float, float, float] = (0.25, 0.5, 1.0, 1.0)
    supervised_exit_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    kd_exit_weights: Tuple[float, float, float] = (0.25, 0.5, 1.0)
    # Progressive KD windows: exit3 first, then exit2, then exit1.
    exit1_kd_window: Tuple[float, float] = (0.60, 1.00)
    exit2_kd_window: Tuple[float, float] = (0.35, 0.60)
    exit3_kd_window: Tuple[float, float] = (0.15, 0.35)


def progressive_kd_weights(progress: float, cfg: ProgressiveSelfDistillConfig) -> Tuple[float, float, float]:
    windows = (cfg.exit1_kd_window, cfg.exit2_kd_window, cfg.exit3_kd_window)
    return tuple(
        float(cfg.kd_max_weight) * float(exit_weight) * cosine_ramp(progress, start, end)
        for exit_weight, (start, end) in zip(cfg.kd_exit_weights, windows)
    )


def _collect_exits(outputs: Mapping[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    try:
        return tuple(outputs[f"exit{i}"] for i in range(1, 5))  # type: ignore[return-value]
    except KeyError as exc:
        raise KeyError("outputs must contain exit1, exit2, exit3, and exit4") from exc


def acl_progressive_self_distill_loss(
    outputs: Mapping[str, Tensor],
    target: Tensor,
    *,
    progress: float,
    cfg: ProgressiveSelfDistillConfig | None = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute L1+FFT GT loss plus progressive exit4 self-distillation.

    Args:
        outputs: dict containing ``exit1`` through ``exit4``.
        target: sharp target image in NCHW format.
        progress: normalized training progress in [0, 1].
        cfg: optional schedule/loss config.
    """
    if cfg is None:
        cfg = ProgressiveSelfDistillConfig()

    exits = _collect_exits(outputs)
    scaled_exits = [scale_image(y, s) for y, s in zip(exits, cfg.exit_scales)]
    scaled_targets = [resize_image(target, y.shape[-2:]) for y in scaled_exits]

    sup_terms = []
    sup_l1_terms = []
    sup_fft_terms = []
    for weight, pred, gt in zip(cfg.supervised_exit_weights, scaled_exits, scaled_targets):
        total, l1, fft = restoration_l1_fft(pred, gt, fft_weight=cfg.fft_weight)
        sup_terms.append(float(weight) * total)
        sup_l1_terms.append(float(weight) * l1)
        sup_fft_terms.append(float(weight) * fft)
    l_sup = sum(sup_terms)
    l_sup_l1 = sum(sup_l1_terms)
    l_sup_fft = sum(sup_fft_terms)

    teacher = exits[3].detach()
    kd_weights = progressive_kd_weights(progress, cfg)
    kd_terms = []
    kd_l1_terms = []
    kd_fft_terms = []
    for idx, kd_weight in enumerate(kd_weights):
        student = scaled_exits[idx]
        teacher_i = resize_image(teacher, student.shape[-2:])
        total, l1, fft = restoration_l1_fft(student, teacher_i, fft_weight=cfg.fft_weight)
        kd_terms.append(float(kd_weight) * total)
        kd_l1_terms.append(float(kd_weight) * l1)
        kd_fft_terms.append(float(kd_weight) * fft)

    l_kd = sum(kd_terms) if kd_terms else target.new_tensor(0.0)
    l_kd_l1 = sum(kd_l1_terms) if kd_l1_terms else target.new_tensor(0.0)
    l_kd_fft = sum(kd_fft_terms) if kd_fft_terms else target.new_tensor(0.0)
    loss = l_sup + l_kd

    log: Dict[str, float] = {
        "loss_total": float(loss.detach().cpu()),
        "loss_sup": float(l_sup.detach().cpu()),
        "loss_sup_l1": float(l_sup_l1.detach().cpu()),
        "loss_sup_fft": float(l_sup_fft.detach().cpu()),
        "loss_kd": float(l_kd.detach().cpu()),
        "loss_kd_l1": float(l_kd_l1.detach().cpu()),
        "loss_kd_fft": float(l_kd_fft.detach().cpu()),
        "kd_w_exit1": float(kd_weights[0]),
        "kd_w_exit2": float(kd_weights[1]),
        "kd_w_exit3": float(kd_weights[2]),
        "progress": float(progress),
    }
    for i, term in enumerate(sup_terms, start=1):
        log[f"loss_sup_exit{i}"] = float(term.detach().cpu())
    for i, term in enumerate(kd_terms, start=1):
        log[f"loss_kd_exit{i}"] = float(term.detach().cpu())
    return loss, log


__all__ = [
    "ProgressiveSelfDistillConfig",
    "acl_progressive_self_distill_loss",
    "compute_psnr",
    "compute_ssim",
    "cosine_ramp",
    "fft_l1_loss",
    "progressive_kd_weights",
    "restoration_l1_fft",
]
