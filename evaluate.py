"""Evaluate one fixed RT-Focuser-V2 exit on the GoPro test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import GoProDataset
from losses import compute_psnr, compute_ssim
from rt_focuser_v2 import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RT-Focuser-V2 on GoPro")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--exit", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device=device)
    model.set_exit_level(args.exit)
    dataset = GoProDataset(args.data_root, "test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    with torch.inference_mode():
        for blur, sharp in tqdm(loader, desc=f"GoPro Y{args.exit}"):
            blur, sharp = blur.to(device), sharp.to(device)
            output = model(blur)
            psnr_values.append(float(compute_psnr(output, sharp).item()))
            ssim_values.append(float(compute_ssim(output, sharp).item()))

    print(f"images={len(dataset)} exit=Y{args.exit}")
    print(f"PSNR={sum(psnr_values) / len(psnr_values):.4f} dB")
    print(f"SSIM={sum(ssim_values) / len(ssim_values):.6f}")


if __name__ == "__main__":
    main()
