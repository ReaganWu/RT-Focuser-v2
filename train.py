"""Train the 4.72M RT-Focuser-V2 model on GoPro."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import GoProDataset, RandomPairedCropFlip
from losses import (
    ProgressiveSelfDistillConfig,
    acl_progressive_self_distill_loss,
    compute_psnr,
    compute_ssim,
)
from rt_focuser_v2 import build_model, count_parameters


def train_epoch(model, loader, optimizer, device, epoch: int, epochs: int, cfg) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "psnr_y4": 0.0}
    for step, (blur, sharp) in enumerate(tqdm(loader, desc=f"train {epoch + 1}/{epochs}", leave=False)):
        blur, sharp = blur.to(device), sharp.to(device)
        progress = (epoch + step / max(len(loader), 1)) / max(epochs, 1)
        optimizer.zero_grad(set_to_none=True)
        outputs = model.forward_all(blur)
        loss, _ = acl_progressive_self_distill_loss(outputs, sharp, progress=progress, cfg=cfg)
        loss.backward()
        optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["psnr_y4"] += float(compute_psnr(outputs["exit4"].detach(), sharp).mean())
    return {key: value / len(loader) for key, value in totals.items()}


@torch.inference_mode()
def validate(model, loader, device) -> dict[str, float]:
    model.eval()
    totals = {f"psnr_y{i}": 0.0 for i in range(1, 5)}
    totals.update({f"ssim_y{i}": 0.0 for i in range(1, 5)})
    for blur, sharp in tqdm(loader, desc="validation", leave=False):
        blur, sharp = blur.to(device), sharp.to(device)
        outputs = model.forward_all(blur)
        for i in range(1, 5):
            output = outputs[f"exit{i}"]
            totals[f"psnr_y{i}"] += float(compute_psnr(output, sharp).mean())
            totals[f"ssim_y{i}"] += float(compute_ssim(output, sharp).mean())
    return {key: value / len(loader) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RT-Focuser-V2 on GoPro")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/rt-focuser-v2"))
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_set = GoProDataset(args.data_root, "train", RandomPairedCropFlip(args.crop_size))
    test_set = GoProDataset(args.data_root, "test")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = build_model(exit_level=4).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_cfg = ProgressiveSelfDistillConfig()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    best_psnr = float("-inf")

    print(f"parameters={count_parameters(model):,} device={device}")
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, args.epochs, loss_cfg)
        val_metrics = validate(model, test_loader, device)
        scheduler.step()
        row = {"epoch": epoch + 1, "train": train_metrics, "validation": val_metrics}
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row))

        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch + 1,
            "args": vars(args),
            "loss_cfg": asdict(loss_cfg),
            "validation": val_metrics,
        }
        torch.save(checkpoint, args.output_dir / "latest.pth")
        if val_metrics["psnr_y4"] > best_psnr:
            best_psnr = val_metrics["psnr_y4"]
            torch.save(checkpoint, args.output_dir / "best.pth")


if __name__ == "__main__":
    main()
