"""Run RT-Focuser-V2 on a single image."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from rt_focuser_v2 import load_model


def load_image(path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    height, width = tensor.shape[-2:]
    pad_h = (16 - height % 16) % 16
    pad_w = (16 - width % 16) % 16
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, (height, width)


def save_image(tensor: torch.Tensor, path: Path, original_size: tuple[int, int]) -> None:
    height, width = original_size
    array = tensor[0, :, :height, :width].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(array * 255.0).astype(np.uint8)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="RT-Focuser-V2 image deblurring")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exit", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device=device)
    model.set_exit_level(args.exit)
    image, original_size = load_image(args.input)
    with torch.inference_mode():
        restored = model(image.to(device))
    save_image(restored, args.output, original_size)
    print(f"Saved exit Y{args.exit} result to {args.output}")


if __name__ == "__main__":
    main()
