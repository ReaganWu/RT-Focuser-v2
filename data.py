"""GoPro paired-image dataset used by RT-Focuser-V2."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


TensorPairTransform = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


def _read_rgb(path: Path) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()


class RandomPairedCropFlip:
    def __init__(self, size: int = 256, horizontal_flip: bool = True):
        self.size = int(size)
        self.horizontal_flip = bool(horizontal_flip)

    def __call__(self, blur: torch.Tensor, sharp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, height, width = blur.shape
        if height < self.size or width < self.size:
            raise ValueError(f"Image {(height, width)} is smaller than crop size {self.size}")
        top = random.randint(0, height - self.size)
        left = random.randint(0, width - self.size)
        blur = blur[:, top : top + self.size, left : left + self.size]
        sharp = sharp[:, top : top + self.size, left : left + self.size]
        if self.horizontal_flip and random.random() < 0.5:
            blur = torch.flip(blur, dims=(-1,))
            sharp = torch.flip(sharp, dims=(-1,))
        return blur.contiguous(), sharp.contiguous()


class GoProDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Read ``<root>/<split>/<sequence>/{blur_gamma,sharp}/*`` pairs."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Optional[TensorPairTransform] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"GoPro split directory not found: {split_dir}")

        self.pairs: list[tuple[Path, Path]] = []
        for blur_path in sorted(split_dir.glob("*/blur_gamma/*")):
            sharp_path = blur_path.parent.parent / "sharp" / blur_path.name
            if sharp_path.is_file():
                self.pairs.append((blur_path, sharp_path))
        if not self.pairs:
            raise RuntimeError(f"No blur/sharp pairs found under {split_dir}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        blur_path, sharp_path = self.pairs[index]
        blur, sharp = _read_rgb(blur_path), _read_rgb(sharp_path)
        if self.transform is not None:
            blur, sharp = self.transform(blur, sharp)
        return blur, sharp


__all__ = ["GoProDataset", "RandomPairedCropFlip"]
