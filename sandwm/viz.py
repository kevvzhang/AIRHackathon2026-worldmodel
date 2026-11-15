"""Rendering helpers: material grids -> RGB images, contact sheets and GIFs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def render(grid: np.ndarray, palette: np.ndarray, scale: int = 1) -> np.ndarray:
    """Material indices (..., H, W) -> uint8 RGB (..., H*scale, W*scale, 3)."""
    img = palette[np.asarray(grid)]
    if scale > 1:
        img = img.repeat(scale, axis=-3).repeat(scale, axis=-2)
    return img


def hstack(images: list[np.ndarray], pad: int = 4, bg: int = 40) -> np.ndarray:
    """Concatenate (H, W, 3) images left-to-right with padding."""
    h = images[0].shape[0]
    gap = np.full((h, pad, 3), bg, dtype=np.uint8)
    out = []
    for i, im in enumerate(images):
        out += ([gap] if i else []) + [im]
    return np.concatenate(out, axis=1)


def vstack(images: list[np.ndarray], pad: int = 4, bg: int = 40) -> np.ndarray:
    w = images[0].shape[1]
    gap = np.full((pad, w, 3), bg, dtype=np.uint8)
    out = []
    for i, im in enumerate(images):
        out += ([gap] if i else []) + [im]
    return np.concatenate(out, axis=0)


def label_strip(text: str, width: int, height: int = 14, bg: int = 40) -> np.ndarray:
    img = Image.new("RGB", (width, height), (bg, bg, bg))
    ImageDraw.Draw(img).text((3, 1), text, fill=(235, 235, 235))
    return np.asarray(img)


def contact_sheet(rows: dict[str, np.ndarray], palette: np.ndarray, times: list[int], scale: int = 3,
                  labels: list[str] | None = None) -> np.ndarray:
    """One labelled row per entry of ``rows`` (each a (T, H, W) grid sequence), one column per time."""
    labels = labels or [f"t={t}" for t in times]
    strips = []
    for name, seq in rows.items():
        row = hstack([render(seq[t], palette, scale) for t in times])
        strips += [label_strip(name, row.shape[1]), row]
    header = hstack([label_strip(text, rows[next(iter(rows))].shape[-1] * scale) for text in labels])
    return vstack([header] + strips, pad=1)


def save_png(path: str | Path, image: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def save_gif(path: str | Path, frames: list[np.ndarray] | np.ndarray, fps: int = 12) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)
