"""Small metric helpers shared by training and evaluation."""

from __future__ import annotations

import numpy as np
import torch


def seg_metrics(pred: torch.Tensor, true: torch.Tensor, n_classes: int) -> tuple[float, float]:
    """Cell accuracy and mean IoU over the non-empty materials (empty space dominates accuracy)."""
    acc = (pred == true).float().mean().item()
    ious = []
    for k in range(1, n_classes):
        union = ((pred == k) | (true == k)).sum().item()
        if union:
            ious.append(((pred == k) & (true == k)).sum().item() / union)
    return acc, float(np.mean(ious)) if ious else float("nan")


def embedding_stats(emb: torch.Tensor) -> tuple[float, float]:
    """Mean per-dimension std (target 1) and effective rank (participation ratio of the covariance spectrum)."""
    z = emb.flatten(0, -2)
    z = z - z.mean(0)
    eig = torch.linalg.svdvals(z).square() / (len(z) - 1)
    return z.std(0).mean().item(), (eig.sum().square() / eig.square().sum()).item()


def within_episode_share(x: torch.Tensor) -> float:
    """x: (episodes, frames, D).  Share of the total variance that lies *within* episodes.

    Near 0 the features only say which episode a frame belongs to; larger values mean they track
    what changes while an episode unfolds.  Compare with the same number computed on raw pixels.
    """
    total = x.flatten(0, 1).var(0, unbiased=False).sum()
    return (x.var(1, unbiased=False).mean(0).sum() / total).item()
