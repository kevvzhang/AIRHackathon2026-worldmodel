"""Dataset storage and sampling.

On disk a dataset is a directory::

    meta.json                 materials, grid size, simulator settings
    train/shard-0000.npz      frames  (N, T+1, H, W) uint8  material indices
    val/shard-0000.npz        actions (N, T, 3)      int16  (x, y, tool);  actions[t]: frames[t] -> frames[t+1]

Material grids are stored rather than rendered pixels: they are lossless, tiny, and can be turned
into RGB observations (model input) or class targets (decoder / metrics) on the fly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .materials import MATERIAL_SETS


def save_shard(path: Path, frames: np.ndarray, actions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frames=frames, actions=actions)


class EpisodeDataset:
    """One split of a dataset, held in RAM as uint8 (64x64 frames cost 4 KB each)."""

    def __init__(self, root: str | Path, split: str = "train", max_episodes: int | None = None) -> None:
        root = Path(root)
        self.meta = json.loads((root / "meta.json").read_text())
        self.materials = MATERIAL_SETS[self.meta["materials"]]
        shards = sorted((root / split).glob("shard-*.npz"))
        if not shards:
            raise FileNotFoundError(f"no shards in {root / split}")
        frames, actions = [], []
        for path in shards:
            with np.load(path) as z:
                frames.append(z["frames"])
                actions.append(z["actions"])
            if max_episodes is not None and sum(map(len, frames)) >= max_episodes:
                break
        self.frames = np.concatenate(frames)[:max_episodes]
        self.actions = np.concatenate(actions)[:max_episodes]
        self.size = self.frames.shape[-1]
        self.n_tools = len(self.materials.tools)
        self.action_dim = 2 + self.n_tools  # per raw frame: x, y, one-hot tool

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def episode_len(self) -> int:
        """Number of actions per episode (frames per episode is one more)."""
        return self.actions.shape[1]

    # ------------------------------------------------------------------------------------
    def window(self, episode: np.ndarray, start: np.ndarray, n_frames: int, frame_skip: int = 1):
        """Gather windows of ``n_frames`` frames taken every ``frame_skip`` raw frames.

        Returns (frames (B, n, H, W) uint8, actions (B, n-1, frame_skip, 3) int16): the block of
        raw actions between consecutive returned frames.
        """
        episode, start = np.asarray(episode)[:, None], np.asarray(start)[:, None]
        t = start + frame_skip * np.arange(n_frames)
        frames = self.frames[episode, t]
        ta = t[:, :-1, None] + np.arange(frame_skip)
        actions = self.actions[episode[:, :, None], ta]
        return frames, actions

    def sample(self, batch: int, n_frames: int, frame_skip: int, rng: np.random.Generator):
        span = (n_frames - 1) * frame_skip
        if span > self.episode_len:
            raise ValueError(f"window of {span} steps does not fit in episodes of {self.episode_len}")
        episode = rng.integers(0, len(self), size=batch)
        start = rng.integers(0, self.episode_len - span + 1, size=batch)
        return self.window(episode, start, n_frames, frame_skip)

    # --- conversion to model inputs ---------------------------------------------------------
    def observe(self, frames: np.ndarray, device: torch.device | str = "cpu") -> torch.Tensor:
        """Material grids (..., H, W) -> RGB observations (..., 3, H, W) in [-1, 1]."""
        return observe(torch.from_numpy(np.ascontiguousarray(frames)).to(device), self.materials.palette)

    def encode_actions(self, actions: np.ndarray, device: torch.device | str = "cpu") -> torch.Tensor:
        """Raw (..., frame_skip, 3) action blocks -> float (..., frame_skip * action_dim)."""
        return encode_actions(torch.from_numpy(np.ascontiguousarray(actions)).to(device), self.size, self.n_tools)


def observe(grid: torch.Tensor, palette: np.ndarray) -> torch.Tensor:
    pal = torch.as_tensor(palette, dtype=torch.float32, device=grid.device) / 127.5 - 1.0
    return pal[grid.long()].movedim(-1, -3)


def encode_actions(actions: torch.Tensor, size: int, n_tools: int) -> torch.Tensor:
    """(x, y, tool) -> [x, y in [-1, 1], one-hot(tool)], flattened over the frame-skip block.

    The cursor position is zeroed when the tool is 'none', so that doing nothing is one single
    action vector rather than a family of meaningless ones.
    """
    xy = actions[..., :2].float() / (size - 1) * 2 - 1
    tool = actions[..., 2].long()
    xy = xy * (tool > 0).unsqueeze(-1)
    out = torch.cat([xy, torch.nn.functional.one_hot(tool, n_tools).float()], dim=-1)
    return out.flatten(-2)
