"""Random initial scenes and a scripted random brush policy for data collection."""

from __future__ import annotations

import numpy as np

from .materials import EMPTY, STATIC, VOID, WALL, MaterialSet


def random_scene(rng: np.random.Generator, size: int, ms: MaterialSet) -> np.ndarray:
    """A random (size, size) world: wall terrain plus blobs of loose material hanging in the air."""
    if size < 32:
        raise ValueError("random_scene needs size >= 32")
    g = np.zeros((size, size), dtype=np.uint8)
    s = size
    loose = [i for i, k in enumerate(ms.kinds) if k not in (VOID, STATIC)]

    def put(mask_or_slice, material):  # write only into empty cells
        region = g[mask_or_slice]
        region[region == EMPTY] = material
        g[mask_or_slice] = region

    # --- terrain -------------------------------------------------------------------------
    for _ in range(rng.integers(0, 5)):  # horizontal platforms
        length, thick = rng.integers(s // 8, s // 2 + 1), rng.integers(1, 3)
        x0, y0 = rng.integers(1, s - 1 - length), rng.integers(s // 4, s - 3)
        g[y0 : y0 + thick, x0 : x0 + length] = WALL
    for _ in range(rng.integers(0, 3)):  # open-topped cups
        w, h = rng.integers(8, s // 2), rng.integers(4, s // 4 + 1)
        x0, yb = rng.integers(1, s - 1 - w), rng.integers(h + 2, s - 1)
        g[yb, x0 : x0 + w] = WALL
        g[yb - h : yb + 1, x0] = WALL
        g[yb - h : yb + 1, x0 + w - 1] = WALL
    for _ in range(rng.integers(0, 3)):  # diagonal ramps
        length, direction = rng.integers(6, s // 3 + 1), rng.choice([-1, 1])
        x0, y0 = rng.integers(length + 1, s - length - 1), rng.integers(4, s - length - 2)
        i = np.arange(length)
        g[y0 + i, x0 + direction * i] = WALL

    # --- loose material ------------------------------------------------------------------
    if rng.random() < 0.3:  # uneven ground
        heights = np.clip(np.cumsum(rng.integers(-1, 2, size=s)) + rng.integers(1, s // 6), 0, s // 4)
        put(np.arange(s)[:, None] >= s - 1 - heights[None, :], rng.choice(loose))
    if rng.random() < 0.25:  # a pool: flood every empty cell below a level
        put(np.s_[s - 1 - rng.integers(2, s // 5) :, :], rng.choice(loose))
    yy, xx = np.mgrid[:s, :s]
    for _ in range(rng.integers(0, 6)):  # blobs
        r, cx, cy = rng.integers(2, s // 8 + 1), rng.integers(2, s - 2), rng.integers(2, s - 2)
        if rng.random() < 0.5:
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        else:
            ry = rng.integers(1, s // 8 + 1)
            mask = (np.abs(yy - cy) <= ry) & (np.abs(xx - cx) <= r)
        put(mask, rng.choice(loose))
    return g


class StrokePolicy:
    """Scripted random brush for a batch of worlds.

    Alternates idle periods with *strokes*: one tool held for several frames while the cursor
    drifts.  Loose materials are poured from a slowly drifting nozzle; walls and the eraser move
    faster and so draw lines.  Returns actions as (B, 3) int16 arrays of (x, y, tool).
    """

    def __init__(
        self,
        batch: int,
        size: int,
        ms: MaterialSet,
        rng: np.random.Generator,
        *,
        p_idle: float = 0.3,
        p_erase: float = 0.12,
        p_wall: float = 0.18,
        stroke_len: tuple[int, int] = (3, 16),
        idle_len: tuple[int, int] = (2, 10),
    ) -> None:
        self.batch, self.size, self.rng = batch, size, rng
        self.p_idle, self.stroke_len, self.idle_len = p_idle, stroke_len, idle_len
        n_loose = len(ms.tools) - 3  # tools: none, erase, wall, <loose materials...>
        self.tool_probs = np.array([0.0, p_erase, p_wall] + [(1 - p_erase - p_wall) / n_loose] * n_loose)
        self.remaining = np.zeros(batch, dtype=np.int64)
        self.tool = np.zeros(batch, dtype=np.int64)
        self.pos = np.zeros((batch, 2))
        self.vel = np.zeros((batch, 2))

    def act(self) -> np.ndarray:
        rng, lo, hi = self.rng, 1.0, self.size - 2.0
        new = np.flatnonzero(self.remaining <= 0)
        if len(new):
            n = len(new)
            idle = rng.random(n) < self.p_idle
            tool = np.where(idle, 0, rng.choice(len(self.tool_probs), size=n, p=self.tool_probs))
            length = np.where(idle, rng.integers(*self.idle_len, size=n, endpoint=True),
                              rng.integers(*self.stroke_len, size=n, endpoint=True))
            fast = (tool == 1) | (tool == 2)  # eraser and wall draw lines
            speed = np.where(fast, rng.uniform(0.5, 2.0, n), rng.uniform(0.0, 0.8, n))
            angle = rng.uniform(0, 2 * np.pi, n)
            self.tool[new], self.remaining[new] = tool, length
            self.pos[new] = rng.uniform(lo, hi, (n, 2))
            self.vel[new] = speed[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1)
        action = np.concatenate([np.rint(self.pos), self.tool[:, None]], axis=1).astype(np.int16)
        self.pos += self.vel
        bounce = (self.pos < lo) | (self.pos > hi)
        self.vel[bounce] *= -1
        np.clip(self.pos, lo, hi, out=self.pos)
        self.remaining -= 1
        return action
