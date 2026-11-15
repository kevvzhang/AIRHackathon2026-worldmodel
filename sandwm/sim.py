"""Batched falling-sand simulator (Margolus block cellular automaton).

The grid is cut into 2x2 blocks and every block is updated independently by `block_rule`.
Shifting the block partition between sub-steps lets matter travel across block boundaries.
Because blocks never overlap there are no move conflicts, every move is a swap (so mass is
conserved exactly), and a whole batch of worlds advances with a single table lookup.

One *frame* is one full cycle of the four partition offsets, which makes the frame-to-frame
dynamics time-homogeneous: the next frame depends only on (current frame, action, noise).
Within a frame a free-falling grain drops 4 cells and every cell gets one chance to move in
each direction.

The physics lives entirely in `block_rule`, a plain scalar function that is evaluated once for
every possible block to build the lookup table.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np

from .materials import BASIC, GAS, LIQUID, POWDER, STATIC, VOID, WALL, MaterialSet

# (row, col) offsets of the block partition for the four sub-steps of a frame.  Rows alternate
# 0,1,0,1 so falling matter moves every sub-step; over the cycle each cell visits all four block
# positions, which removes the left/right bias of the classic two-partition scheme.
OFFSETS = ((0, 0), (1, 1), (0, 1), (1, 0))

N_BITS = 4  # random bits per block: sink_left, sink_right, slide, spread


def block_rule(ms: MaterialSet, a: int, b: int, c: int, d: int, sink_l: bool, sink_r: bool, slide: bool, spread: bool):
    """Update one block.  Cell layout:   a b   (top row)
                                         c d   (bottom row)
    Returns the new (a, b, c, d).  Only swaps are performed.
    """
    kind, dens = ms.kinds, ms.densities

    def yields(y):  # y can be pushed out of the way
        return kind[y] in (VOID, LIQUID, GAS)

    def F(x, y):  # x may take y's place: x is mobile, y yields, and x is denser
        return kind[x] != STATIC and yields(y) and dens[x] > dens[y]

    def G(x, y):  # as F, but x must be a fluid too (used for gas motion)
        return yields(x) and yields(y) and dens[x] > dens[y]

    def heavy(x):
        return kind[x] in (POWDER, LIQUID)

    # 1. Gravity.  Free fall into empty space always happens; pushing through a medium (sand
    #    sinking in water, gas bubbling up through anything) happens with probability sink_prob.
    if F(a, c) and (kind[c] == VOID or sink_l):
        a, c = c, a
    if F(b, d) and (kind[d] == VOID or sink_r):
        b, d = d, b

    # 2. Diagonal slide.  Powders and liquids that cannot fall roll off a ledge if both the side
    #    cell and the diagonal cell are free (so 1-pixel diagonal walls do not leak).  Gases do
    #    the mirror image upwards.
    if slide:
        if heavy(a) and not F(a, c) and F(a, b) and F(a, d):
            a, d = d, a
        elif heavy(b) and not F(b, d) and F(b, a) and F(b, c):
            b, c = c, b
        if kind[c] == GAS and not F(a, c) and G(b, c) and G(d, c):
            c, b = b, c
        elif kind[d] == GAS and not F(b, d) and G(a, d) and G(c, d):
            d, a = a, d

    # 3. Sideways spread.  A liquid resting on something flows sideways (top row only: that is
    #    where the block can see that it is supported).  Gases trapped under a ceiling likewise.
    if spread:
        if kind[a] == LIQUID and not F(a, c) and F(a, b):
            a, b = b, a
        elif kind[b] == LIQUID and not F(b, d) and F(b, a):
            a, b = b, a
        if kind[c] == GAS and not F(a, c) and G(d, c):
            c, d = d, c
        elif kind[d] == GAS and not F(b, d) and G(c, d):
            c, d = d, c

    return a, b, c, d


@lru_cache(maxsize=None)
def build_rule_table(ms: MaterialSet) -> np.ndarray:
    """Tabulate `block_rule` for every block state and random-bit combination.

    Returns a (2**N_BITS * K**4, 4) uint8 array; the row index is
    ``a + K*b + K**2*c + K**3*d + K**4*bits`` and the row holds the new (a, b, c, d).
    """
    K = len(ms)
    table = np.zeros(((1 << N_BITS) * K**4, 4), dtype=np.uint8)
    for bits in range(1 << N_BITS):
        flags = [bool(bits >> i & 1) for i in range(N_BITS)]
        base = bits * K**4
        # product() varies the last element fastest, so iterate (d, c, b, a) to get the index order
        for i, (d, c, b, a) in enumerate(product(range(K), repeat=4)):
            table[base + i] = block_rule(ms, a, b, c, d, *flags)
    table.setflags(write=False)
    return table


class SandSim:
    """A batch of independent falling-sand worlds.

    ``grid`` is a (B, H, W) uint8 array of material indices with a one-cell wall border.

    Actions are integer arrays of shape (B, 3) holding ``(x, y, tool)``: a square brush of
    radius ``brush_radius`` centred on column x, row y.  Tool 0 does nothing, tool 1 erases,
    and tool k >= 2 paints material k-1 into *empty* cells only.  The border cannot be edited.
    """

    def __init__(
        self,
        batch: int,
        size: int = 64,
        materials: MaterialSet = BASIC,
        *,
        stochastic: bool = True,
        sink_prob: float = 0.5,
        slide_prob: float = 1.0,
        spread_prob: float = 0.9,
        brush_radius: int = 1,
        seed: int | None = 0,
    ) -> None:
        if size % 2 or size < 8:
            raise ValueError("size must be even and >= 8")
        self.batch, self.size, self.materials = batch, size, materials
        self.stochastic = stochastic
        self.probs = (sink_prob, sink_prob, slide_prob, spread_prob) if stochastic else (1.0,) * N_BITS
        self.brush_radius = brush_radius
        self.rng = np.random.default_rng(seed)

        self._K4 = len(materials) ** 4
        self._table = build_rule_table(materials)
        self._all_bits = ((1 << N_BITS) - 1) * self._K4
        self._bits_lut = self._make_bits_lut(self.probs) * self._K4
        self._needs_rng = any(p < 1.0 for p in self.probs)

        coords = np.arange(size)
        self._yy, self._xx = coords[None, :, None], coords[None, None, :]
        self._interior = ((self._yy > 0) & (self._yy < size - 1) & (self._xx > 0) & (self._xx < size - 1))

        self.grid = np.zeros((batch, size, size), dtype=np.uint8)
        self.reset()

    @staticmethod
    def _make_bits_lut(probs) -> np.ndarray:
        """Map a uniform uint16 to a bit pattern whose bit i is set with probability probs[i]."""
        n = 1 << N_BITS
        p = np.ones(n)
        for combo in range(n):
            for i, pi in enumerate(probs):
                p[combo] *= pi if combo >> i & 1 else 1.0 - pi
        edges = np.round(np.cumsum(p) * 65536).astype(np.int64)
        lut = np.searchsorted(edges, np.arange(65536), side="right").clip(max=n - 1)
        return lut.astype(np.int64)

    # ------------------------------------------------------------------------------------
    def reset(self, grids: np.ndarray | None = None) -> np.ndarray:
        """Set the world state (empty worlds if ``grids`` is None).  The wall border is enforced."""
        if grids is None:
            self.grid[:] = 0
        else:
            self.grid[:] = grids
        self.grid[:, [0, -1], :] = WALL
        self.grid[:, :, [0, -1]] = WALL
        return self.grid

    def paint(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions)
        x, y, tool = actions[:, 0], actions[:, 1], actions[:, 2]
        mat = self.materials.tool_material[tool]
        active = mat >= 0
        if not active.any():
            return
        r = self.brush_radius
        mask = (np.abs(self._yy - y[:, None, None]) <= r) & (np.abs(self._xx - x[:, None, None]) <= r)
        mask &= active[:, None, None] & self._interior
        mask &= (mat == 0)[:, None, None] | (self.grid == 0)  # erase anything, paint only into empty
        np.copyto(self.grid, mat.astype(np.uint8)[:, None, None], where=mask)

    def _substep(self, oy: int, ox: int) -> None:
        n = self.size
        v = self.grid[:, oy : n - oy, ox : n - ox]
        a, b, c, d = v[:, 0::2, 0::2], v[:, 0::2, 1::2], v[:, 1::2, 0::2], v[:, 1::2, 1::2]
        K = len(self.materials)
        idx = a.astype(np.int64)
        idx += b.astype(np.int64) * K
        idx += c.astype(np.int64) * (K * K)
        idx += d.astype(np.int64) * (K * K * K)
        if self._needs_rng:
            idx += self._bits_lut[self.rng.integers(0, 65536, size=idx.shape, dtype=np.uint16)]
        else:
            idx += self._all_bits
        out = self._table[idx]
        a[...], b[...], c[...], d[...] = out[..., 0], out[..., 1], out[..., 2], out[..., 3]

    def step(self, actions: np.ndarray | None = None) -> np.ndarray:
        """Apply the brush actions, then advance physics by one frame.  Returns the live grid."""
        if actions is not None:
            self.paint(actions)
        for oy, ox in OFFSETS:
            self._substep(oy, ox)
        return self.grid

    def counts(self) -> np.ndarray:
        """(B, K) number of cells of each material."""
        K = len(self.materials)
        flat = self.grid.reshape(self.batch, -1).astype(np.int64) + np.arange(self.batch)[:, None] * K
        return np.bincount(flat.ravel(), minlength=self.batch * K).reshape(self.batch, K)
