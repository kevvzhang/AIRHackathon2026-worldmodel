"""Material tables for the falling-sand world.

A material is fully described by its *kind* (how it moves) and its *density* (what it sinks
through).  The simulator derives all of its physics from this table, so adding e.g. oil is a
one-line change here -- no simulator code needs to be touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

# --- kinds -----------------------------------------------------------------------------------
VOID = 0  # empty space; behaves like a very light fluid that everything heavier falls through
STATIC = 1  # never moves, never yields (walls)
POWDER = 2  # falls, slides diagonally, piles up; does not yield to anything
LIQUID = 3  # falls, slides, and spreads sideways when resting on something
GAS = 4  # lighter than void: rises, creeps around obstacles, spreads under ceilings

KIND_NAMES = ("void", "static", "powder", "liquid", "gas")


@dataclass(frozen=True)
class Material:
    name: str
    kind: int
    density: int  # void is 1; gases must be 0; liquids/powders > 1
    color: tuple[int, int, int]


@dataclass(frozen=True)
class MaterialSet:
    """An ordered set of materials.  Index 0 must be empty space and index 1 the wall."""

    name: str
    materials: tuple[Material, ...]

    def __post_init__(self) -> None:
        if self.materials[0].kind != VOID or self.materials[1].kind != STATIC:
            raise ValueError("material 0 must be VOID (empty) and material 1 STATIC (wall)")
        if len(self.materials) > 16:
            raise ValueError("at most 16 materials are supported by the rule table")
        names = [m.name for m in self.materials]
        if len(set(names)) != len(names):
            raise ValueError("material names must be unique")

    def __len__(self) -> int:
        return len(self.materials)

    def index(self, name: str) -> int:
        for i, m in enumerate(self.materials):
            if m.name == name:
                return i
        raise KeyError(name)

    @cached_property
    def names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.materials)

    @cached_property
    def kinds(self) -> tuple[int, ...]:
        return tuple(m.kind for m in self.materials)

    @cached_property
    def densities(self) -> tuple[int, ...]:
        return tuple(m.density for m in self.materials)

    @cached_property
    def palette(self) -> np.ndarray:
        """(K, 3) uint8 RGB colour of each material."""
        return np.array([m.color for m in self.materials], dtype=np.uint8)

    # --- brush tools ---------------------------------------------------------------------
    # Tool 0 does nothing, tool 1 erases, tools 2.. paint material (tool - 1).
    @cached_property
    def tools(self) -> tuple[str, ...]:
        return ("none", "erase") + self.names[1:]

    @cached_property
    def tool_material(self) -> np.ndarray:
        """Material written by each tool; -1 means 'do nothing'."""
        return np.array([-1, 0] + list(range(1, len(self))), dtype=np.int16)

    def tool(self, name: str) -> int:
        return self.tools.index(name)


EMPTY, WALL = 0, 1  # fixed indices in every material set

BASIC = MaterialSet(
    name="basic",
    materials=(
        Material("empty", VOID, 1, (16, 16, 26)),
        Material("wall", STATIC, 255, (128, 128, 140)),
        Material("sand", POWDER, 8, (232, 196, 94)),
        Material("water", LIQUID, 5, (58, 120, 232)),
    ),
)

# A richer set to grow into: oil floats on water, smoke rises.  (No reactions yet.)
EXTENDED = MaterialSet(
    name="extended",
    materials=BASIC.materials
    + (
        Material("oil", LIQUID, 3, (150, 92, 44)),
        Material("smoke", GAS, 0, (92, 92, 104)),
    ),
)

MATERIAL_SETS = {ms.name: ms for ms in (BASIC, EXTENDED)}
