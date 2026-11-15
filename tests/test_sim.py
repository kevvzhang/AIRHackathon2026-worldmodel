import numpy as np
import pytest

from sandwm.materials import BASIC, EMPTY, EXTENDED, STATIC, WALL
from sandwm.scenes import StrokePolicy, random_scene
from sandwm.sim import N_BITS, SandSim, build_rule_table

SAND, WATER = BASIC.index("sand"), BASIC.index("water")


def empty_world(size=32, **kw):
    return SandSim(1, size, BASIC, **kw)


def run(sim, frames):
    for _ in range(frames):
        sim.step()
    return sim.grid[0]


@pytest.mark.parametrize("ms", [BASIC, EXTENDED])
def test_rule_table_only_permutes_and_never_moves_statics(ms):
    """Exhaustive over every block state and random-bit pattern."""
    K, table = len(ms), build_rule_table(ms)
    assert table.shape == (2**N_BITS * K**4, 4)
    idx = np.arange(len(table)) % K**4
    before = np.stack([idx % K, idx // K % K, idx // K**2 % K, idx // K**3], axis=1)
    assert (np.sort(before, 1) == np.sort(table, 1)).all(), "a block update created or destroyed matter"
    static = np.array(ms.kinds)[before] == STATIC
    assert (table[static] == before[static]).all(), "a static cell moved"


def test_grain_free_falls_four_cells_per_frame_then_rests():
    sim = empty_world(stochastic=False)
    sim.grid[0, 2, 10] = SAND
    rows = []
    for _ in range(10):
        sim.step()
        rows.append(int(np.argwhere(sim.grid[0] == SAND)[0, 0]))
    assert rows[:7] == [6, 10, 14, 18, 22, 26, 30]
    assert rows[-1] == 30  # resting on the floor (row 31 is the border)
    assert (sim.grid[0] == SAND).sum() == 1


def test_sand_forms_a_stable_pile():
    sim = empty_world(stochastic=False)
    pour = np.array([[16, 3, BASIC.tool("sand")]])
    for _ in range(30):
        sim.step(pour)
    run(sim, 60)
    settled = sim.grid[0].copy()
    assert (run(sim, 10) == settled).all(), "pile kept moving"
    heights = (settled == SAND).sum(0)
    assert heights.max() >= 8 and (np.abs(np.diff(heights)) <= 1).all(), "expected a pile with slopes of at most 45 degrees"


def test_water_levels_out():
    sim = empty_world(seed=0)
    sim.grid[0, 5:25, 2:5] = WATER  # a tall column of 60 cells against the left wall
    g = run(sim, 400)
    assert (g == WATER).sum() == 60
    assert (g == WATER).sum(0).max() <= 3, "water should spread into a shallow layer across 30 columns"


def test_sand_sinks_through_water():
    sim = empty_world(seed=0)
    sim.grid[0, 24:31, 1:31] = WATER
    sim.grid[0, 10:14, 12:20] = SAND
    g = run(sim, 200)
    sand_rows, water_rows = np.argwhere(g == SAND)[:, 0], np.argwhere(g == WATER)[:, 0]
    assert sand_rows.min() >= 27, "sand should have sunk to the bottom"
    assert not ((g[:-1] == SAND) & (g[1:] == WATER)).any(), "no sand should be resting on water"
    assert (g == SAND).sum() == 32 and (g == WATER).sum() == 210 and water_rows.min() < 24  # displaced water rose


def test_one_pixel_diagonal_wall_does_not_leak():
    sim = empty_world(seed=0)
    i = np.arange(20)
    sim.grid[0, 8 + i, 4 + i] = WALL  # ramp going down to the right
    sim.grid[0, 28, 4:25] = WALL
    sim.grid[0, 8:29, 4] = WALL  # closed triangle underneath the ramp
    sim.grid[0, 2:6, 6:26] = WATER
    g = run(sim, 300)
    inside = np.zeros_like(g, bool)
    for k in i:
        inside[9 + k : 28, 4 + k] = True
    assert not ((g != EMPTY) & (g != WALL) & inside).any(), "water leaked through a diagonal wall"


def test_extended_materials_stratify_by_density():
    ms = EXTENDED
    oil, water, smoke = ms.index("oil"), ms.index("water"), ms.index("smoke")
    sim = SandSim(1, 32, ms, seed=0)
    sim.grid[0, 20:25, 1:31] = oil
    sim.grid[0, 10:15, 1:31] = water
    sim.grid[0, 28:30, 1:31] = smoke
    g = run(sim, 500)
    rows = lambda m: np.argwhere(g == m)[:, 0]
    assert rows(smoke).max() < rows(oil).min() and rows(oil).max() <= rows(water).min(), "expected smoke above oil above water"
    assert rows(smoke).min() == 1  # collected under the ceiling


def test_deterministic_mode_ignores_the_seed_and_stochastic_mode_is_reproducible():
    scene = random_scene(np.random.default_rng(3), 32, BASIC)

    def rollout(**kw):
        sim = SandSim(1, 32, BASIC, **kw)
        sim.reset(scene[None])
        return run(sim, 40).copy()

    assert (rollout(stochastic=False, seed=1) == rollout(stochastic=False, seed=2)).all()
    assert (rollout(seed=5) == rollout(seed=5)).all()
    assert (rollout(seed=5) != rollout(seed=6)).any()


def test_worlds_in_a_batch_are_independent():
    rng = np.random.default_rng(0)
    scenes = np.stack([random_scene(rng, 32, BASIC) for _ in range(4)])
    batch = SandSim(4, 32, BASIC, stochastic=False)
    batch.reset(scenes)
    for _ in range(20):
        batch.step()
    for i in range(4):
        single = SandSim(1, 32, BASIC, stochastic=False)
        single.reset(scenes[i : i + 1])
        assert (run(single, 20) == batch.grid[i]).all()


def test_brush_semantics():
    sim = empty_world()
    sim.grid[0, 10, 10] = WALL
    sim.paint(np.array([[10, 10, BASIC.tool("sand")]]))
    block = sim.grid[0, 9:12, 9:12]
    assert block[1, 1] == WALL and (block == SAND).sum() == 8, "painting fills empty cells only"
    sim.paint(np.array([[10, 10, BASIC.tool("erase")]]))
    assert (sim.grid[0, 9:12, 9:12] == EMPTY).all(), "erasing removes everything, walls included"
    before = sim.grid.copy()
    sim.paint(np.array([[10, 10, BASIC.tool("none")]]))
    assert (sim.grid == before).all()
    sim.paint(np.array([[0, 0, BASIC.tool("erase")]]))
    assert (sim.grid[0, 0, :] == WALL).all() and (sim.grid[0, :, 0] == WALL).all(), "the border cannot be edited"


def test_random_episodes_keep_the_border_and_only_the_brush_changes_mass():
    rng = np.random.default_rng(0)
    B, size = 16, 32
    sim = SandSim(B, size, BASIC, seed=0)
    sim.reset(np.stack([random_scene(rng, size, BASIC) for _ in range(B)]))
    policy = StrokePolicy(B, size, BASIC, rng)
    tools_seen = set()
    for _ in range(60):
        action = policy.act()
        assert action[:, :2].min() >= 1 and action[:, :2].max() <= size - 2
        tools_seen |= set(action[:, 2].tolist())
        sim.paint(action)
        counts = sim.counts()
        sim.step()  # physics only
        assert (sim.counts() == counts).all(), "physics changed the amount of some material"
    assert tools_seen == set(range(len(BASIC.tools)))
    g = sim.grid
    assert (g[:, 0] == WALL).all() and (g[:, -1] == WALL).all() and (g[:, :, 0] == WALL).all() and (g[:, :, -1] == WALL).all()
