"""Generate a falling-sand dataset.

    uv run python -m sandwm.generate --out data/basic64 --episodes 2000 --val-episodes 200

Every episode starts from a random scene and is driven by the scripted random brush policy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import viz
from .data import save_shard
from .materials import MATERIAL_SETS, MaterialSet
from .scenes import StrokePolicy, random_scene
from .sim import SandSim


def simulate_episodes(n: int, steps: int, size: int, ms: MaterialSet, seed, sim_kwargs: dict | None = None):
    """Simulate ``n`` episodes in one batch.  Returns frames (n, steps+1, H, W) and actions (n, steps, 3)."""
    rng = np.random.default_rng([*np.atleast_1d(seed), 0])
    sim = SandSim(n, size, ms, seed=[*np.atleast_1d(seed), 1], **(sim_kwargs or {}))
    sim.reset(np.stack([random_scene(rng, size, ms) for _ in range(n)]))
    policy = StrokePolicy(n, size, ms, rng)
    frames = np.empty((n, steps + 1, size, size), dtype=np.uint8)
    actions = np.empty((n, steps, 3), dtype=np.int16)
    frames[:, 0] = sim.grid
    for t in range(steps):
        actions[:, t] = policy.act()
        frames[:, t + 1] = sim.step(actions[:, t])
    return frames, actions


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="dataset directory to create")
    p.add_argument("--episodes", type=int, default=2000, help="training episodes")
    p.add_argument("--val-episodes", type=int, default=200)
    p.add_argument("--steps", type=int, default=64, help="actions per episode (frames = steps + 1)")
    p.add_argument("--size", type=int, default=64, help="world is size x size cells, including the wall border")
    p.add_argument("--materials", choices=sorted(MATERIAL_SETS), default="basic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", type=int, default=500, help="episodes per shard file")
    p.add_argument("--deterministic", action="store_true", help="switch off all randomness in the physics")
    p.add_argument("--sink-prob", type=float, default=0.5, help="chance per sub-step to sink through a medium")
    p.add_argument("--slide-prob", type=float, default=1.0, help="chance per sub-step to slide off a ledge")
    p.add_argument("--spread-prob", type=float, default=0.9, help="chance per sub-step for a liquid to flow sideways")
    p.add_argument("--brush-radius", type=int, default=1)
    args = p.parse_args(argv)

    if (args.out / "meta.json").exists():
        raise SystemExit(f"{args.out} already holds a dataset; choose another --out or delete it first")
    ms = MATERIAL_SETS[args.materials]
    sim_kwargs = dict(stochastic=not args.deterministic, sink_prob=args.sink_prob, slide_prob=args.slide_prob,
                      spread_prob=args.spread_prob, brush_radius=args.brush_radius)

    t0 = time.perf_counter()
    preview = None
    for split_id, (split, n_total) in enumerate([("train", args.episodes), ("val", args.val_episodes)]):
        for shard_id, first in enumerate(range(0, n_total, args.shard)):
            n = min(args.shard, n_total - first)
            frames, actions = simulate_episodes(n, args.steps, args.size, ms, [args.seed, split_id, shard_id], sim_kwargs)
            save_shard(args.out / split / f"shard-{shard_id:04d}.npz", frames, actions)
            if preview is None:
                preview = frames[:6].copy()
            print(f"{split}: {first + n}/{n_total} episodes  ({time.perf_counter() - t0:.0f}s)", flush=True)

    meta = dict(format=1, materials=ms.name, size=args.size, steps=args.steps, seed=args.seed, sim=sim_kwargs,
                tools=list(ms.tools), episodes=dict(train=args.episodes, val=args.val_episodes))
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

    times = sorted({round(i * args.steps / 8) for i in range(9)})
    sheet = viz.contact_sheet({f"episode {i}": ep for i, ep in enumerate(preview)}, ms.palette, times)
    viz.save_png(args.out / "preview.png", sheet)
    gif = [viz.hstack([viz.render(ep[t], ms.palette, 3) for ep in preview]) for t in range(args.steps + 1)]
    viz.save_gif(args.out / "preview.gif", gif)
    size_mb = sum(f.stat().st_size for f in args.out.rglob("*.npz")) / 1e6
    print(f"wrote {args.out}  ({size_mb:.1f} MB on disk; see preview.png / preview.gif)")


if __name__ == "__main__":
    main()
