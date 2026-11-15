"""The model side of sandwm.play: one simulator world and one world model, driven by the same brush.

sandwm.play (the page and its web server) runs this module as a separate process, on the code the
run was trained with (see sandwm.play), and talks to it over stdin/stdout: one JSON request per
line, one JSON reply per line.

    python -m sandwm.playsession --run .lab/runs/r0003-grid8-15k

It only relies on parts of sandwm that have been stable since the player was written (load_run,
decode, miou, encode_actions, observe, random_scene, SandSim, MATERIAL_SETS), so that it can be
dropped into a copy of older or newer code.
"""

from __future__ import annotations

import argparse
import base64
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .data import encode_actions, observe
from .evaluate import decode, load_run, miou
from .materials import MATERIAL_SETS
from .models import count_params
from .scenes import random_scene
from .sim import SandSim


class PlaySession:
    """One simulator world and one world model, driven by the same stream of brush actions.

    Bookkeeping is at the model's time resolution: with frame skip k, every k raw actions make one
    model step.  ``frames`` holds the true grids and ``latents`` the model's state at the last few
    model steps; ``actions`` holds the encoded action block taken at each of those steps, so that
    ``latents[i]`` and ``actions[i]`` line up as in ``JEPA.predict``.
    """

    def __init__(self, run: Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.sim: SandSim | None = None
        self.auto_sync = 0  # model steps between automatic syncs; 0 = never
        self.model_sees_brush = True
        self.load(run)

    # --- model ----------------------------------------------------------------------------
    def load(self, run: Path) -> None:
        """Switch to another run.  The world is kept when it fits the new run's world (size and
        materials); it then runs on the new run's simulator settings and is shown to the new model."""
        run = Path(run)
        try:
            model, decoder, ckpt = load_run(run, self.device)
        except TypeError as e:  # ModelConfig(**saved config) meets an option this code does not have
            raise ValueError(f"{run.name} was trained with model options this code does not have ({e})") from e
        meta = ckpt["data_meta"]
        world = None if self.sim is None else self.sim.grid[0].copy()
        self.run, self.model, self.decoder, self.ckpt = run, model, decoder, ckpt
        self.ms, self.size = MATERIAL_SETS[meta["materials"]], int(meta["size"])
        self.sim_kwargs = dict(meta.get("sim", {}))
        # new random scenes are drawn like the run's training scenes (as far as this code's random_scene can)
        accepted = inspect.signature(random_scene).parameters
        self.terrain = {k: tuple(v) if isinstance(v, list) else v
                        for k, v in meta.get("terrain", {}).items() if k in accepted}
        self.skip, self.history = int(ckpt["args"]["frame_skip"]), model.cfg.history
        if self.fits(world):
            self._start(world)
        else:
            self.reset()

    def info(self) -> dict:
        cfg, args = self.model.cfg, self.ckpt["args"]
        return {
            "run": self.run.as_posix(),
            "name": self.run.name,
            "train_steps": int(self.ckpt["step"]),
            "frame_skip": self.skip,
            "history": self.history,
            "encoder": cfg.encoder,
            "params": count_params(self.model),
            "embed_dim": cfg.embed_dim,
            "recon_weight": float(args.get("recon_weight", 0.0)),
            "data": Path(str(args["data"])).name,
            "materials": list(self.ms.names),
            "tools": list(self.ms.tools),
            "palette": self.ms.palette.tolist(),
            "size": self.size,
            "code_dir": str(Path(__file__).resolve().parent),  # which copy of sandwm this process runs
        }

    @torch.no_grad()
    def _encode(self, grids: np.ndarray) -> torch.Tensor:
        """True grids (T, H, W) -> embeddings (1, T, D)."""
        return self.model.encode(observe(torch.from_numpy(np.ascontiguousarray(grids[None])).to(self.device), self.ms.palette))

    def _decode(self, z: torch.Tensor) -> np.ndarray:
        """Embedding (D,) -> material grid (H, W)."""
        return decode(self.decoder, z[None])[0]

    # --- world --------------------------------------------------------------------------------
    def fits(self, grid: np.ndarray | None) -> bool:
        return grid is not None and grid.shape == (self.size, self.size) and int(grid.max(initial=0)) < len(self.ms)

    def reset(self, scene: str = "random", seed: int | None = None) -> None:
        self.seed = int(np.random.default_rng().integers(1 << 31)) if seed is None else int(seed)
        rng = np.random.default_rng(self.seed)
        grid = random_scene(rng, self.size, self.ms, **self.terrain) if scene == "random" else np.zeros((self.size,) * 2, np.uint8)
        self.scene = scene
        self.tick = 0  # raw simulator frames since the reset
        self.step_count = 0  # model steps since the reset
        self._start(grid)

    def _start(self, grid: np.ndarray) -> None:
        """A simulator on this run's settings, holding ``grid``; the model is shown it."""
        self.sim = SandSim(1, self.size, self.ms, seed=[self.seed, 1], **self.sim_kwargs)
        self.sim.reset(grid[None])
        self.frames = [self.sim.grid[0].copy()]
        self.actions: list[torch.Tensor] = []
        self.pending: list[np.ndarray] = []
        self.sync()

    def world(self) -> dict:
        """The world as it is now, for another session (possibly on other code) to carry on with."""
        return {"grid": _b64(self.sim.grid[0]), "size": self.size, "seed": self.seed, "scene": self.scene,
                "tick": self.tick, "step": self.step_count}

    def adopt(self, world: dict) -> bool:
        """Carry on with another session's world, if it fits this run's world."""
        grid = np.frombuffer(base64.b64decode(world["grid"]), np.uint8)
        if grid.size != self.size**2:
            return False
        grid = grid.reshape(self.size, self.size).copy()
        if not self.fits(grid):
            return False
        self.seed, self.scene = int(world["seed"]), str(world["scene"])
        self.tick, self.step_count = int(world["tick"]), int(world["step"])
        self._start(grid)
        return True

    def sync(self) -> None:
        """Show the model reality: replace its latent history by the embeddings of the true frames."""
        self.latents = self._encode(np.stack(self.frames))[0]  # (T, D)
        self.frozen_latent = self.imagined_latent = self.latents[-1]
        self.frozen = self._decode(self.frozen_latent)  # what "nothing changes from here" looks like
        self.since_sync = 0
        self.synced_at = self.step_count

    def step(self, action) -> dict:
        """Apply one raw brush action ``(x, y, tool)`` to reality, and to the model every k-th frame."""
        action = np.asarray(action, dtype=np.int16).reshape(3)
        self.sim.step(action[None])
        self.tick += 1
        self.pending.append(action)
        if len(self.pending) < self.skip:
            return self.snapshot(model_stepped=False)

        block = np.stack(self.pending)  # (k, 3)
        self.pending = []
        if not self.model_sees_brush:
            block = block.copy()
            block[:, 2] = 0
        a = encode_actions(torch.from_numpy(block).to(self.device), self.size, len(self.ms.tools))  # (k * A,)
        self.actions = (self.actions + [a])[-self.history :]
        ctx = self.latents[-len(self.actions) :]
        with torch.no_grad():
            nxt = self.model.predict(ctx[None], torch.stack(self.actions[-len(ctx) :])[None])[0, -1]
        self.latents = torch.cat([self.latents, nxt[None]])[-self.history :]
        self.frames = (self.frames + [self.sim.grid[0].copy()])[-self.history :]
        self.step_count += 1
        self.since_sync += 1
        self.imagined_latent = nxt
        synced = bool(self.auto_sync and self.since_sync >= self.auto_sync)
        if synced:
            snap = self.snapshot(model_stepped=True)  # report the imagination before it is corrected
            self.sync()
            snap["synced_after"] = True
            return snap
        return self.snapshot(model_stepped=True)

    # --- reporting -------------------------------------------------------------------------
    def snapshot(self, model_stepped: bool = False) -> dict:
        """Everything the page shows.  ``model_stepped`` marks a new point for the drift chart."""
        truth = self.sim.grid[0]
        z_true = self._encode(truth[None])[0, 0]
        imagined, recon = decode(self.decoder, torch.stack([self.imagined_latent, z_true]))  # one decoder call
        K = len(self.ms)
        return {
            "tick": self.tick,
            "step": self.step_count,
            "since_sync": self.since_sync,
            "synced_at": self.synced_at,
            "seed": self.seed,
            "scene": self.scene,
            "auto_sync": self.auto_sync,
            "model_sees_brush": self.model_sees_brush,
            "model_stepped": model_stepped,
            "truth": _b64(truth),
            "imagined": _b64(imagined),
            "recon": _b64(recon),
            "metrics": {  # mIoU against reality: open-loop imagination, "nothing changes since the sync", ceiling
                "imagined": miou(imagined, truth, K),
                "frozen": miou(self.frozen, truth, K),
                "ceiling": miou(recon, truth, K),
                "latent_mse": float((self.imagined_latent - z_true).square().mean()),
                "latent_mse_frozen": float((self.frozen_latent - z_true).square().mean()),
            },
        }


def _b64(grid: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(grid, dtype=np.uint8).tobytes()).decode("ascii")


def _nan_to_none(obj):
    """JSON has no NaN (mIoU is NaN on a frame with no material at all)."""
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    return obj


# --- requests -----------------------------------------------------------------------------------
def handle(session: PlaySession, cmd: str, body: dict) -> dict:
    """One request from the page (relayed by sandwm.play).  Every request may carry the settings."""
    if "auto_sync" in body:
        session.auto_sync = max(0, int(body["auto_sync"]))
    if "model_sees_brush" in body:
        session.model_sees_brush = bool(body["model_sees_brush"])
    if cmd == "state":
        return {"info": session.info(), "state": session.snapshot()}
    if cmd == "step":
        return session.step(body["action"])
    if cmd == "sync":
        session.sync()
        return session.snapshot()
    if cmd == "reset":
        session.reset(body.get("scene", "random"), body.get("seed"))
        return session.snapshot()
    if cmd == "load":
        session.load(Path(body["run"]))
        return {"info": session.info(), "state": session.snapshot()}
    if cmd == "world":
        return session.world()
    if cmd == "adopt":
        carried = session.adopt(body)
        return {"info": session.info(), "state": session.snapshot(), "carried": carried}
    if cmd == "settings":
        return {"ok": True}
    raise ValueError(f"unknown request {cmd!r}")


def serve(argv=None) -> None:
    """Answer requests on stdin, one JSON object per line, until stdin closes.  The first line
    written is the reply to loading the run: {"info", "state"} or {"error"}."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    out = sys.stdout
    sys.stdout = sys.stderr  # anything else that gets printed goes to the terminal, not into the replies

    def reply(obj) -> None:
        out.write(json.dumps(_nan_to_none(obj)) + "\n")
        out.flush()

    torch.set_num_threads(args.threads)
    try:
        session = PlaySession(args.run, args.device)
    except Exception as e:
        reply({"error": f"{type(e).__name__}: {e}"})
        return
    reply({"info": session.info(), "state": session.snapshot()})
    while line := sys.stdin.readline():
        try:
            req = json.loads(line)
            reply(handle(session, req["cmd"], req.get("body") or {}))
        except Exception as e:  # report to the page rather than dying
            reply({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    serve()
