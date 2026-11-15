"""Play with a trained world model in the browser.

    uv run python -m sandwm.play --run runs/fix-recon1

Opens a local page showing the real simulator and the world model's imagination side by side.
Your brush drives both: reality steps the falling-sand simulator, while the model only gets the
embedding of the last frame it was shown plus your actions, and rolls its latent state forward on
its own predictions.  The decoder turns that imagined latent into a picture.  "Sync" shows
the model reality again (re-encodes the latest true frames); between syncs it is dreaming.

Everything runs locally on the CPU (about 30 ms per step for the small CNN model) and the server
only listens on 127.0.0.1.
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from .data import encode_actions, observe
from .evaluate import decode, load_run, miou
from .materials import MATERIAL_SETS
from .models import count_params
from .scenes import random_scene
from .sim import SandSim

PAGE = Path(__file__).with_name("play.html")


def find_runs(runs_dir: Path) -> list[str]:
    """Names of the run directories under ``runs_dir`` that hold a checkpoint."""
    return sorted(p.parent.name for p in Path(runs_dir).glob("*/ckpt.pt"))


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
        run = Path(run)
        model, decoder, ckpt = load_run(run, self.device)
        meta = ckpt["data_meta"]
        ms, size = MATERIAL_SETS[meta["materials"]], int(meta["size"])
        same_world = self.sim is not None and ms is self.ms and size == self.size
        self.run, self.model, self.decoder, self.ckpt = run, model, decoder, ckpt
        self.ms, self.size, self.sim_kwargs = ms, size, dict(meta.get("sim", {}))
        self.skip, self.history = int(ckpt["args"]["frame_skip"]), model.cfg.history
        if same_world:  # keep the current world and show it to the new model (its frame skip may differ)
            self.frames = [self.sim.grid[0].copy()]
            self.actions, self.pending = [], []
            self.sync()
        else:
            self.reset()

    def info(self) -> dict:
        cfg, args = self.model.cfg, self.ckpt["args"]
        return {
            "run": self.run.name,
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
        }

    @torch.no_grad()
    def _encode(self, grids: np.ndarray) -> torch.Tensor:
        """True grids (T, H, W) -> embeddings (1, T, D)."""
        return self.model.encode(observe(torch.from_numpy(np.ascontiguousarray(grids[None])).to(self.device), self.ms.palette))

    def _decode(self, z: torch.Tensor) -> np.ndarray:
        """Embedding (D,) -> material grid (H, W)."""
        return decode(self.decoder, z[None])[0]

    # --- world --------------------------------------------------------------------------------
    def reset(self, scene: str = "random", seed: int | None = None) -> None:
        self.seed = int(np.random.default_rng().integers(1 << 31)) if seed is None else int(seed)
        rng = np.random.default_rng(self.seed)
        grid = random_scene(rng, self.size, self.ms) if scene == "random" else np.zeros((self.size,) * 2, np.uint8)
        self.scene = scene
        self.sim = SandSim(1, self.size, self.ms, seed=[self.seed, 1], **self.sim_kwargs)
        self.sim.reset(grid[None])
        self.tick = 0  # raw simulator frames since the reset
        self.step_count = 0  # model steps since the reset
        self.frames = [self.sim.grid[0].copy()]
        self.actions: list[torch.Tensor] = []
        self.pending: list[np.ndarray] = []
        self.sync()

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


# --- HTTP ---------------------------------------------------------------------------------------
def make_handler(session: PlaySession, runs_dir: Path):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the terminal quiet; errors still print
            pass

        def _send(self, body: bytes, ctype: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(json.dumps(_nan_to_none(obj)).encode(), "application/json", status)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                with lock:
                    self._json({"info": session.info(), "runs": find_runs(runs_dir), "state": session.snapshot()})
            else:
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                with lock:
                    if "auto_sync" in body:
                        session.auto_sync = max(0, int(body["auto_sync"]))
                    if "model_sees_brush" in body:
                        session.model_sees_brush = bool(body["model_sees_brush"])
                    if self.path == "/api/step":
                        self._json(session.step(body["action"]))
                    elif self.path == "/api/sync":
                        session.sync()
                        self._json(session.snapshot())
                    elif self.path == "/api/reset":
                        session.reset(body.get("scene", "random"), body.get("seed"))
                        self._json(session.snapshot())
                    elif self.path == "/api/load":
                        name = str(body["run"])
                        if name not in find_runs(runs_dir):
                            raise ValueError(f"no checkpoint for run {name!r} in {runs_dir}")
                        session.load(runs_dir / name)
                        self._json({"info": session.info(), "runs": find_runs(runs_dir), "state": session.snapshot()})
                    elif self.path == "/api/settings":
                        self._json({"ok": True})
                    else:
                        self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
            except Exception as e:  # report to the page rather than dropping the connection
                self._json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.BAD_REQUEST)

    return Handler


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=None, help="run directory with ckpt.pt (default: newest under --runs-dir)")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"), help="where the page looks for other runs to switch to")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--threads", type=int, default=1,
                   help="CPU threads for the model; one frame at a time runs fastest on one (more may help a large ViT)")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    torch.set_num_threads(args.threads)
    run = args.run
    if run is None:
        ckpts = sorted(args.runs_dir.glob("*/ckpt.pt"), key=lambda q: q.stat().st_mtime)
        if not ckpts:
            raise SystemExit(f"no runs with a ckpt.pt under {args.runs_dir}; pass --run")
        run = ckpts[-1].parent
    if args.run is not None and run.parent.resolve() != args.runs_dir.resolve():
        args.runs_dir = run.parent  # so the run picker lists the chosen run's siblings
    session = PlaySession(run, args.device)

    for port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(session, args.runs_dir))
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"no free port in {args.port}..{args.port + 19}")
    url = f"http://127.0.0.1:{port}/"
    info = session.info()
    print(f"world model: {info['run']} ({info['encoder']} encoder, {info['params']:,} params, frame skip {info['frame_skip']}, "
          f"{info['train_steps']} training steps)")
    print(f"open {url}   (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
