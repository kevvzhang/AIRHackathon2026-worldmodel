"""Evaluate a trained world model.

    uv run python -m sandwm.evaluate --run runs/lewm

1. Open-loop imagination: encode a few context frames, then roll the predictor forward on its own
   outputs, feeding it only the actions.  Compared against "nothing changes" baselines, in latent
   space and (through the probe decoder) in material space.
2. Linear probes: can physical quantities be read off the embedding with a linear map?
3. Pictures: truth / reconstruction / imagination side by side (PNG contact sheet + GIF).

Everything is written to <run>/eval/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import viz
from .data import EpisodeDataset
from .models import JEPA, Decoder, ModelConfig


def load_run(run: Path, device: torch.device):
    ckpt = torch.load(run / "ckpt.pt", map_location=device, weights_only=True)
    cfg = ModelConfig.from_dict(ckpt["config"])
    model, decoder = JEPA(cfg).to(device), Decoder(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    decoder.load_state_dict(ckpt["decoder"])
    return model.eval(), decoder.eval(), ckpt


@torch.no_grad()
def encode(model: JEPA, ds: EpisodeDataset, frames: np.ndarray, device, chunk: int = 256) -> torch.Tensor:
    """Material grids (N, T, H, W) -> embeddings (N, T, D)."""
    return torch.cat([model.encode(ds.observe(frames[i : i + chunk], device)) for i in range(0, len(frames), chunk)])


@torch.no_grad()
def decode(decoder: Decoder, emb: torch.Tensor, chunk: int = 1024) -> np.ndarray:
    """Embeddings (..., D) -> most likely material per cell (..., H, W)."""
    flat = emb.reshape(-1, emb.size(-1))
    out = torch.cat([decoder(flat[i : i + chunk]).argmax(1) for i in range(0, len(flat), chunk)])
    return out.view(*emb.shape[:-1], *out.shape[-2:]).cpu().numpy().astype(np.uint8)


def miou(pred: np.ndarray, true: np.ndarray, n_classes: int) -> float:
    """Mean IoU over the non-empty materials present in either array."""
    ious = []
    for k in range(1, n_classes):
        union = ((pred == k) | (true == k)).sum()
        if union:
            ious.append(((pred == k) & (true == k)).sum() / union)
    return float(np.mean(ious)) if ious else float("nan")


# --- 1. open-loop rollouts --------------------------------------------------------------------
@torch.no_grad()
def rollout_eval(model, decoder, ds, n_windows: int, horizon: int, frame_skip: int, seed: int, device) -> tuple[dict, dict]:
    C, K = model.cfg.history, model.cfg.n_classes
    frames, actions = ds.sample(n_windows, C + horizon, frame_skip, np.random.default_rng(seed))
    idle = actions.copy()
    idle[:, C - 1 :, :, 2] = 0  # ablation: pretend the brush does nothing from the last context frame on
    emb = encode(model, ds, frames, device)
    imagined = model.rollout(emb[:, :C], ds.encode_actions(actions, device))
    imagined_idle = model.rollout(emb[:, :C], ds.encode_actions(idle, device))

    future, last = emb[:, C:], emb[:, C - 1 : C]
    true, last_grid = frames[:, C:], frames[:, C - 1 : C]
    grid_imagined, grid_recon = decode(decoder, imagined), decode(decoder, future)
    changed = true != last_grid  # cells that differ from the last context frame: where the dynamics are

    def per_step(fn):
        return [float(fn(h)) for h in range(horizon)]

    metrics = {
        "horizon": list(range(1, horizon + 1)),
        "latent_mse": {
            "model": per_step(lambda h: (imagined[:, h] - future[:, h]).square().mean()),
            "copy-last baseline": per_step(lambda h: (last[:, 0] - future[:, h]).square().mean()),
            "model, actions hidden": per_step(lambda h: (imagined_idle[:, h] - future[:, h]).square().mean()),
        },
        "miou": {
            "model (imagined)": per_step(lambda h: miou(grid_imagined[:, h], true[:, h], K)),
            "copy-last baseline": per_step(lambda h: miou(last_grid[:, 0], true[:, h], K)),
            "encode+decode of the true frame (ceiling)": per_step(lambda h: miou(grid_recon[:, h], true[:, h], K)),
        },
        "changed_cell_accuracy": {
            "model (imagined)": per_step(lambda h: (grid_imagined[:, h] == true[:, h])[changed[:, h]].mean()),
            "encode+decode of the true frame (ceiling)": per_step(lambda h: (grid_recon[:, h] == true[:, h])[changed[:, h]].mean()),
        },
        "changed_cell_fraction": per_step(lambda h: changed[:, h].mean()),
    }
    context_recon = decode(decoder, emb[:, :C])
    pictures = {
        "truth": frames,
        "decode(encode(truth))": np.concatenate([context_recon, grid_recon], axis=1),
        "imagined (open loop)": np.concatenate([context_recon, grid_imagined], axis=1),
    }
    return metrics, pictures


# --- 2. linear probes ---------------------------------------------------------------------------
def physical_quantities(grid: np.ndarray, ds: EpisodeDataset) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Scalar properties of each frame: name -> (value per frame, validity mask)."""
    n, h, w = grid.shape
    out = {}
    for k, name in enumerate(ds.materials.names[1:], start=1):
        mask = grid == k
        count = mask.sum((1, 2))
        present = count >= 8  # centre of mass is meaningless for a handful of cells
        safe = np.maximum(count, 1)
        out[f"{name}: amount"] = (count / (h * w), np.ones(n, bool))
        out[f"{name}: centre x"] = (mask.sum(1) @ np.arange(w) / safe / w, present)
        out[f"{name}: centre y"] = (mask.sum(2) @ np.arange(h) / safe / h, present)
    return out


def ridge_r2(z_train, y_train, z_test, y_test, alpha: float = 1.0) -> float:
    """R^2 on held-out data of a ridge regression from embeddings to a scalar."""
    mu, sd = z_train.mean(0), z_train.std(0) + 1e-6
    a, b = (z_train - mu) / sd, (z_test - mu) / sd
    y0 = y_train.mean()
    w = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ (y_train - y0))
    resid = y_test - (b @ w + y0)
    return float(1 - resid.var() / max(y_test.var(), 1e-12))


def probe_eval(model, train: EpisodeDataset, val: EpisodeDataset, n_train: int, n_val: int, seed: int, device) -> dict:
    def sample(ds, n, s):
        rng = np.random.default_rng(s)
        frames = ds.frames[rng.integers(0, len(ds), n), rng.integers(0, ds.frames.shape[1], n)]
        return encode(model, ds, frames[:, None], device)[:, 0].cpu().numpy().astype(np.float64), physical_quantities(frames, ds)

    z_tr, q_tr = sample(train, n_train, seed)
    z_va, q_va = sample(val, n_val, seed + 1)
    return {name: ridge_r2(z_tr[q_tr[name][1]], q_tr[name][0][q_tr[name][1]], z_va[q_va[name][1]], q_va[name][0][q_va[name][1]])
            for name in q_tr}


# --- 3. report ----------------------------------------------------------------------------------
def plot_rollout(metrics: dict, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = metrics["horizon"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    panels = [("latent_mse", "latent MSE to the true embedding (lower is better)", True),
              ("miou", "decoded mIoU over materials (higher is better)", False),
              ("changed_cell_accuracy", "accuracy on cells that changed since the context", False)]
    for a, (key, name, log) in zip(ax, panels):
        for label, values in metrics[key].items():
            a.plot(h, values, "o-", ms=3, label=label)
        a.set(xlabel="imagined steps", title=name, yscale="log" if log else "linear")
        if not log:
            a.set_ylim(0, 1)
        a.legend(fontsize=8), a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="run directory containing ckpt.pt")
    p.add_argument("--data", type=Path, default=None, help="dataset (default: the one the run was trained on)")
    p.add_argument("--horizon", type=int, default=16, help="imagined latent steps")
    p.add_argument("--windows", type=int, default=256, help="validation windows for the rollout metrics")
    p.add_argument("--show", type=int, default=4, help="episodes shown in the pictures")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    model, decoder, ckpt = load_run(args.run, device)
    data = args.data or Path(ckpt["args"]["data"])
    skip, C = int(ckpt["args"]["frame_skip"]), model.cfg.history
    train, val = EpisodeDataset(data, "train", 400), EpisodeDataset(data, "val")
    horizon = min(args.horizon, val.episode_len // skip - C + 1)
    out = args.run / "eval"
    out.mkdir(exist_ok=True)

    metrics, pictures = rollout_eval(model, decoder, val, args.windows, horizon, skip, args.seed, device)
    probes = probe_eval(model, train, val, 6000, 2000, args.seed, device)
    (out / "metrics.json").write_text(json.dumps({"step": ckpt["step"], "frame_skip": skip, "rollout": metrics, "probes": probes}, indent=2))

    title = f"{args.run.name}: open-loop imagination from {C} context frames ({args.windows} val windows, frame skip {skip})"
    plot_rollout(metrics, out / "rollout_curves.png", title)
    pal = val.materials.palette
    steps = [h for h in (1, 2, 4, 8, 12, 16, 24, 32) if h <= horizon]
    times, labels = [C - 1] + [C - 1 + h for h in steps], ["last context frame"] + [f"+{h} steps" for h in steps]
    sheets = [viz.contact_sheet({name: seq[i] for name, seq in pictures.items()}, pal, times, labels=labels)
              for i in range(args.show)]
    viz.save_png(out / "rollout_sheet.png", viz.vstack(sheets, pad=10))
    gif = [viz.vstack([viz.hstack([viz.render(seq[i, t], pal, 3) for seq in pictures.values()]) for i in range(args.show)])
           for t in range(C + horizon)]
    viz.save_gif(out / "rollout.gif", gif, fps=6)

    print(f"\n{title}")
    lm, mi, ch = metrics["latent_mse"], metrics["miou"], metrics["changed_cell_accuracy"]
    print(f"\n  {'steps':>5} | {'latent MSE':^31} | {'decoded mIoU':^27} | changed-cell accuracy")
    print(f"  {'':>5} | {'model':>8} {'copy':>8} {'no-action':>12} | {'model':>7} {'copy':>7} {'ceiling':>9} | {'model':>7} {'ceiling':>8} {'(% cells)':>10}")
    for i, h in enumerate(metrics["horizon"]):
        if h in (1, 2, 4, 8, 12, 16, 24, 32) or h == horizon:
            print(f"  {h:5d} | {lm['model'][i]:8.4f} {lm['copy-last baseline'][i]:8.4f} {lm['model, actions hidden'][i]:12.4f} | "
                  f"{mi['model (imagined)'][i]:7.3f} {mi['copy-last baseline'][i]:7.3f} "
                  f"{mi['encode+decode of the true frame (ceiling)'][i]:9.3f} | {ch['model (imagined)'][i]:7.3f} "
                  f"{ch['encode+decode of the true frame (ceiling)'][i]:8.3f} {100 * metrics['changed_cell_fraction'][i]:9.1f}%")
    print("\n  linear probes, R^2 on validation frames (1 = perfectly linearly readable from the embedding):")
    for name, r2 in probes.items():
        print(f"    {name:20s} {r2:6.3f}")
    print(f"\n  wrote {out}/ (metrics.json, rollout_curves.png, rollout_sheet.png, rollout.gif)")


if __name__ == "__main__":
    main()
