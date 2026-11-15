"""A scorecard: named checks of what a world model gets right, on a frozen benchmark.

    uv run python -m sandwm.scorecard --run runs/lewm --bench data/bench-basic-v1
    uv run python -m sandwm.scorecard --run runs/lewm --bench data/bench-basic-v1 --gates gates.toml
    uv run python -m sandwm.scorecard --diff runs/old runs/new

Every check is one number with a direction, read off open-loop imagination on the same windows
of the benchmark: window i is benchmark episode i, the model sees the frames up to frame 16 and
imagines from there, and the checks are read 16 and 48 frames later.  Any two runs are therefore
judged on identical scenes, whatever their frame skip.

The checks are the model's regression tests.  When a new way of being wrong turns up (walls that
drift, sand that multiplies), add the check that would have caught it: a function that takes a
`Rollout` and returns {name: value}, decorated with @check("higher" | "lower" | "info").

Writes <run>/eval/scorecard.json and, under <run>/eval/bench/, pictures of the first windows and
of the worst ones (truth / imagination / the most this latent could show).
"""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from . import viz
from .data import EpisodeDataset
from .evaluate import decode, encode, load_run, miou
from .materials import WALL

CONTEXT_FRAMES = 16  # the last frame the model sees; imagination starts here in every window
FRAMES = (16, 48)  # raw simulator frames of imagination after which the checks are read
SMOOTH = 16  # cells added to a material count before dividing by it (one 3x3 brush stroke is 9 cells)
TOLERANCE = 0.02  # a check has to move by more than this to count as better or worse


@dataclass
class Rollout:
    """Truth and imagination on the same windows.  Time index 0 is the last frame the model saw."""

    truth: np.ndarray  # (windows, steps + 1, H, W) material per cell
    imagined: np.ndarray  # open-loop imagination, decoded; [:, 0] is the decoded last context frame
    idle: np.ndarray  # the same imagination with the brush hidden from the model
    recon: np.ndarray  # decode(encode(truth)): the most this latent and decoder could show
    frame_skip: int
    names: tuple[str, ...]  # material names; 0 is empty space and 1 is wall
    frames: tuple[int, ...] = FRAMES

    def at(self, frames_ahead: int) -> int | None:
        """Time index that many raw frames after the context, or None if the rollout has no such frame."""
        step, rest = divmod(frames_ahead, self.frame_skip)
        return step if rest == 0 and 0 < step < self.truth.shape[1] else None

    @property
    def horizons(self) -> list[int]:
        return [n for n in self.frames if self.at(n) is not None]


@torch.no_grad()
def collect(model, decoder, ds: EpisodeDataset, frame_skip: int, windows: int | None = None,
            frames: tuple[int, ...] = FRAMES, context_frames: int = CONTEXT_FRAMES, device="cpu") -> Rollout:
    """Imagine forward from frame `context_frames` of the first `windows` episodes of `ds`."""
    C = model.cfg.history
    if len(ds.materials) != model.cfg.n_classes or ds.action_dim * frame_skip != model.cfg.action_dim:
        raise ValueError(f"the model was trained on a different world than {ds.meta['materials']!r} "
                         f"({model.cfg.n_classes} materials, action size {model.cfg.action_dim})")
    if (C - 1) * frame_skip > context_frames:
        raise ValueError(f"history {C} at frame skip {frame_skip} needs {(C - 1) * frame_skip} frames of context, "
                         f"more than the {context_frames} the scorecard allows")
    usable = [n for n in frames if n % frame_skip == 0 and context_frames + n <= ds.episode_len]
    if not usable:
        raise ValueError(f"none of the read-out points {frames} fits: frame skip {frame_skip}, episodes of {ds.episode_len} steps")
    steps = max(usable) // frame_skip
    episode = np.arange(min(windows or len(ds), len(ds)))
    start = np.full(len(episode), context_frames - (C - 1) * frame_skip)
    grids, actions = ds.window(episode, start, C + steps, frame_skip)
    hidden = actions.copy()
    hidden[:, C - 1 :, :, 2] = 0  # the brush does nothing from the last context frame on
    emb = encode(model, ds, grids, device)

    def imagine(acts: np.ndarray) -> np.ndarray:
        future = model.rollout(emb[:, :C], ds.encode_actions(acts, device))
        return decode(decoder, torch.cat([emb[:, C - 1 : C], future], dim=1))

    return Rollout(truth=grids[:, C - 1 :], imagined=imagine(actions), idle=imagine(hidden),
                   recon=decode(decoder, emb[:, C - 1 :]), frame_skip=frame_skip, names=tuple(ds.materials.names),
                   frames=tuple(frames))


# --- the checks -----------------------------------------------------------------------------------
CHECKS: list[tuple[Callable[[Rollout], dict[str, float]], str]] = []


def check(better: str):
    """Register a check.  better = "higher" or "lower", or "info" for a signed diagnostic that is never gated."""
    assert better in ("higher", "lower", "info")

    def register(fn):
        CHECKS.append((fn, better))
        return fn

    return register


def iou(a: np.ndarray, b: np.ndarray, k: int) -> float:
    union = ((a == k) | (b == k)).sum()
    return float(((a == k) & (b == k)).sum() / union) if union else float("nan")


def loose(r: Rollout) -> list[tuple[int, str]]:
    """The materials that can move or be poured: everything but empty space and wall."""
    return [(k, name) for k, name in enumerate(r.names) if k > WALL]


@check("higher")
def fidelity(r: Rollout) -> dict:
    """What survives encode -> decode of a single true frame, before any prediction."""
    out = {"fidelity/miou": miou(r.recon[:, 0], r.truth[:, 0], len(r.names))}
    out.update({f"fidelity/iou/{name}": iou(r.recon[:, 0], r.truth[:, 0], k) for k, name in enumerate(r.names) if k})
    # the same quantity as walls/kept, at time 0: walls lost here are lost by the latent, not by the prediction
    inner_wall = (r.truth[:, 0] == WALL)[:, 1:-1, 1:-1]
    out["fidelity/walls"] = float((r.recon[:, 0, 1:-1, 1:-1] == WALL)[inner_wall].mean())
    return out


@check("higher")
def rollout(r: Rollout) -> dict:
    """Imagination against the truth, overall and per material; `beats_copy` is the margin over repeating the last true frame."""
    out, K = {}, len(r.names)
    for n in r.horizons:
        s = r.at(n)
        out[f"rollout/miou@{n}"] = miou(r.imagined[:, s], r.truth[:, s], K)
        out[f"rollout/beats_copy@{n}"] = out[f"rollout/miou@{n}"] - miou(r.truth[:, 0], r.truth[:, s], K)
        out.update({f"iou/{name}@{n}": iou(r.imagined[:, s], r.truth[:, s], k) for k, name in enumerate(r.names) if k})
    return out


@check("higher")
def dynamics(r: Rollout) -> dict:
    """Accuracy on the cells that changed since the context and on those that did not; and what the brush adds."""
    out = {}
    for n in r.horizons:
        s = r.at(n)
        changed = r.truth[:, s] != r.truth[:, 0]
        right = r.imagined[:, s] == r.truth[:, s]
        out[f"dynamics/changed@{n}"] = float(right[changed].mean())
        out[f"dynamics/unchanged@{n}"] = float(right[~changed].mean())
        out[f"actions/gain@{n}"] = float(right[changed].mean() - (r.idle[:, s] == r.truth[:, s])[changed].mean())
    return out


@check("higher")
def walls_kept(r: Rollout) -> dict:
    """Of the cells that are wall both before and after (border excluded), the share still imagined as wall."""
    out = {}
    for n in r.horizons:
        s = r.at(n)
        stays = ((r.truth[:, 0] == WALL) & (r.truth[:, s] == WALL))[:, 1:-1, 1:-1]
        out[f"walls/kept@{n}"] = float((r.imagined[:, s, 1:-1, 1:-1] == WALL)[stays].mean())
    return out


@check("lower")
def walls_invented(r: Rollout) -> dict:
    """Imagined wall where the truth has none, as a share of the true wall cells (border excluded)."""
    out = {}
    for n in r.horizons:
        s = r.at(n)
        true_wall = (r.truth[:, s] == WALL)[:, 1:-1, 1:-1]
        wrong = (r.imagined[:, s, 1:-1, 1:-1] == WALL) & ~true_wall
        out[f"walls/invented@{n}"] = float(wrong.sum() / max(true_wall.sum(), 1))
    return out


def _mass_ratio(r: Rollout, k: int, s: int) -> np.ndarray:
    """Per window: (imagined - true) amount of material k, relative to the true amount."""
    true = (r.truth[:, s] == k).sum((1, 2))
    return ((r.imagined[:, s] == k).sum((1, 2)) - true) / (true + SMOOTH)


@check("lower")
def mass_error(r: Rollout) -> dict:
    """How far the imagined amount of each loose material is from the true amount (0 = conserved)."""
    return {f"mass/{name}@{n}": float(np.abs(_mass_ratio(r, k, r.at(n))).mean()) for n in r.horizons for k, name in loose(r)}


@check("info")
def mass_bias(r: Rollout) -> dict:
    """The same with its sign: above 0 the model creates material, below 0 it loses it."""
    return {f"bias/{name}@{n}": float(_mass_ratio(r, k, r.at(n)).mean()) for n in r.horizons for k, name in loose(r)}


def run_checks(r: Rollout) -> dict[str, dict]:
    """name -> {"value": number or None, "better": "higher" | "lower" | "info"}."""
    out = {}
    for fn, better in CHECKS:
        for name, value in fn(r).items():
            out[name] = {"value": None if value != value else round(float(value), 4), "better": better}
    return out


# --- comparing and gating ---------------------------------------------------------------------------
def diff(old: dict, new: dict, tol: float = TOLERANCE) -> list[tuple[str, float | None, float | None, str]]:
    """(check, old value, new value, verdict) for every check in either card, regressions first."""
    rows = []
    for name in sorted(set(old["checks"]) | set(new["checks"])):
        a, b = old["checks"].get(name), new["checks"].get(name)
        va, vb = (a or {}).get("value"), (b or {}).get("value")
        better = (b or a)["better"]
        if va is None or vb is None:
            verdict, gain = ("new" if a is None else "gone" if b is None else "n/a"), 0.0
        else:
            gain = 0.0 if better == "info" else (vb - va) * (1 if better == "higher" else -1)
            verdict = "info" if better == "info" else "REGRESSED" if gain < -tol else "improved" if gain > tol else "same"
        rows.append((gain, name, va, vb, verdict))
    rows.sort(key=lambda row: (row[0], row[1]))
    return [(name, va, vb, verdict) for _, name, va, vb, verdict in rows]


def load_gates(path: Path) -> dict[str, float]:
    """[gates] table of a TOML file: check name -> threshold.  The check's own direction says which side passes."""
    return {name: float(value) for name, value in tomllib.loads(Path(path).read_text(encoding="utf-8")).get("gates", {}).items()}


def gate_results(card: dict, gates: dict[str, float]) -> list[tuple[str, float | None, float, bool]]:
    """(check, value, threshold, passed) per gate; a check the card lacks does not pass."""
    rows = []
    for name, threshold in gates.items():
        entry = card["checks"].get(name)
        value = entry["value"] if entry else None
        passed = value is not None and (value >= threshold if entry["better"] == "higher" else value <= threshold)
        rows.append((name, value, threshold, passed))
    return rows


def load_card(run: Path) -> dict:
    return json.loads((Path(run) / "eval" / "scorecard.json").read_text(encoding="utf-8"))


def fmt(value: float | None) -> str:
    return "   n/a" if value is None else f"{value:6.3f}"


def format_card(card: dict, gates: dict[str, float] | None = None) -> str:
    lines = [f"{card['run']} on {card['bench']}: {card['windows']} windows, frame skip {card['frame_skip']}, step {card['step']}"]
    marks = {name: (threshold, ok) for name, _, threshold, ok in gate_results(card, gates)} if gates else {}
    arrow = {"higher": "higher is better", "lower": "lower is better", "info": "signed, for information"}
    for name, entry in card["checks"].items():
        gate = f"   gate {marks[name][0]:g}: {'pass' if marks[name][1] else 'FAIL'}" if name in marks else ""
        lines.append(f"  {name:26s} {fmt(entry['value'])}   {arrow[entry['better']]}{gate}")
    if gates:
        missing = [name for name in gates if name not in card["checks"]]
        lines += [f"  {name:26s}    n/a   gate {gates[name]:g}: FAIL (the card has no such check)" for name in missing]
        lines.append(f"  gates passed: {sum(ok for _, ok in marks.values())} of {len(gates)}")
    return "\n".join(lines)


def format_diff(old: dict, new: dict, tol: float = TOLERANCE) -> str:
    rows = diff(old, new, tol)
    lines = [f"{old['run']} -> {new['run']}   (a check must move by more than {tol:g} to count)",
             f"  {'check':26s} {old['run'][:10]:>10s} {new['run'][:10]:>10s}"]
    lines += [f"  {name:26s} {fmt(a):>10s} {fmt(b):>10s}   {verdict}" for name, a, b, verdict in rows if verdict != "same"]
    same = sum(verdict == "same" for *_, verdict in rows)
    worse = sum(verdict == "REGRESSED" for *_, verdict in rows)
    better = sum(verdict == "improved" for *_, verdict in rows)
    lines.append(f"  {worse} regressed, {better} improved, {same} unchanged")
    return "\n".join(lines)


# --- pictures -----------------------------------------------------------------------------------------
def save_pictures(r: Rollout, palette: np.ndarray, out: Path, per_sheet: int = 3, sheets: int = 2) -> dict:
    """PNG sheets of the first windows (the same in every run) and of the worst ones at the longest read-out."""
    last = r.at(max(r.horizons))
    score = np.array([miou(r.imagined[i, last], r.truth[i, last], len(r.names)) for i in range(len(r.truth))])
    worst = np.argsort(np.nan_to_num(score, nan=1.0))[: per_sheet * sheets]
    steps = sorted({0, last} | {s for s in (r.at(n) for n in (4, 8, 16, 32)) if s is not None and s < last})
    labels = ["last frame seen"] + [f"+{s * r.frame_skip} frames" for s in steps[1:]]
    out.mkdir(parents=True, exist_ok=True)
    for kind, chosen in (("fixed", np.arange(min(per_sheet * sheets, len(r.truth)))), ("worst", worst)):
        for page in range(0, len(chosen), per_sheet):
            rows = [viz.contact_sheet({f"window {i}: truth": r.truth[i], f"window {i}: imagined": r.imagined[i],
                                       f"window {i}: best this latent can show": r.recon[i]}, palette, steps, labels=labels)
                    for i in chosen[page : page + per_sheet]]
            viz.save_png(out / f"{kind}-{page // per_sheet + 1}.png", viz.vstack(rows, pad=10))
    return {"shown_fixed": list(range(min(per_sheet * sheets, len(r.truth)))), "shown_worst": [int(i) for i in worst],
            "miou_per_window": [None if v != v else round(float(v), 3) for v in score]}


# --- command line -----------------------------------------------------------------------------------------
def score(run: Path, bench: Path, windows: int | None = None, frames: tuple[int, ...] = FRAMES, device: str = "cpu",
          pictures: bool = True) -> dict:
    """Score a run on a benchmark dataset (its `val` split) and write <run>/eval/scorecard.json."""
    run, bench = Path(run), Path(bench)
    dev = torch.device(device)
    model, decoder, ckpt = load_run(run, dev)
    ds = EpisodeDataset(bench, "val")
    r = collect(model, decoder, ds, int(ckpt["args"]["frame_skip"]), windows, frames, device=dev)
    card = {"run": run.name, "bench": bench.name, "windows": len(r.truth), "frame_skip": r.frame_skip, "step": ckpt["step"],
            "context_frames": CONTEXT_FRAMES, "read_at_frames": r.horizons, "checks": run_checks(r)}
    out = run / "eval"
    out.mkdir(exist_ok=True)
    if pictures:
        (out / "bench" / "windows.json").parent.mkdir(parents=True, exist_ok=True)
        (out / "bench" / "windows.json").write_text(json.dumps(save_pictures(r, ds.materials.palette, out / "bench")))
    (out / "scorecard.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, help="run directory containing ckpt.pt")
    p.add_argument("--bench", type=Path, help="benchmark dataset; omit to print the scorecard the run already has")
    p.add_argument("--windows", type=int, default=None, help="use only the first N benchmark episodes (default: all)")
    p.add_argument("--frames", type=int, nargs="+", default=list(FRAMES), help="raw frames of imagination to read the checks at")
    p.add_argument("--gates", type=Path, help="TOML file with a [gates] table: check name = threshold")
    p.add_argument("--diff", type=Path, nargs=2, metavar=("OLD", "NEW"), help="compare the scorecards of two runs")
    p.add_argument("--tol", type=float, default=TOLERANCE, help="smallest change that counts in --diff")
    p.add_argument("--no-pictures", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    if args.diff:
        print(format_diff(load_card(args.diff[0]), load_card(args.diff[1]), args.tol))
        return
    if not args.run:
        p.error("give --run, or --diff OLD NEW")
    card = score(args.run, args.bench, args.windows, tuple(args.frames), args.device, not args.no_pictures) if args.bench \
        else load_card(args.run)
    print(format_card(card, load_gates(args.gates) if args.gates else None))
    if args.bench:
        print(f"wrote {args.run / 'eval'}/scorecard.json" + ("" if args.no_pictures else " and eval/bench/*.png"))


if __name__ == "__main__":
    main()
