"""Compare evaluated runs side by side.

    uv run python -m sandwm.compare runs/cnn-k1 runs/vit-k2

Each run must have been through `sandwm.evaluate`.  Imagination is compared a fixed number of raw
simulator frames into the future, so runs with different frame skips are judged over the same
stretch of simulated time.  See "How to read the evaluation" in the README for the columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarise(run: Path, frames_ahead: int = 16) -> dict:
    """The handful of numbers worth comparing across runs."""
    run = Path(run)
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    val = [r for r in rows if "val_pred" in r][-1]
    timing = [r for r in rows if "sec_per_step" in r][-1]
    ev = json.loads((run / "eval" / "metrics.json").read_text())
    roll, skip = ev["rollout"], ev["frame_skip"]
    i = min(frames_ahead // skip, len(roll["horizon"])) - 1
    lm, mi, ch = roll["latent_mse"], roll["miou"], roll["changed_cell_accuracy"]
    probes = ev["probes"]
    return {
        "steps": val["step"],
        "skip": skip,
        "min": round(timing["step"] * timing["sec_per_step"] / 60, 1),
        "rank": round(val["emb_rank"]),
        "1-step MSE model/frozen": round(val["val_pred"] / val["val_copy"], 3),
        f"+{frames_ahead}f MSE model/frozen": round(lm["model"][i] / lm["copy"][i], 3),
        f"+{frames_ahead}f MSE no-action/frozen": round(lm["no_action"][i] / lm["copy"][i], 3),
        "mIoU model/frozen/ceiling": "/".join(f"{mi[k][i]:.3f}" for k in ("model", "frozen", "ceiling")),
        "changed-cell acc model/frozen/ceiling": "/".join(f"{ch[k][i]:.3f}" for k in ("model", "frozen", "ceiling")),
        "probe R2 amount": round(_mean(v for k, v in probes.items() if "amount" in k), 3),
        "probe R2 position": round(_mean(v for k, v in probes.items() if "centre" in k), 3),
    }


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def format_table(summaries: dict[str, dict]) -> str:
    """One row per run, one column per quantity."""
    columns = ["run"] + list(next(iter(summaries.values())))
    rows = [[name] + [str(v) for v in s.values()] for name, s in summaries.items()]
    widths = [max(len(str(c)) for c in col) for col in zip(columns, *rows)]
    lines = [" | ".join(str(c).ljust(w) for c, w in zip(r, widths)) for r in [columns] + rows]
    return "\n".join([lines[0], "-|-".join("-" * w for w in widths)] + lines[1:])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+", help="run directories")
    p.add_argument("--frames-ahead", type=int, default=16, help="raw frames of imagination to compare at")
    args = p.parse_args(argv)
    names = [run.name for run in args.runs]
    labels = names if len(set(names)) == len(names) else [str(run) for run in args.runs]
    print(format_table({label: summarise(run, args.frames_ahead) for label, run in zip(labels, args.runs)}))


if __name__ == "__main__":
    main()
