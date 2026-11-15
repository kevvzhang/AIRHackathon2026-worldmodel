"""Run sandwm on Modal GPUs (https://modal.com).

Datasets and runs live on a Modal volume named `sandwm`, as  data/<dataset>/  and  runs/<run>/.

    # simulate a dataset in the cloud (CPU only, skipped if it already exists)
    uv run modal run cloud.py::generate --name basic64-10k --args "--episodes 10000 --val-episodes 500"

    # train one model on a GPU, then evaluate it; `--args` goes to `python -m sandwm.train`
    uv run modal run --detach cloud.py::train --name vit-k2 --data basic64-10k \
        --args "--preset lewm --frame-skip 2 --steps 20000 --batch 128"

    # train several models side by side, one GPU each, and print a comparison table
    uv run modal run --detach cloud.py::sweep --data basic64-10k \
        --spec "cnn: --steps 20000; vit: --preset lewm --steps 20000"

    # copy a finished run (curves, pictures, metrics, checkpoint) back to this machine
    uv run modal volume get sandwm runs/vit-k2 runs/

`--detach` keeps the job alive if this terminal closes.  Environment variables read at launch:
SANDWM_GPU, a comma-separated order of preference (default "L4,A10G,T4"; also L40S, A100, H100),
and SANDWM_MAX_GPUS (default 4), the most GPUs a sweep may hold at once.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import sys
from importlib.metadata import version
from pathlib import Path

import modal

app = modal.App("sandwm")
volume = modal.Volume.from_name("sandwm", create_if_missing=True)
VOL = Path("/vol")
# A comma-separated list is an order of preference: Modal takes the first type it can schedule.
GPU = os.environ.get("SANDWM_GPU", "L4,A10G,T4").split(",")
MAX_GPUS = int(os.environ.get("SANDWM_MAX_GPUS", "4"))

# Same package versions as the local environment; on Linux the default torch wheel includes CUDA.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*[f"{pkg}=={version(pkg)}" for pkg in ("torch", "numpy", "pillow", "matplotlib")])
    .add_local_python_source("sandwm")
)


@app.function(image=image, volumes={str(VOL): volume}, cpu=8, memory=16384, timeout=2 * 3600)
def generate(name: str = "basic64", args: str = "") -> None:
    """Simulate data/<name> on the volume; `args` are passed to `python -m sandwm.generate`."""
    from sandwm import generate as generator

    out = VOL / "data" / name
    if (out / "meta.json").exists():
        print(f"data/{name} already exists on the volume -- nothing to do")
        return
    generator.main(["--out", str(out), *shlex.split(args)])
    volume.commit()


@app.function(image=image, volumes={str(VOL): volume}, gpu=GPU, cpu=4, memory=24576, timeout=8 * 3600,
              max_containers=MAX_GPUS)
def train(name: str, data: str = "basic64", args: str = "", horizon: int = 16) -> dict:
    """Train runs/<name> on data/<data>, evaluate it, and return a summary of the results."""
    from sandwm import evaluate
    from sandwm import train as trainer
    from sandwm.compare import summarise

    run = VOL / "runs" / name
    with tagged(name):
        trainer.main(["--data", str(VOL / "data" / data), "--out", str(run), *shlex.split(args)])
        volume.commit()  # the checkpoint is safe even if the evaluation fails
        evaluate.main(["--run", str(run), "--horizon", str(horizon)])
    volume.commit()
    return summarise(run)


@app.function(image=image, volumes={str(VOL): volume}, gpu=GPU, cpu=4, memory=24576, timeout=3600, max_containers=MAX_GPUS)
def evaluate(name: str, horizon: int = 16) -> dict:
    """Evaluate runs/<name> again, e.g. after changing the evaluation code."""
    from sandwm import evaluate as evaluator
    from sandwm.compare import summarise

    run = VOL / "runs" / name
    with tagged(name):
        evaluator.main(["--run", str(run), "--horizon", str(horizon)])
    volume.commit()
    return summarise(run)


class _Tagged:
    """A stream that prefixes every line, so the logs of parallel runs can be told apart."""

    def __init__(self, stream, tag: str) -> None:
        self.stream, self.tag, self.at_line_start = stream, tag, True

    def write(self, text: str) -> int:
        for line in text.splitlines(keepends=True):
            self.stream.write((self.tag if self.at_line_start else "") + line)
            self.at_line_start = line.endswith("\n")
        return len(text)

    def flush(self) -> None:
        self.stream.flush()


def tagged(name: str):
    return contextlib.redirect_stdout(_Tagged(sys.stdout, f"[{name}] "))


def report(names: list[str], results: list) -> None:
    from sandwm.compare import LEGEND, format_table

    done = {name: result for name, result in zip(names, results) if not isinstance(result, Exception)}
    print("\n=== results ===")
    if done:
        print(format_table(done))
        print(LEGEND)
    for name, result in zip(names, results):
        if isinstance(result, Exception):
            print(f"{name} FAILED: {result!r}")


@app.local_entrypoint()
def sweep(spec: str, data: str = "basic64", horizon: int = 16) -> None:
    """Train several runs in parallel.  spec is  "name: train args; other-name: train args"."""
    jobs = [(name.strip(), data, args.strip(), horizon)
            for name, _, args in (item.partition(":") for item in spec.split(";") if item.strip())]
    names = [job[0] for job in jobs]
    print(f"launching {len(jobs)} runs on {'/'.join(GPU)} (at most {MAX_GPUS} at a time): {', '.join(names)}", flush=True)
    report(names, list(train.starmap(jobs, return_exceptions=True)))


@app.local_entrypoint()
def reevaluate(names: str, horizon: int = 16) -> None:
    """Evaluate existing runs again, in parallel.  names is comma-separated."""
    runs = [name.strip() for name in names.split(",") if name.strip()]
    report(runs, list(evaluate.starmap([(run, horizon) for run in runs], return_exceptions=True)))
