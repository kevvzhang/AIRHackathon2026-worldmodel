"""Run sandwm on Modal GPUs (https://modal.com).

Datasets and runs live on a Modal volume named `sandwm`, as  data/<dataset>/  and  runs/<run>/.

    # simulate a dataset in the cloud (CPU only, skipped if it already exists)
    uv run modal run cloud.py::generate --name basic64-10k --args "--episodes 10000 --val-episodes 500"

    # train one model on a GPU, then evaluate it; `--args` goes to `python -m sandwm.train`
    uv run modal run cloud.py::train --name vit-k2 --data basic64-10k \
        --args "--preset lewm --frame-skip 2 --steps 20000 --batch 128"

    # train several models side by side, one GPU each, and print a comparison table
    uv run modal run cloud.py::sweep --data basic64-10k \
        --spec "cnn: --steps 20000; vit: --preset lewm --steps 20000"

    # copy a finished run (curves, pictures, metrics, checkpoint) back to this machine
    uv run modal volume get sandwm runs/vit-k2 runs/

`--bench bench-basic-v1` on train, sweep or reevaluate also scores the run on a benchmark dataset
(`sandwm.scorecard`).  `modal run --detach` keeps a job alive if this terminal closes.  Environment
variables read at launch: SANDWM_GPU, a comma-separated order of preference (default "L4,A10G,T4";
also L40S, A100, H100), and SANDWM_MAX_GPUS (default 4), the most GPUs a sweep may hold at once.

Once the agent lab exists (`python -m lab setup`, see lab/README.md), every GPU job started here
is first booked against the lab's budget and refused if it does not fit, and no job may run
longer than the lab's per-run limit.  The lab also runs a limited number of GPU jobs at once, all
launches together (`max_concurrent_gpu_runs`, in place of SANDWM_MAX_GPUS): a job beyond that
waits in line inside its launch command and starts by itself when a slot is free.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import sys
import time
from importlib.metadata import version
from pathlib import Path

import modal


def lab_jobs():
    """The agent lab's bookkeeping (lab/jobs.py).  None in the cloud container, which has no lab package."""
    try:
        from lab import jobs
    except ImportError:
        return None
    return jobs


_limits = lab_jobs().limits() if lab_jobs() else None  # None until `python -m lab setup` has run here

app = modal.App("sandwm")
volume = modal.Volume.from_name("sandwm", create_if_missing=True)
VOL = Path("/vol")
# A comma-separated list is an order of preference: Modal takes the first type it can schedule.
GPU = os.environ.get("SANDWM_GPU", "L4,A10G,T4").split(",")
MAX_GPUS = int(os.environ.get("SANDWM_MAX_GPUS", "4"))
RUN_SECONDS, CPU_SECONDS = 8 * 3600, 2 * 3600  # the longest one GPU job / one CPU job may run
POLL = 10.0  # seconds between two looks at the lab's line for a GPU slot, and at a running job for its result
if _limits:
    MAX_GPUS = max(_limits.max_concurrent_gpu_runs, 1)  # the lab's line decides what starts when (lab/budget.py)
    RUN_SECONDS, CPU_SECONDS = int(60 * _limits.gpu_minutes_per_run), int(60 * _limits.cpu_minutes_per_job)

# Same package versions as the local environment; on Linux the default torch wheel includes CUDA.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*[f"{pkg}=={version(pkg)}" for pkg in ("torch", "numpy", "pillow", "matplotlib")])
    .add_local_python_source("sandwm")
)


# --- what runs in the cloud ---------------------------------------------------------------------------
# Start these through the entry points further down (generate, train, sweep, reevaluate): those
# are the ones that keep the lab's ledger.
@app.function(image=image, volumes={str(VOL): volume}, cpu=8, memory=16384, timeout=CPU_SECONDS)
def generate_job(name: str, args: str = "") -> None:
    """Simulate data/<name> on the volume; `args` are passed to `python -m sandwm.generate`."""
    from sandwm import generate as generator

    out = VOL / "data" / name
    if (out / "meta.json").exists():
        print(f"data/{name} already exists on the volume -- nothing to do")
        return
    generator.main(["--out", str(out), *shlex.split(args)])
    volume.commit()


def _cost(started: float) -> dict:
    """What the ledger needs to know about a finished GPU job."""
    import time

    import torch

    return {"gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "gpu min": round((time.time() - started) / 60, 2)}


@app.function(image=image, volumes={str(VOL): volume}, gpu=GPU, cpu=4, memory=24576, timeout=RUN_SECONDS,
              max_containers=MAX_GPUS)
def train_job(name: str, data: str = "basic64", args: str = "", horizon: int = 16, bench: str = "", minutes: float = 0.0) -> dict:
    """Train runs/<name> on data/<data>, evaluate it, score it on data/<bench>, and return a summary.

    With `minutes` set, training stops itself after 80% of them so that the evaluation still fits.
    """
    import time

    from sandwm import evaluate, scorecard
    from sandwm import train as trainer
    from sandwm.compare import summarise

    started = time.time()
    run = VOL / "runs" / name
    if (run / "ckpt.pt").exists():
        raise FileExistsError(f"runs/{name} already holds a trained model; a run is never overwritten, so choose a new name")
    limit = ["--max-minutes", f"{0.8 * minutes:g}"] if minutes else []
    with tagged(name):
        trainer.main(["--data", str(VOL / "data" / data), "--out", str(run), *limit, *shlex.split(args)])
        volume.commit()  # the checkpoint is safe even if the evaluation fails
        evaluate.main(["--run", str(run), "--horizon", str(horizon)])
        if bench:
            scorecard.main(["--run", str(run), "--bench", str(VOL / "data" / bench)])
    volume.commit()
    return {**summarise(run), **_cost(started)}


@app.function(image=image, volumes={str(VOL): volume}, gpu=GPU, cpu=4, memory=24576, timeout=min(RUN_SECONDS, 3600),
              max_containers=MAX_GPUS)
def evaluate_job(name: str, horizon: int = 16, bench: str = "") -> dict:
    """Evaluate runs/<name> again, e.g. after changing the evaluation code."""
    import time

    from sandwm import evaluate as evaluator
    from sandwm import scorecard
    from sandwm.compare import summarise

    started = time.time()
    run = VOL / "runs" / name
    with tagged(name):
        evaluator.main(["--run", str(run), "--horizon", str(horizon)])
        if bench:
            scorecard.main(["--run", str(run), "--bench", str(VOL / "data" / bench)])
    volume.commit()
    return {**summarise(run), **_cost(started)}


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


# --- what runs on this machine --------------------------------------------------------------------------
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


def fly(job, names: list[str], calls: list[tuple], booking=None, every: float | None = None) -> list:
    """Run one cloud job per entry of `calls` and return what each gave, in the same order: its result, or the
    exception it ended with.

    With a booking from the lab, a job starts when the lab has a GPU slot for it (`booking.poll`), and its minutes
    are settled the moment it reports back, which frees the slot for the next in line.  Without a booking all start
    at once, and Modal runs at most MAX_GPUS of them at a time.
    """
    every = POLL if every is None else every  # seconds between two looks at the line
    waiting, flying, results = dict(zip(names, calls)), {}, {}
    first = True
    while waiting or flying:
        for name, state in (booking.poll(every) if booking else dict.fromkeys(waiting, "go")).items():
            call = waiting.pop(name)
            if state != "go":
                results[name] = RuntimeError("not started: it left the lab's line for a GPU (the lab was stopped, or this "
                                             "launcher was silent for minutes); nothing was charged")
                print(f"{name}: {results[name]}", flush=True)
                continue
            if not first:
                print(f"{name}: a GPU slot is free; starting", flush=True)
            try:
                flying[name] = job.spawn(*call)
            except Exception as err:  # it could not even be handed to the cloud
                results[name] = err
                if booking:
                    booking.done(name, err)
        first = False
        if not flying:
            if waiting:
                time.sleep(every)
            continue
        patience = max((every if waiting else 6 * every) / len(flying), min(every, 1.0))  # seconds to wait for each result in turn
        for name, call in list(flying.items()):
            asked = time.perf_counter()
            try:
                result = call.get(timeout=patience)
            except Exception as err:
                if type(err) is TimeoutError and time.perf_counter() - asked >= patience / 2:
                    continue  # Modal's "not finished yet"; a TimeoutError the job itself raised would come back at once
                result = err
            del flying[name]
            results[name] = result
            if booking:
                booking.done(name, result)
    return [results[name] for name in names]


def launch(job, names: list[str], calls: list[tuple], kind: str, minutes: float, record: list[dict]) -> None:
    """Run one cloud GPU job per entry of `calls`, side by side, inside the lab's budget if there is a lab."""
    jobs = lab_jobs()
    booking = None
    if jobs:
        if app.app_id:  # the Modal app this launch runs as: `modal app list` shows when it stopped
            record = [{**r, "app": app.app_id} for r in record]
        try:
            booking = jobs.book(record, kind, minutes)
        except jobs.Refused as refusal:
            raise SystemExit(f"not started: {refusal}")
    print(f"launching {len(calls)} {kind} job(s) on {'/'.join(GPU)} (at most {MAX_GPUS} at a time): {', '.join(names)}"
          + (f"; {booking.minutes:g} GPU min booked for each" if booking else ""), flush=True)
    waits = booking.waits() if booking else ""
    if waits:
        print(waits, flush=True)
    results = []
    try:
        results = fly(job, names, calls, booking)
    finally:
        if booking:
            booking.close()  # anything that did not report back is charged for the time it had
    report(names, results)
    if any(isinstance(result, Exception) for result in results):
        raise SystemExit(1)


@app.local_entrypoint()
def generate(name: str = "basic64", args: str = "", note: str = "") -> None:
    """Simulate data/<name> on the volume (CPU only); `args` are passed to `python -m sandwm.generate`."""
    jobs = lab_jobs()
    if jobs:
        try:
            jobs.record("generate", dataset=name, args=args, note=note)
        except jobs.Refused as refusal:
            raise SystemExit(f"not started: {refusal}")
    generate_job.remote(name, args)


def train_runs(runs: list[tuple[str, str]], data: str, horizon: int, bench: str, note: str, minutes: float) -> None:
    """Train (name, train args) pairs side by side."""
    cap = minutes or (_limits.gpu_minutes_per_run if _limits else 0.0)  # 0 = train until --steps, however long
    launch(train_job, [name for name, _ in runs], [(name, data, args, horizon, bench, cap) for name, args in runs], "train",
           minutes, [{"run": name, "data": data, "bench": bench, "args": args, "note": note} for name, args in runs])


@app.local_entrypoint()
def train(name: str, data: str = "basic64", args: str = "", horizon: int = 16, bench: str = "", note: str = "",
          minutes: float = 0.0) -> None:
    """Train one run on a GPU, then evaluate it.  `args` are passed to `python -m sandwm.train`."""
    train_runs([(name, args)], data, horizon, bench, note, minutes)


@app.local_entrypoint()
def sweep(spec: str, data: str = "basic64", horizon: int = 16, bench: str = "", note: str = "", minutes: float = 0.0) -> None:
    """Train several runs in parallel.  spec is  "name: train args; other-name: train args"."""
    train_runs([(name.strip(), args.strip()) for name, _, args in (item.partition(":") for item in spec.split(";") if item.strip())],
               data, horizon, bench, note, minutes)


@app.local_entrypoint()
def reevaluate(names: str, horizon: int = 16, bench: str = "") -> None:
    """Evaluate existing runs again, in parallel.  names is comma-separated."""
    runs = [name.strip() for name in names.split(",") if name.strip()]
    minutes = min(10.0, _limits.gpu_minutes_per_run) if _limits else 0.0
    launch(evaluate_job, runs, [(run, horizon, bench) for run in runs], "evaluate", minutes,
           [{"run": run, "bench": bench} for run in runs])
