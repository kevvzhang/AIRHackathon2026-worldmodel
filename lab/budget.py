"""The lab's GPU budget.

`limits.toml` on the board is the human's: how many GPU minutes the lab may spend in total, how
long one job may run, and how many may run at once.  `ledger.jsonl` beside it gets a "start" line
before every cloud GPU job and an "end" line after it.  A job that started and never reported an
end is charged its full allowance, so the ledger can over-count but never under-count.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from .paths import LabError, locked, now, parse_time, stamp

NO_WAY_AROUND = ("Do not look for another way to start it. Carry on with work that needs no GPU; if more budget would "
                 "change what the lab can achieve, say so once to the human:  uv run python -m lab msg human ...")


class Refused(LabError):
    """The job may not start.  The message says why and what to do instead."""


@dataclass(frozen=True)
class Limits:
    gpu_minutes_total: float
    gpu_minutes_per_run: float
    max_concurrent_gpu_runs: int
    cpu_minutes_per_job: float = 20.0

    @classmethod
    def load(cls, path: Path) -> "Limits":
        try:
            raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
            return cls(float(raw["gpu_minutes_total"]), float(raw["gpu_minutes_per_run"]), int(raw["max_concurrent_gpu_runs"]),
                       float(raw.get("cpu_minutes_per_job", 20.0)))
        except (OSError, KeyError, TypeError, ValueError) as err:
            raise Refused(f"cannot read the lab's limits from {path} ({err!r}); no cloud job may start until the human repairs it")


class Ledger:
    def __init__(self, board: Path) -> None:
        self.board = Path(board)
        self.path = self.board / "ledger.jsonl"
        self.lock = self.board.parent / "ledger.lock"

    @property
    def limits(self) -> Limits:
        return Limits.load(self.board / "limits.toml")

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def jobs(self) -> list[dict]:
        """Every GPU job in the order it started: its start line, plus `end` (its end line or None) and `charged` minutes."""
        events = self.events()
        ends = {e["job"]: e for e in events if e.get("event") == "end"}
        return [{**e, "end": ends.get(e["job"]), "charged": float(ends[e["job"]]["gpu_min"] if e["job"] in ends else e["cap_min"])}
                for e in events if e.get("event") == "start"]

    def spent(self) -> float:
        """GPU minutes used by finished jobs plus the full allowance of every job that has not reported back."""
        return sum(job["charged"] for job in self.jobs())

    def running(self) -> list[dict]:
        """Jobs with no end line that are young enough to still be running."""
        t = now()
        return [j for j in self.jobs() if j["end"] is None and t - parse_time(j["t"]) < timedelta(minutes=float(j["cap_min"]) + 10)]

    def reserve(self, runs: list[dict], kind: str = "train", minutes: float = 0.0) -> list[str]:
        """Book up to `minutes` (0 = the per-run limit) for each run, or raise Refused.

        Each entry of `runs` needs a "run" name and may carry anything else worth recording
        (data, bench, args, note, by, commit).  Returns one job id per run, for `settle`.
        """
        names = [r["run"] for r in runs]
        with locked(self.lock):
            limits = self.limits
            if (self.board / "STOP").exists():
                raise Refused("the human has stopped the lab (STOP is on the board); no cloud job may start")
            cap = float(minutes) if minutes else limits.gpu_minutes_per_run
            if cap > limits.gpu_minutes_per_run:
                raise Refused(f"{cap:g} minutes is more than the {limits.gpu_minutes_per_run:g} one job may take. {NO_WAY_AROUND}")
            jobs = self.jobs()
            if len(set(names)) != len(names):
                raise Refused(f"run names must differ: {', '.join(names)}")
            if kind == "train":
                used = sorted(set(names) & {j["run"] for j in jobs if j.get("kind") == "train"})
                if used:
                    raise Refused(f"run name already used: {', '.join(used)}. Every training run needs a new name.")
            spent = sum(j["charged"] for j in jobs)
            if spent + cap * len(names) > limits.gpu_minutes_total + 1e-9:
                raise Refused(f"GPU budget: {spent:.1f} of {limits.gpu_minutes_total:g} minutes are spent or booked, so "
                              f"{len(names)} more job(s) of up to {cap:g} min do not fit. {NO_WAY_AROUND}")
            busy = self.running()
            if len(busy) + len(names) > limits.max_concurrent_gpu_runs:
                raise Refused(f"{len(busy)} GPU job(s) are running ({', '.join(j['run'] for j in busy)}) and at most "
                              f"{limits.max_concurrent_gpu_runs} may run at once. Start this one when one of them has finished.")
            t = stamp()
            counts = {name: sum(j["run"] == name for j in jobs) for name in names}
            ids = [f"{name}#{counts[name] + 1}" for name in names]
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                for run, job in zip(runs, ids):
                    f.write(json.dumps({"event": "start", "job": job, "kind": kind, "t": t, "cap_min": cap, **run}) + "\n")
            return ids

    def settle(self, job: str, gpu_min: float, status: str, **extra) -> None:
        """Record that a job has ended and how many GPU minutes it really used."""
        with locked(self.lock):
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"event": "end", "job": job, "t": stamp(), "gpu_min": round(float(gpu_min), 2),
                                    "status": status, **extra}) + "\n")

    def record(self, **event) -> None:
        """A line for the record that costs no GPU time, e.g. a dataset being generated."""
        with locked(self.lock):
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"event": "note", "t": stamp(), **event}) + "\n")

    def summary(self) -> dict:
        jobs, limits = self.jobs(), self.limits
        by: dict[str, float] = {}
        for j in jobs:
            by[j.get("by", "?")] = by.get(j.get("by", "?"), 0.0) + j["charged"]
        return {"spent": sum(j["charged"] for j in jobs), "total": limits.gpu_minutes_total, "per_run": limits.gpu_minutes_per_run,
                "concurrent": limits.max_concurrent_gpu_runs, "running": self.running(), "by": by, "jobs": len(jobs)}
