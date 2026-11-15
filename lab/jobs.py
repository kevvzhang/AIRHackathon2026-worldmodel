"""Bookkeeping around cloud jobs: book GPU minutes before a job starts, settle them when it ends.

cloud.py calls this from its local entry points, so every documented way of starting a GPU job
passes through the ledger.  Where no lab has been set up, `book` returns None and nothing is
enforced.
"""

from __future__ import annotations

import time

from . import board as boards
from . import sessions
from .budget import Ledger, Limits, Refused  # noqa: F401  (Refused is part of this module's interface)
from .paths import Lab, LabError, current_role, git


def find(cwd=None) -> Lab | None:
    """The lab this checkout belongs to, or None if there is none or it has not been set up."""
    try:
        lab = Lab.find(cwd)
    except (LabError, OSError):  # not a git repository, or no git at all
        return None
    return lab if lab.ready else None


def limits(cwd=None) -> Limits | None:
    """The lab's limits; None without a lab.  An unreadable limits file counts as none here and refuses at launch."""
    lab = find(cwd)
    try:
        return Ledger(lab.board).limits if lab else None
    except LabError:
        return None


def _save(lab: Lab, message: str) -> None:
    try:
        boards.commit(lab.board, message)
    except LabError as err:  # the ledger file itself is already written; its history can catch up later
        print(f"(could not record the ledger in the board's history: {err})")


class Booking:
    """GPU minutes booked for a set of runs.  Call `done` as each reports back and `close` when all are in."""

    def __init__(self, lab: Lab, ledger: Ledger, jobs: dict[str, str], minutes: float, limit: float) -> None:
        self.lab, self.ledger, self.jobs = lab, ledger, dict(jobs)
        self.minutes, self.limit = minutes, limit  # booked per run; the most the cloud lets one job run
        self.started = time.time()

    def done(self, run: str, result) -> None:
        """`result` is the dict the cloud job returned, or whatever went wrong instead."""
        job = self.jobs.pop(run, None)
        if job is None:
            return
        if isinstance(result, dict) and "gpu min" in result:
            self.ledger.settle(job, result["gpu min"], "ok", gpu=str(result.get("gpu", "?")))
        else:  # failed or cut off: it used at most the time since launch, and at most what the cloud allows one job
            elapsed = (time.time() - self.started) / 60
            self.ledger.settle(job, min(elapsed, self.limit), "failed", error=repr(result)[:300])

    def close(self) -> None:
        for run in list(self.jobs):
            self.done(run, "no result: interrupted")
        _save(self.lab, "ledger: jobs ended")


def book(runs: list[dict], kind: str = "train", minutes: float = 0.0, cwd=None) -> Booking | None:
    """Book GPU time for `runs` (dicts with a "run" name and anything worth recording).

    Returns None where there is no lab.  Raises Refused when the jobs may not start.
    """
    lab = find(cwd)
    if lab is None:
        return None
    ledger = Ledger(lab.board)
    who, commit = current_role(cwd), git("rev-parse", "--short", "HEAD", cwd=cwd, check=False)
    sessions.hold_jobs(lab.board, who)  # a conversation that is due for a fresh start hands over first
    ids = ledger.reserve([{**r, "by": who, "commit": commit} for r in runs], kind, minutes)
    names = [r["run"] for r in runs]
    if kind == "train":  # keep the exact code of every run reachable, whatever happens to the branch later
        for name in names:
            git("tag", f"run/{name}", "HEAD", cwd=cwd, check=False)
    _save(lab, f"ledger: {who} starts {', '.join(names)}")
    limit = ledger.limits.gpu_minutes_per_run
    return Booking(lab, ledger, dict(zip(names, ids)), minutes or limit, limit)


def record(what: str, cwd=None, **fields) -> None:
    """Note a job that needs no GPU (a dataset being generated).  Raises Refused if the lab is stopped."""
    lab = find(cwd)
    if lab is None:
        return
    if (lab.board / "STOP").exists():
        raise Refused("the human has stopped the lab (STOP is on the board); no cloud job may start")
    sessions.hold_jobs(lab.board, current_role(cwd))
    Ledger(lab.board).record(what=what, by=current_role(cwd), commit=git("rev-parse", "--short", "HEAD", cwd=cwd, check=False), **fields)
    _save(lab, f"ledger: {what} {fields.get('dataset', '')}".strip())
