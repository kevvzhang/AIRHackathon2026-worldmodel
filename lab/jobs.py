"""Bookkeeping around cloud jobs: book GPU minutes before a job starts, settle them when it ends.

cloud.py calls this from its local entry points, so every documented way of starting a GPU job
passes through the ledger.  Where no lab has been set up, `book` returns None and nothing is
enforced.

Booked is not started.  The lab runs a limited number of GPU jobs at once (lab/budget.py), and a job
booked beyond that waits in line.  The launcher asks `Booking.poll` every few seconds which of its
runs may start, and starts each one when it is told "go".
"""

from __future__ import annotations

import threading
import time

from . import board as boards
from . import sessions
from .budget import BEAT_MIN, Ledger, Limits, Refused  # noqa: F401  (Refused is part of this module's interface)
from .paths import Lab, LabError, current_role, git

POLL = 10.0  # seconds between two looks at the line by a launcher that has runs waiting for a GPU slot


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
    """GPU minutes booked for a set of runs.  Call `done` as each reports back and `close` when all are in.

    A run may start only once `poll` has said "go" for it: until then it waits in line for a GPU slot.

    While it is open it writes a heartbeat for each job that has not reported back, so that the ledger can tell a
    job whose launcher was killed from one that is still running or waiting (lab/budget.py).
    """

    def __init__(self, lab: Lab, ledger: Ledger, jobs: dict[str, str], minutes: float, limit: float) -> None:
        self.lab, self.ledger, self.jobs = lab, ledger, dict(jobs)
        self.minutes, self.limit = minutes, limit  # booked per run; the most the cloud lets one job run
        self.booked = time.time()
        self.line = list(self.jobs)  # the runs that have no GPU slot yet, in the order they were booked
        self.began: dict[str, float] = {}  # run -> when it got its slot
        self.news: dict[str, str] = {}  # what `poll` has not handed out yet
        self.asked = 0.0
        self.stopped, self.ids = threading.Event(), list(self.jobs.values())
        self._beat()
        self._look(save=False)  # the runs for which a slot was free at booking have it already
        self.beating = threading.Thread(target=self._beating, name="lab-heartbeat", daemon=True)
        self.beating.start()

    def _look(self, save: bool = True) -> None:
        """Ask the ledger which of the waiting runs may start now."""
        self.asked = time.perf_counter()
        try:
            states = self.ledger.admit([self.jobs[run] for run in self.line])
        except (LabError, OSError) as err:  # the lock is busy, or a file is being edited: the next look will do
            print(f"(could not look at the line for a GPU slot: {err})", flush=True)
            return
        changed = [run for run in self.line if states[self.jobs[run]] != "wait"]
        for run in changed:
            state = states[self.jobs[run]]
            self.line.remove(run)
            self.news[run] = state
            if state == "go":
                self.began[run] = time.time()
            else:  # it ended without starting, and the ledger has its end line
                self.jobs.pop(run)
        if changed and save:  # the ledger has new lines: a run has started after waiting, or has left the line
            _save(self.lab, f"ledger: {', '.join(changed)} left the line for a GPU slot")

    def poll(self, every: float = POLL) -> dict[str, str]:
        """What is new for the runs that waited: {run: "go"} for each that may start now, {run: "gone"} for each that
        lost its place without starting (the lab was stopped, or this launcher was silent for minutes).  The first
        call names the runs that had a slot from the start.  It looks at the ledger at most once in `every` seconds.
        """
        if self.line and time.perf_counter() - self.asked >= every:
            self._look()
        news, self.news = self.news, {}
        return news

    def waits(self) -> str:
        """A sentence for the launcher's output about the runs that have to wait, or "" if none does."""
        if not self.line:
            return ""
        try:
            s = self.ledger.summary()
        except LabError:
            return ""
        ahead = [j["job"] for j in s["waiting"]]
        first = self.jobs[self.line[0]]
        before = ahead.index(first) if first in ahead else 0
        return (f"waiting in line for a GPU: {', '.join(self.line)}. The lab runs at most {s['concurrent']} GPU job(s) at once; "
                f"{len(s['running'])} are running" + (f" and {before} other(s) wait ahead" if before else "") + ". "
                "Each starts by itself when a slot is free, first come, first served. Keep this command running and "
                "do not start the run a second time: its GPU minutes are booked.")

    def _beat(self) -> None:
        for job in list(self.jobs.values()):
            try:
                self.ledger.beat(job)
            except OSError:  # a missed beat is harmless; five in a row would mark the job as cut
                pass

    def _beating(self) -> None:
        while not self.stopped.wait(BEAT_MIN * 60):
            self._beat()

    def done(self, run: str, result) -> None:
        """`result` is the dict the cloud job returned, or whatever went wrong instead."""
        job = self.jobs.pop(run, None)
        if job is None:
            return
        if run in self.line:  # it never got a GPU slot, so there is nothing to pay
            self.line.remove(run)
            self.ledger.settle(job, 0.0, "dropped", error="its launcher ended while the job waited for a GPU; it never started")
        elif isinstance(result, dict) and "gpu min" in result:
            self.ledger.settle(job, result["gpu min"], "ok", gpu=str(result.get("gpu", "?")))
        else:  # failed or cut off: it used at most the time since it got its slot, and at most what the cloud allows one job
            elapsed = (time.time() - self.began.get(run, self.booked)) / 60
            self.ledger.settle(job, min(elapsed, self.limit), "failed", error=repr(result)[:300])

    def close(self) -> None:
        self.stopped.set()
        self.beating.join(timeout=10)
        for run in list(self.jobs):
            self.done(run, "no result: interrupted")
        for job in self.ids:  # a beat written just as its job reported back
            self.ledger.unbeat(job)
        _save(self.lab, "ledger: jobs ended")


def book(runs: list[dict], kind: str = "train", minutes: float = 0.0, cwd=None) -> Booking | None:
    """Book GPU time for `runs` (dicts with a "run" name and anything worth recording).

    Returns None where there is no lab.  Raises Refused when the jobs may not be booked (the budget, the
    per-run limit, a used name, STOP).  A booked run starts when `Booking.poll` says "go" for it.
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
