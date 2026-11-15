"""The lab's GPU budget.

`limits.toml` on the board is the human's: how many GPU minutes the lab may spend in total, how
long one job may run, and how many may run at once.  `ledger.jsonl` beside it gets a "start" line
before every cloud GPU job and an "end" line after it.  A job that started and never reported an
end is charged its full allowance, so the ledger can over-count but never under-count.

The process that launched a job (cloud.py, through lab/jobs.py) writes a heartbeat for it every
minute.  A cloud job ends with its launcher, so when the heartbeat stops for good (the launcher was
killed, its tab closed) the job is "cut": charged up to its last heartbeat plus a few minutes, and
no longer counted as running.  The next booking writes its end line.

The total is counted per budget period.  A period ends when the human starts the lab again from a
standstill (lab/launch.ps1, or `python -m lab budget --new`): a "period" line goes into the ledger
and the count begins at zero, while the jobs of earlier periods stay on record.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .paths import LabError, locked, now, parse_time, stamp

BEAT_MIN = 1.0  # a launcher writes the heartbeat of its jobs this often (lab/jobs.py)
GONE_MIN = 5.0  # a heartbeat this old means the launcher is gone, and the cloud has stopped its jobs
STOP_MIN = 5.0  # charged after the last heartbeat: Modal stopped r0061-r0064 about 2 minutes after their launchers died

NO_WAY_AROUND = ("Do not look for another way to start it. Carry on with work that needs no GPU; if more budget would "
                 "change what the lab can achieve, say so once to the human:  uv run python -m lab msg human ...")


def track_of(run: str) -> str | None:
    """T-001 for a run named t001-..., the naming that marks a run as part of a research track; else None."""
    m = re.match(r"t(\d{3})-", run)
    return f"T-{m[1]}" if m else None


class Refused(LabError):
    """The job may not start.  The message says why and what to do instead."""


@dataclass(frozen=True)
class Limits:
    gpu_minutes_total: float
    gpu_minutes_per_run: float
    max_concurrent_gpu_runs: int
    cpu_minutes_per_job: float = 20.0
    gpu_minutes_tracks: float | None = None  # of the total, the most all research tracks may use; None = a third
    session_hours: float = 6.0  # an agent's conversation is replaced by an empty one after this long; 0 = never (lab/sessions.py)

    @property
    def tracks_cap(self) -> float:
        return self.gpu_minutes_total / 3 if self.gpu_minutes_tracks is None else self.gpu_minutes_tracks

    @classmethod
    def load(cls, path: Path) -> "Limits":
        try:
            raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
            tracks = raw.get("gpu_minutes_tracks")
            return cls(float(raw["gpu_minutes_total"]), float(raw["gpu_minutes_per_run"]), int(raw["max_concurrent_gpu_runs"]),
                       float(raw.get("cpu_minutes_per_job", 20.0)), None if tracks is None else float(tracks),
                       float(raw.get("session_hours", cls.session_hours)))
        except (OSError, KeyError, TypeError, ValueError) as err:
            raise Refused(f"cannot read the lab's limits from {path} ({err!r}); no cloud job may start until the human repairs it")


class Ledger:
    def __init__(self, board: Path) -> None:
        self.board = Path(board)
        self.path = self.board / "ledger.jsonl"
        self.lock = self.board.parent / "ledger.lock"
        self.alive = self.board.parent / "alive"  # one heartbeat file per job whose launcher is running; not on the board

    @property
    def limits(self) -> Limits:
        return Limits.load(self.board / "limits.toml")

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # --- heartbeats: is the launcher of a job still there? ---
    def _beat_file(self, job: str) -> Path:
        return self.alive / (re.sub(r"[^A-Za-z0-9._-]", "_", job) + ".txt")

    def beat(self, job: str) -> None:
        """The launcher of `job` is alive (lab/jobs.py calls this every BEAT_MIN minutes)."""
        self.alive.mkdir(parents=True, exist_ok=True)
        self._beat_file(job).write_text(stamp(), encoding="utf-8")

    def unbeat(self, job: str) -> None:
        self._beat_file(job).unlink(missing_ok=True)

    def last_beat(self, job: str) -> datetime | None:
        try:
            return parse_time(self._beat_file(job).read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    def _cut(self, start: dict) -> dict | None:
        """The end line of a job whose launcher has stopped beating; None if it still beats or never did."""
        last = self.last_beat(start["job"])
        if last is None or now() - last < timedelta(minutes=GONE_MIN):
            return None
        began, cap = parse_time(start["t"]), float(start["cap_min"])
        ended = min(last + timedelta(minutes=STOP_MIN), began + timedelta(minutes=cap))
        return {"event": "end", "job": start["job"], "t": stamp(ended), "gpu_min": round(max((ended - began).total_seconds() / 60, 0.0), 2),
                "status": "cut", "error": f"its launcher stopped at {stamp(last)} (killed, or its tab closed); the cloud ends a job with its launcher"}

    def jobs(self) -> list[dict]:
        """Every GPU job in the order it started: its start line, plus `end` (its end line or None), `charged` minutes
        and `period`, the budget period it was booked in (0 until the first `new_period`, then 1, ...).  A job whose
        launcher has died gets the end line `settle_cut` would write, marked "unsettled" until it is written."""
        events = self.events()
        ends = {e["job"]: e for e in events if e.get("event") == "end"}
        period, found = 0, []
        for e in events:
            if e.get("event") == "period":
                period += 1
            elif e.get("event") == "start":
                end = ends.get(e["job"])
                if end is None and (cut := self._cut(e)):
                    end = {**cut, "unsettled": True}
                found.append({**e, "end": end, "period": period, "charged": float(end["gpu_min"] if end else e["cap_min"])})
        return found

    def _write_cut(self, jobs: list[dict]) -> list[str]:
        """Write the end lines of the jobs that were cut; the caller holds the lock."""
        cut = [j for j in jobs if j["end"] and j["end"].get("unsettled")]
        if cut:
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                for j in cut:
                    f.write(json.dumps({k: v for k, v in j["end"].items() if k != "unsettled"}) + "\n")
            for j in cut:
                self.unbeat(j["job"])
        return [j["job"] for j in cut]

    def settle_cut(self) -> list[str]:
        """Record the end of every job whose launcher has died.  Returns their job ids."""
        with locked(self.lock):
            return self._write_cut(self.jobs())

    def period(self) -> tuple[int, str | None]:
        """The number of the current budget period and when it began (None for the first: it began with the lab)."""
        marks = [e for e in self.events() if e.get("event") == "period"]
        return len(marks), (marks[-1]["t"] if marks else None)

    def current(self, jobs: list[dict] | None = None) -> list[dict]:
        """The jobs of the current budget period: the ones `gpu_minutes_total` is counted over."""
        number = self.period()[0]
        return [j for j in (self.jobs() if jobs is None else jobs) if j["period"] == number]

    def spent(self) -> float:
        """GPU minutes of the current budget period: what finished jobs used, plus the full allowance of every job
        that has not reported back."""
        return sum(job["charged"] for job in self.current())

    def new_period(self, by: str = "human") -> float | None:
        """Start a new budget period: `gpu_minutes_total` counts from zero again.  The human's, when the lab is started.

        Returns the minutes the period that ends had used, or None if it had used none (then it simply goes on).
        Nothing is forgotten: the jobs stay in the ledger, and a run name stays used for good.
        """
        with locked(self.lock):
            used = sum(job["charged"] for job in self.current())
            if not used:
                return None
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"event": "period", "t": stamp(), "by": by, "closed_min": round(used, 2)}) + "\n")
            return used

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
            self._write_cut(jobs)  # jobs whose launcher died give back their minutes and their place for good
            if len(set(names)) != len(names):
                raise Refused(f"run names must differ: {', '.join(names)}")
            if kind == "train":
                used = sorted(set(names) & {j["run"] for j in jobs if j.get("kind") == "train"})
                if used:
                    raise Refused(f"run name already used: {', '.join(used)}. Every training run needs a new name.")
            counted = self.current(jobs)  # the total and the tracks' ceiling count the current budget period only
            spent = sum(j["charged"] for j in counted)
            if spent + cap * len(names) > limits.gpu_minutes_total + 1e-9:
                raise Refused(f"GPU budget: {spent:.1f} of {limits.gpu_minutes_total:g} minutes are spent or booked, so "
                              f"{len(names)} more job(s) of up to {cap:g} min do not fit. {NO_WAY_AROUND}")
            on_tracks = [name for name in names if track_of(name)]
            if on_tracks:  # side lines may not crowd out the main line
                used = sum(j["charged"] for j in counted if track_of(j["run"]))
                if used + cap * len(on_tracks) > limits.tracks_cap + 1e-9:
                    raise Refused(f"track budget: research tracks have used or booked {used:.1f} of the {limits.tracks_cap:g} GPU "
                                  f"minutes they may use together, so {len(on_tracks)} more track job(s) of up to {cap:g} min do "
                                  "not fit. The rest of the budget is the main line's. Tell the research agent; do not rename the run.")
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

    def settle(self, job: str, gpu_min: float, status: str, at: str | None = None, **extra) -> None:
        """Record that a job has ended (at `at`, else now) and how many GPU minutes it really used."""
        with locked(self.lock):
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"event": "end", "job": job, "t": at or stamp(), "gpu_min": round(float(gpu_min), 2),
                                    "status": status, **extra}) + "\n")
        self.unbeat(job)

    def record(self, **event) -> None:
        """A line for the record that costs no GPU time, e.g. a dataset being generated."""
        with locked(self.lock):
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"event": "note", "t": stamp(), **event}) + "\n")

    def summary(self) -> dict:
        """The budget as it stands.  `spent`, `by` and `tracks` are of the current budget period; `before` is what
        earlier periods used, `since` when this one began (None: with the lab)."""
        jobs, limits = self.jobs(), self.limits
        number, since = self.period()
        counted = [j for j in jobs if j["period"] == number]
        by: dict[str, float] = {}
        tracks: dict[str, float] = {}
        for j in counted:
            by[j.get("by", "?")] = by.get(j.get("by", "?"), 0.0) + j["charged"]
            if track_of(j["run"]):
                tracks[track_of(j["run"])] = tracks.get(track_of(j["run"]), 0.0) + j["charged"]
        return {"spent": sum(j["charged"] for j in counted), "total": limits.gpu_minutes_total, "per_run": limits.gpu_minutes_per_run,
                "concurrent": limits.max_concurrent_gpu_runs, "running": self.running(), "by": by, "jobs": len(jobs),
                "tracks": tracks, "tracks_cap": limits.tracks_cap, "period": number, "since": since,
                "before": sum(j["charged"] for j in jobs if j["period"] < number)}
