"""Where the lab's things live, and the few helpers every other module needs."""

from __future__ import annotations

import contextlib
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROLES = ("eval", "data", "train")
TRUNK = "lab"  # the branch the agents publish to; `main` stays the human's
BOARD_BRANCH = "lab-board"
DEFAULT_BENCH = "bench-basic-v1"
BENCH_SEEDS = 9000  # dataset seeds from here up are reserved for benchmarks, which are never trained on

# Who looks after which code.  Publishing a change to someone else's file is allowed but announced;
# the tooling in lab/ and cloud.py is the human's and cannot be published by an agent.
AREAS = {
    "eval": ("sandwm/evaluate.py", "sandwm/scorecard.py", "sandwm/compare.py", "sandwm/metrics.py", "sandwm/viz.py"),
    "data": ("sandwm/sim.py", "sandwm/materials.py", "sandwm/scenes.py", "sandwm/generate.py", "sandwm/data.py"),
    "train": ("sandwm/models.py", "sandwm/train.py"),
}
HUMAN_ONLY = ("lab/", "cloud.py")


class LabError(Exception):
    """Something the caller should read and act on.  The command line prints it without a traceback."""


def git(*args: str, cwd: Path | str | None = None, check: bool = True) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and done.returncode:
        raise LabError(f"git {' '.join(args)} failed:\n{(done.stderr or done.stdout).strip()}")
    return done.stdout.strip()


def git_ok(*args: str, cwd: Path | str | None = None) -> bool:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True).returncode == 0


@dataclass(frozen=True)
class Lab:
    """The folders of one lab.  `home` is the main checkout: the folder that holds the real .git directory."""

    home: Path

    @classmethod
    def find(cls, cwd: Path | str | None = None) -> "Lab":
        """The lab of the repository `cwd` is in, whichever of its worktrees that is."""
        return cls(Path(git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=cwd)).parent)

    @property
    def state(self) -> Path:
        return self.home / ".lab"

    @property
    def board(self) -> Path:
        return self.state / "board"

    @property
    def runs(self) -> Path:
        return self.state / "runs"

    @property
    def datasets(self) -> Path:
        """Local copies of datasets and benchmarks (not `data`: that is the data agent's worktree)."""
        return self.state / "datasets"

    def worktree(self, role: str) -> Path:
        return self.state / role

    @property
    def ready(self) -> bool:
        """True once `python -m lab setup` has run here."""
        return (self.board / "limits.toml").exists()


def current_role(cwd: Path | str | None = None) -> str:
    """eval, data or train inside an agent's session or worktree; otherwise human."""
    named = os.environ.get("LAB_ROLE", "").strip()
    if named:
        return named
    branch = git("branch", "--show-current", cwd=cwd, check=False)
    return branch[4:] if branch.startswith("lab-") and branch[4:] in ROLES else "human"


# --- time -------------------------------------------------------------------------------------------
def now() -> datetime:
    return datetime.now(timezone.utc)


def stamp(t: datetime | None = None) -> str:
    return (t or now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def ago(text: str) -> str:
    """'3 min ago' for a time written by `stamp`."""
    minutes = (now() - parse_time(text)).total_seconds() / 60
    if minutes < 1:
        return "just now"
    return f"{minutes:.0f} min ago" if minutes < 120 else f"{minutes / 60:.1f} h ago"


# --- locking ------------------------------------------------------------------------------------------
@contextlib.contextmanager
def locked(path: Path, wait: float = 60.0, stale: float = 120.0):
    """Hold a cross-process lock: a file created exclusively.  A lock older than `stale` seconds is abandoned."""
    path = Path(path)
    deadline = time.monotonic() + wait
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except (FileExistsError, PermissionError):
            with contextlib.suppress(OSError):
                if time.time() - path.stat().st_mtime > stale:
                    path.unlink()
                    continue
            if time.monotonic() > deadline:
                raise LabError(f"{path} is held by another process; try again in a moment")
            time.sleep(0.2)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        with contextlib.suppress(OSError):
            path.unlink()
