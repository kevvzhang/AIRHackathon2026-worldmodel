"""Fresh starts: every agent's conversation is replaced by an empty one from time to time.

A conversation that runs for many hours narrows: each step builds on the one before, and in the
end the agent only polishes the idea it already has.  Claude Code cannot clear a running
conversation from outside, so the launcher (lab/agent.ps1) ends the session's process and starts
a new one in the same terminal.  This module decides when.

- The guard hook keeps a clock for each conversation in status/<role>.json: when it began, how
  many tool calls it has made, when the last one was, and which launcher process runs it.
- A conversation is *due* once it is older than `session_hours` (limits.toml) and has done some
  work.
- The agent then asks for the fresh start itself and leaves a handover
  (`python -m lab fresh TEXT`).  Until it has, it starts no new cloud job.
- The launcher asks for it (`python -m lab fresh --idle`) when a due conversation sits idle.

Either way the request is a file, status/<role>.fresh, which the launcher acts on within seconds.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import board as boards
from .budget import Ledger, Limits, Refused
from .paths import ROLES, Lab, LabError, git, now, parse_time, stamp

MIN_CALLS = 50     # tool calls: a conversation that has hardly worked has nothing to shake off
IDLE_MINUTES = 20  # without a tool call, before the launcher replaces a due conversation unasked
EARLY_HOURS = 1.0  # an agent may ask for a fresh start before it is due, but not sooner than this
# So that the three do not all start afresh at the same moment, data's and train's limits are a little longer.
STAGGER = {"data": 1.1, "train": 1.2}


@dataclass(frozen=True)
class Conversation:
    role: str
    since: datetime | None  # when it began; None if the guard has not seen one
    seen: datetime | None   # its last tool call
    calls: int              # tool calls so far
    launch: str             # the key of the launcher process that can replace it; "" if none can
    limit: float            # hours it may run; 0 = as long as it likes

    @property
    def age(self) -> float:
        """Hours since it began."""
        return (now() - self.since).total_seconds() / 3600 if self.since else 0.0

    @property
    def idle(self) -> float:
        """Minutes since its last tool call."""
        return (now() - self.seen).total_seconds() / 60 if self.seen else 0.0

    @property
    def due(self) -> bool:
        return self.limit > 0 and self.since is not None and self.age >= self.limit and self.calls >= MIN_CALLS

    @property
    def mine(self) -> bool:
        """True inside the conversation itself and in its launcher: the processes that carry its key."""
        return bool(self.launch) and os.environ.get("LAB_SESSION", "") == self.launch


def read(board: Path, role: str) -> Conversation:
    seen = boards.last_seen(board, role) or {}
    try:
        limit = Limits.load(board / "limits.toml").session_hours
    except LabError:
        limit = Limits.session_hours

    def when(key: str) -> datetime | None:
        try:
            return parse_time(str(seen[key]))
        except (KeyError, ValueError):
            return None

    try:
        calls = int(seen.get("calls") or 0)
    except (TypeError, ValueError):
        calls = 0
    return Conversation(role, when("since"), when("t"), calls, str(seen.get("launch") or ""), limit * STAGGER.get(role, 1.0))


def request_file(board: Path, role: str) -> Path:
    """Its existence tells the launcher to replace `role`'s conversation now."""
    return board / "status" / f"{role}.fresh"


def busy(board: Path, role: str) -> list[str]:
    """The cloud GPU jobs `role` has running: they would die with its conversation."""
    return [job["run"] for job in Ledger(board).running() if job.get("by") == role]


def describe(c: Conversation) -> str:
    """A few words for `lab status`."""
    if c.since is None:
        return ""
    text = f"conversation {c.age:.1f} h old"
    if not c.launch:
        return text + " (old launcher: no fresh starts until it is restarted)"
    if c.limit <= 0:
        return text
    if c.due:
        return text + ", fresh start due"
    return text + (f", fresh start in {c.limit - c.age:.1f} h" if c.age < c.limit else ", fresh start once it has done more work")


def due_notice(c: Conversation) -> str:
    return (f"FRESH START DUE: this conversation is {c.age:.1f} h old (limit {c.limit:g} h). Start no new cloud job. When "
            "the jobs you have running are in and recorded, commit your work, then as the last thing you do:\n"
            '  uv run python -m lab fresh "<handover: what is half done and where, what you promised, what you wait for>"\n'
            '  (PROGRAM.md, "Fresh starts")')


def hold_jobs(board: Path, role: str) -> None:
    """Raise Refused if `role` is an agent whose conversation is due: it should hand over, not start something new."""
    if role not in ROLES:
        return
    c = read(board, role)
    if c.due and c.mine:
        raise Refused(f"your conversation is {c.age:.1f} h old and due for a fresh start (PROGRAM.md, \"Fresh starts\"), so it "
                      "starts no new cloud job. Record what is still running when it ends, commit your work, then:  "
                      'uv run python -m lab fresh "<handover>"   The next conversation starts the job.')


def _request(board: Path, role: str, why: str) -> None:
    path = request_file(board, role)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"t": stamp(), "why": why}) + "\n", encoding="utf-8")


def _journal(board: Path, role: str, text: str) -> None:
    boards.note(board, role, text)
    try:
        boards.commit(board, f"[{role}] handover")
    except LabError:  # the note is in the journal file; the board's history catches up with the next commit
        pass


def ask(lab: Lab, role: str, text: str) -> str:
    """The agent asks for its own fresh start and leaves a handover.  Raises LabError when now is not the moment."""
    board = lab.board
    c = read(board, role)
    if not c.mine:
        raise LabError("this conversation cannot be replaced from here: it was not started by a launcher that does fresh "
                       "starts. Carry on as you are; the human can restart you with lab\\launch.ps1.")
    if not text.strip():
        raise LabError("say what the next conversation must know:\n  uv run python -m lab fresh "
                       '"<what is half done and where, what you promised to whom, what you are waiting for>"')
    if not c.due and c.age < EARLY_HOURS:
        raise LabError(f"this conversation is only {c.age * 60:.0f} minutes old; a fresh start is for one that has run for a while")
    running = busy(board, role)
    if running:
        raise LabError(f"still running: {', '.join(running)}. Ending this conversation would stop {'it' if len(running) == 1 else 'them'}. "
                       "Wait, record the results, then ask again.")
    tree = lab.worktree(role)
    dirty = git("status", "--porcelain", "--untracked-files=no", cwd=tree, check=False) if tree.exists() else ""
    if dirty:
        raise LabError("commit what is in your worktree first, even unfinished (it need not be published), so that the next "
                       "conversation finds it:\n" + dirty)
    _journal(board, role, f"HANDOVER before a fresh start (conversation {c.age:.1f} h old, {c.calls} tool calls): {text}")
    _request(board, role, "asked")
    return ("Handover recorded in your journal. This conversation ends within seconds and a new one starts in this terminal.\n"
            "Call no other tool. End your turn now.")


def if_idle(lab: Lab, role: str) -> str | None:
    """Called by the launcher every few minutes: ask for the fresh start of a conversation that is due and sits idle."""
    board = lab.board
    if (board / "STOP").exists() or request_file(board, role).exists():
        return None
    c = read(board, role)
    if not (c.mine and c.due and c.idle >= IDLE_MINUTES) or busy(board, role):
        return None
    _journal(board, role, f"(the launcher ended the conversation: {c.age:.1f} h old, due for a fresh start and idle for "
                          f"{c.idle:.0f} min; it wrote no handover)")
    _request(board, role, "idle")
    return f"{role}: due and idle, a fresh start is requested"
