"""The board: the lab's shared state, a folder of small text files with one keeper each.

It is its own git branch (lab-board) checked out at .lab/board, so every change is recorded and
nothing that gets overwritten is lost.  Messages are files in inbox/<role>/; reading one does
not consume it, acknowledging it moves it to inbox/<role>/done/.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from .paths import git, locked, now, stamp

RECIPIENTS = ("eval", "data", "train", "research", "human")


def slug(text: str, limit: int = 48) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit].strip("-") or "message"


def send(board: Path, sender: str, to: str, subject: str, body: str = "") -> Path:
    """Put a message in `to`'s inbox and return its path."""
    folder = board / "inbox" / to
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{now().strftime('%Y%m%d-%H%M%S')}-{sender}-{slug(subject)}.md"
    n = 1
    while path.exists():  # two messages with the same subject in the same second
        n += 1
        path = path.with_name(f"{path.stem.rsplit('~', 1)[0]}~{n}.md")
    path.write_text(f"from: {sender}\nto: {to}\ntime: {stamp()}\nsubject: {subject}\n\n{body.strip()}\n", encoding="utf-8", newline="\n")
    return path


def unread(board: Path, role: str) -> list[Path]:
    return sorted((board / "inbox" / role).glob("*.md"))


def subject(message: Path) -> str:
    for line in message.read_text(encoding="utf-8", errors="replace").splitlines()[:6]:
        if line.startswith("subject:"):
            return line[8:].strip()
    return message.stem


def acknowledge(board: Path, role: str, names: list[str] | None = None) -> list[Path]:
    """Move messages (all unread ones, or those named) from `role`'s inbox to its done/ folder."""
    done = board / "inbox" / role / "done"
    done.mkdir(parents=True, exist_ok=True)
    wanted = None if names is None else {Path(name).name for name in names}
    moved = []
    for path in unread(board, role):
        if wanted is None or path.name in wanted:
            shutil.move(str(path), str(done / path.name))
            moved.append(done / path.name)
    return moved


def note(board: Path, role: str, text: str) -> None:
    """Add a dated line to `role`'s journal: what was done, what was learnt, what comes next."""
    journal = board / "journal" / f"{role}.md"
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8", newline="\n") as f:
        f.write(f"- {stamp()}  {' '.join(text.split())}\n")


def last_note(board: Path, role: str) -> str:
    journal = board / "journal" / f"{role}.md"
    lines = [line for line in journal.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith("- ")] \
        if journal.exists() else []
    return lines[-1][2:] if lines else ""


def last_seen(board: Path, role: str) -> dict | None:
    """When `role`'s session last made a tool call, as written by the guard hook."""
    try:
        return json.loads((board / "status" / f"{role}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def gaps(board: Path) -> list[dict]:
    """The entries of gaps.md: headings of the form  ### G-007 | title | severity | status."""
    path = board / "gaps.md"
    found = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"###\s+(G-\d+)\s*\|\s*(.*?)\s*\|\s*(\w+)\s*\|\s*(\w[\w ]*)\s*$", line)
            if m:
                found.append({"id": m[1], "title": m[2], "severity": m[3].lower(), "status": m[4].strip().lower()})
    return found


def coverage(board: Path) -> list[dict]:
    """The entries of coverage.md: headings of the form  ### C-001 | title | priority | status."""
    path = board / "coverage.md"
    found = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"###\s+(C-\d+)\s*\|\s*(.*?)\s*\|\s*(\w+)\s*\|\s*(\w[\w ]*)\s*$", line)
            if m:
                found.append({"id": m[1], "title": m[2], "priority": m[3].lower(), "status": m[4].strip().lower()})
    return found


def tracks(board: Path) -> list[dict]:
    """The entries of tracks.md: headings of the form  ### T-001 | title | status."""
    path = board / "tracks.md"
    found = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"###\s+(T-\d+)\s*\|\s*(.*?)\s*\|\s*(\w[\w ]*)\s*$", line)
            if m:
                found.append({"id": m[1], "title": m[2], "status": m[3].strip().lower()})
    return found


def champion(board: Path) -> str:
    """The run the eval agent currently holds up as the best: the first word of champion.txt."""
    path = board / "champion.txt"
    words = path.read_text(encoding="utf-8").split() if path.exists() else []
    return words[0] if words else ""


def commit(board: Path, message: str) -> bool:
    """Record the board as it now stands in its git history.  False if nothing had changed."""
    with locked(board.parent / "board.lock"):
        git("add", "-A", cwd=board)
        if not git("status", "--porcelain", cwd=board):
            return False
        git("commit", "-q", "-m", message, cwd=board)
        return True
