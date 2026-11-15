"""PreToolUse hook for the lab's agents: the stop switch, and a few fences.

Claude Code runs this before every tool call an agent makes (`python -m lab setup` wires it into
each agent worktree's .claude/settings.local.json).  Exit code 2 blocks the call and shows the
agent the reason printed on stderr; any other exit lets the call through.

The fences catch honest mistakes: writing in someone else's worktree, touching the budget,
launching cloud jobs past the ledger, reaching for the human's other projects.  They are not a
sandbox, and a shell command can get around them; the permission mode is the real boundary.

This file runs as a plain script from the main checkout, so it imports nothing from the lab.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

ROLES = ("eval", "data", "train", "research")
FILE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
SHELL_TOOLS = {"Bash", "PowerShell"}
READ_ONLY_TOOLS = {"Read", "Grep", "Glob"}  # still allowed when the lab is stopped, so an agent can answer questions
HUMAN_FILES = {"limits.toml", "ledger.jsonl", "PROGRAM.md", "STOP"}  # on the board; so is everything under roles/
KEEPERS = {"gaps.md": "eval", "coverage.md": "eval", "features.md": "eval", "gates.toml": "eval", "champion.txt": "eval",
           "datasets.md": "data", "recipes.md": "train", "tracks.md": "research"}

STOPPED = ("STOP: the human has stopped the lab. Do not retry and do not work around this. End your turn now, "
           "without scheduling another wake-up; you will be started again.")

TRIPWIRES = [
    (r"limits\.toml|ledger\.jsonl",
     "The lab's limits and ledger are read with the Read tool or `uv run python -m lab budget`, and changed only by the human."),
    (r"\blab\s+budget\s+(-\S+\s+)*--new\b",
     "A new budget period is the human's to start; it happens when they start the lab. If the budget is in your way, say what "
     "more would buy:  uv run python -m lab msg human ..."),
    (r"\.modal\.toml|\bmodal\s+(token|secret|profile|config|setup|deploy|serve|shell|workspace|environment)\b",
     "Modal credentials, configuration and deployments are off limits."),
    (r"\bmodal\s+run\b",
     "Start cloud jobs with `uv run python -m lab train ...` or `uv run python -m lab generate ...`: they enforce the GPU budget "
     "and record the job."),
    (r"\bmodal\s+(app\s+stop|volume\s+(rm|put|cp|rename|delete|create)|container\s+(stop|exec))\b",
     "Runs and datasets on the volume are never deleted or overwritten by hand, and cloud apps are stopped by ending the "
     "`lab train` command that started them."),
    (r"\bpwa-", "That names one of the human's other Modal projects. The lab uses only the app and the volume called sandwm."),
    (r"\bgit\s+push\b", "The lab pushes nowhere. Share work with `uv run python -m lab publish`."),
    (r"\bgit\s+(worktree\s+(remove|prune|move)|update-ref|branch\s+(-[a-zA-Z]*[dDfMm]\b|--delete|--force|--move)|tag\s+(-d\b|--delete))",
     "Branches, tags and worktrees of the lab are managed by `python -m lab`; leave them as they are."),
    (r"\bgit\s+(checkout|switch)\s+(-\S+\s+)*(main|lab-trunk|lab-board)\s*($|[;&|])",
     "Stay on your own branch. To take in the others' work: `uv run python -m lab sync`."),
]


def normal(path: str, cwd: str | None = None) -> str:
    """An absolute, case-folded path; accepts the /c/... spelling Git Bash uses."""
    path = re.sub(r"^/([a-zA-Z])/", r"\1:/", path) if os.name == "nt" else path
    if not os.path.isabs(path):
        path = os.path.join(cwd or os.getcwd(), path)
    return os.path.normcase(os.path.abspath(path))


def inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("\\/") + os.sep)


def file_fence(path: str, role: str, home: str) -> str | None:
    """Why `role` may not write `path`, or None if it may.  Both paths are already `normal`."""
    lab = os.path.join(home, ".lab")
    board = os.path.join(lab, "board")
    if inside(path, board):
        rel = os.path.relpath(path, board).replace("\\", "/")
        if rel.lower() in {name.lower() for name in HUMAN_FILES} or rel.startswith("roles/"):
            return (f"{rel} on the board is the human's. If it ought to change, say what and why:  "
                    "uv run python -m lab msg human \"<subject>\" --body \"...\"")
        keeper = {name.lower(): who for name, who in KEEPERS.items()}.get(rel.lower())
        if keeper and keeper != role:
            return f"{rel} is kept by the {keeper} agent. Send them what should change:  uv run python -m lab msg {keeper} ..."
        if rel.startswith("journal/") and rel != f"journal/{role}.md":
            return "That is another agent's journal; write in your own with  uv run python -m lab note \"...\""
        if rel.startswith("tracks/") and role != "research":
            return "Track plans are kept by the research agent. Send it what should change:  uv run python -m lab msg research ..."
        return None
    if f"{os.sep}.claude{os.sep}" in path + os.sep:
        return "Claude Code's settings are set up by the human and are not edited by agents."
    for other in ROLES:
        if other != role and inside(path, os.path.join(lab, other)):
            return (f"That file is in the {other} agent's worktree. Change your own copy and publish it, or ask them:  "
                    f"uv run python -m lab msg {other} ...")
    if inside(path, home) and not inside(path, lab):
        return "The main checkout is the human's. Work in your own worktree and share it with  uv run python -m lab publish"
    return None


def decide(event: dict, role: str, home: str) -> str | None:
    """The reason to block this tool call, or None to let it through."""
    home = normal(home)
    tool = str(event.get("tool_name", ""))
    args = event.get("tool_input") or {}
    if os.path.exists(os.path.join(home, ".lab", "board", "STOP")) and tool not in READ_ONLY_TOOLS:
        return STOPPED
    if tool in FILE_TOOLS:
        target = args.get("file_path") or args.get("notebook_path")
        if target:
            return file_fence(normal(str(target), event.get("cwd")), role, home)
    elif tool in SHELL_TOOLS:
        command = str(args.get("command", ""))
        for pattern, reason in TRIPWIRES:
            if re.search(pattern, command, re.IGNORECASE):
                return reason + " (Blocked by the lab's guard. If those words were only part of a message you were writing, reword it.)"
    return None


def utc(seconds: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds))


def beat(role: str, home: str, event: dict) -> None:
    """Leave a trace of this agent's last tool call, and keep the clock of its conversation.

    status/<role>.json is read by `python -m lab status` and by lab/sessions.py, which replaces a
    conversation that has run for too long: `t` and `tool` are the last tool call, `session` is
    Claude Code's id of the conversation, `since` is when it began, `calls` counts its tool
    calls, and `launch` is the key of the launcher process that can replace it (LAB_SESSION,
    set by lab/agent.ps1; empty when no launcher can).
    """
    folder = os.path.join(home, ".lab", "board", "status")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{role}.json")
    launch = os.environ.get("LAB_SESSION", "")
    session = str(event.get("session_id") or launch)
    try:
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
    except (FileNotFoundError, ValueError):
        old = None  # the first call ever, or a file cut short: begin a new record
    except OSError:
        return  # another call of the same agent is writing it at this moment: that one counts
    if isinstance(old, dict) and session and old.get("session") == session and old.get("since"):
        since, calls = str(old["since"]), int(old.get("calls") or 0) + 1
    else:  # a new conversation: it began when its transcript was created
        try:
            since = utc(os.path.getctime(str(event["transcript_path"])))
        except (KeyError, OSError, ValueError):
            since = utc()
        calls = 1
    record = {"t": utc(), "tool": str(event.get("tool_name", "")), "session": session, "since": since, "calls": calls,
              "launch": launch}
    scratch = f"{path}.{os.getpid()}.tmp"  # written beside the file and moved over it, so a reader never sees half of one
    with open(scratch, "w", encoding="utf-8") as f:
        json.dump(record, f)
    try:
        os.replace(scratch, path)
    except OSError:
        os.unlink(scratch)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True)
    p.add_argument("--home", required=True, help="the main checkout")
    args = p.parse_args(argv)
    try:
        event = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace") or "{}")
        reason = decide(event, args.role, args.home)
    except Exception as err:  # a bug here must not take the agent down with it
        print(f"lab guard: {err!r}", file=sys.stderr)
        return 1
    try:
        beat(args.role, normal(args.home), event)
    except Exception as err:  # the heartbeat must neither cost the agent a tool call nor let a blocked one through
        print(f"lab guard (heartbeat): {err!r}", file=sys.stderr)
        if not reason:
            return 1
    if reason:
        print(reason, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
