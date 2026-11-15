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

ROLES = ("eval", "data", "train")
FILE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
SHELL_TOOLS = {"Bash", "PowerShell"}
READ_ONLY_TOOLS = {"Read", "Grep", "Glob"}  # still allowed when the lab is stopped, so an agent can answer questions
HUMAN_FILES = {"limits.toml", "ledger.jsonl", "PROGRAM.md", "STOP"}  # on the board; so is everything under roles/
KEEPERS = {"gaps.md": "eval", "features.md": "eval", "gates.toml": "eval", "champion.txt": "eval",
           "datasets.md": "data", "recipes.md": "train"}

STOPPED = ("STOP: the human has stopped the lab. Do not retry and do not work around this. End your turn now, "
           "without scheduling another wake-up; you will be started again.")

TRIPWIRES = [
    (r"limits\.toml|ledger\.jsonl",
     "The lab's limits and ledger are read with the Read tool or `uv run python -m lab budget`, and changed only by the human."),
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
    (r"\bgit\s+(checkout|switch)\s+(-\S+\s+)*(main|lab|lab-board)\s*($|[;&|])",
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


def beat(role: str, home: str, tool: str) -> None:
    """Leave a trace of this agent's last tool call, for `python -m lab status`."""
    folder = os.path.join(home, ".lab", "board", "status")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{role}.json"), "w", encoding="utf-8") as f:
        json.dump({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tool": tool}, f)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True)
    p.add_argument("--home", required=True, help="the main checkout")
    args = p.parse_args(argv)
    try:
        event = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace") or "{}")
        reason = decide(event, args.role, args.home)
        beat(args.role, normal(args.home), str(event.get("tool_name", "")))
    except Exception as err:  # a bug here must not take the agent down with it
        print(f"lab guard: {err!r}", file=sys.stderr)
        return 1
    if reason:
        print(reason, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
