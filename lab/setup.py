"""Create the lab on this machine: the board, the trunk, and one worktree per agent.  Safe to repeat."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from . import board as boards
from .paths import BENCH_SEEDS, BOARD_BRANCH, DEFAULT_BENCH, ROLES, TRUNK, Lab, LabError, git, git_ok

TEMPLATE = Path(__file__).parent / "template" / "board"

# What an agent does all day, allowed without a permission check.  Everything else is decided by
# the session's permission mode (in `auto` mode, by Claude Code's safety classifier).
ROUTINE = (
    "uv run python -m lab", "uv run python -m lab *",
    "uv run python -m sandwm.*",
    "uv run pytest", "uv run pytest *",
    "uv sync", "uv sync *",
    "git status", "git status *", "git diff", "git diff *", "git log", "git log *", "git show *",
    "git add *", "git commit *", "git merge *", "git branch --show-current", "git rev-parse *",
    "uv run modal app list", "uv run modal app list *", "uv run modal app logs *", "uv run modal volume ls *",
)


def branch_exists(name: str, cwd: Path) -> bool:
    return git_ok("rev-parse", "--verify", "--quiet", f"refs/heads/{name}", cwd=cwd)


def agent_settings(lab: Lab, role: str) -> dict:
    """Claude Code settings for one agent's worktree: routine commands, extra folders, and the guard hook."""
    parts = [str(p).replace("\\", "/") for p in (sys.executable, lab.home / "lab" / "guard.py")]
    parts += ["--role", role, "--home", str(lab.home).replace("\\", "/")]
    return {
        "permissions": {
            "allow": [f"{shell}({pattern})" for shell in ("Bash", "PowerShell") for pattern in ROUTINE],
            "additionalDirectories": [str(lab.board), str(lab.runs), str(lab.datasets)],
        },
        "env": {"LAB_ROLE": role, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        "hooks": {"PreToolUse": [{"matcher": "*", "hooks": [
            {"type": "command", "command": " ".join(f'"{p}"' if " " in p else p for p in parts), "timeout": 30}]}]},
    }


def fill(text: str, lab: Lab) -> str:
    """Put this machine's paths into a template."""
    for key, value in {"HOME": lab.home, "LAB": lab.state, "BOARD": lab.board, "RUNS": lab.runs, "DATA": lab.datasets,
                       "BENCH": DEFAULT_BENCH, "TRUNK": TRUNK}.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def copy_templates(lab: Lab) -> list[str]:
    """Copy every template the board does not have yet; files already on the board are never overwritten."""
    added = []
    for source in sorted(TEMPLATE.rglob("*")):
        if source.is_dir():
            continue
        rel = source.relative_to(TEMPLATE)
        name = "." + rel.name[4:] if rel.name.startswith("dot.") else rel.name  # dot.gitignore -> .gitignore
        target = lab.board / rel.parent / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(fill(source.read_text(encoding="utf-8"), lab), encoding="utf-8", newline="\n")
            added.append(target.relative_to(lab.board).as_posix())
    return added


def setup(lab: Lab, sync: bool = True, bench: bool = True, say=print) -> None:
    home = lab.home
    if not git_ok("cat-file", "-e", "HEAD:lab/guard.py", cwd=home):
        raise LabError("the lab's tooling (lab/) is not committed on this branch yet; commit it first, then run setup again")
    lab.state.mkdir(exist_ok=True)

    # 1. the board: a branch with no history in common with the code, checked out at .lab/board
    if not (lab.board / ".git").exists():
        if branch_exists(BOARD_BRANCH, home):
            git("worktree", "add", str(lab.board), BOARD_BRANCH, cwd=home)
        else:
            git("worktree", "add", "--orphan", "-b", BOARD_BRANCH, str(lab.board), cwd=home)
        say(f"board      {lab.board}  (branch {BOARD_BRANCH})")
    added = copy_templates(lab)
    if added:
        boards.commit(lab.board, "board: add " + ", ".join(added[:6]) + (" ..." if len(added) > 6 else ""))
        say(f"board      added {len(added)} file(s): {', '.join(added[:8])}{' ...' if len(added) > 8 else ''}")

    # 2. the trunk the agents publish to; `main` is left alone from here on
    if not branch_exists(TRUNK, home):
        git("branch", TRUNK, "HEAD", cwd=home)
        say(f"trunk      branch {TRUNK} created at {git('rev-parse', '--short', 'HEAD', cwd=home)}")

    # 3. one worktree and one branch per agent
    for role in ROLES:
        tree = lab.worktree(role)
        if not (tree / ".git").exists():
            if branch_exists(f"lab-{role}", home):
                git("worktree", "add", str(tree), f"lab-{role}", cwd=home)
            else:
                git("worktree", "add", "-b", f"lab-{role}", str(tree), TRUNK, cwd=home)
            say(f"{role:10s} {tree}  (branch lab-{role})")
        settings = tree / ".claude" / "settings.local.json"
        settings.parent.mkdir(exist_ok=True)
        wanted = json.dumps(agent_settings(lab, role), indent=2) + "\n"
        if not settings.exists() or settings.read_text(encoding="utf-8") != wanted:  # leave a running agent's file alone
            settings.write_text(wanted, encoding="utf-8", newline="\n")
        if sync:
            say(f"{role:10s} installing the Python environment (uv sync) ...")
            if subprocess.call(["uv", "sync", "--quiet"], cwd=tree):
                raise LabError(f"uv sync failed in {tree}")

    # 4. shared folders for what comes back from the cloud, and the first benchmark
    lab.runs.mkdir(exist_ok=True)
    lab.datasets.mkdir(exist_ok=True)
    if bench and not (lab.datasets / DEFAULT_BENCH / "meta.json").exists():
        say(f"bench      simulating {DEFAULT_BENCH} (256 episodes, seed {BENCH_SEEDS + 1}) ...")
        if subprocess.call(["uv", "run", "python", "-m", "sandwm.generate", "--out", str(lab.datasets / DEFAULT_BENCH),
                            "--episodes", "0", "--val-episodes", "256", "--seed", str(BENCH_SEEDS + 1)], cwd=home):
            raise LabError("could not generate the benchmark dataset")
    say("the lab is set up.  Nothing is running.  See lab/README.md for how to start it.")
