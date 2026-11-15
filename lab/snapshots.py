"""Named snapshots of the lab, and going back to one.

A snapshot is two git tags: snapshot/<name>/code on the trunk and snapshot/<name>/board on the
board.  Restoring puts the trunk, the three agent branches and the board back to those commits.

Nothing is thrown away by a restore: the state being left, uncommitted work included, is tagged
kept/<time>/... first.  Three things are never rewound: the ledger (money already spent stays
spent), the `main` branch, and what is on the cloud volume.
"""

from __future__ import annotations

import json

from . import board as boards
from .budget import Ledger
from .paths import BOARD_BRANCH, ROLES, TRUNK, Lab, LabError, git, git_ok, locked, now
from .setup import agent_settings


def tags(name: str) -> tuple[str, str]:
    return f"snapshot/{name}/code", f"snapshot/{name}/board"


def has_tag(lab: Lab, tag: str) -> bool:
    return git_ok("rev-parse", "--verify", "--quiet", f"refs/tags/{tag}", cwd=lab.home)


def take(lab: Lab, name: str) -> str:
    """Name the lab as it is now: the trunk's commit and the board's."""
    code, board = tags(name)
    if has_tag(lab, code) or has_tag(lab, board):
        raise LabError(f"a snapshot called {name} already exists; choose another name")
    boards.commit(lab.board, f"[human] snapshot {name}")  # include board edits nobody has recorded yet
    git("tag", code, TRUNK, cwd=lab.home)
    git("tag", board, BOARD_BRANCH, cwd=lab.home)
    return (f"snapshot {name}: code at {git('rev-parse', '--short', code, cwd=lab.home)} (the trunk), "
            f"board at {git('rev-parse', '--short', board, cwd=lab.home)}")


def restore(lab: Lab, name: str) -> list[str]:
    """Put the trunk, the agent branches and the board back to snapshot `name`.  Returns what was done."""
    code, board_tag = tags(name)
    if not (has_tag(lab, code) and has_tag(lab, board_tag)):
        raise LabError(f"there is no snapshot called {name}.  `git tag --list \"snapshot/*\"` lists them.")
    if not (lab.board / "STOP").exists():
        raise LabError("stop the lab first (uv run python -m lab stop) and close the agents' tabs: "
                       "a restore must not happen under agents that are still working")
    code_sha = git("rev-parse", f"{code}^{{commit}}", cwd=lab.home)
    board_sha = git("rev-parse", f"{board_tag}^{{commit}}", cwd=lab.home)
    stamp = now().strftime("%Y%m%d-%H%M%S")
    done = []

    # 1. keep what is there now, uncommitted work included, under kept/<time>/...
    roles = [role for role in ROLES if (lab.worktree(role) / ".git").exists()]  # a lab set up before a role existed lacks it
    for role in roles:
        tree = lab.worktree(role)
        if git("status", "--porcelain", cwd=tree):
            try:
                git("add", "-A", cwd=tree)
                git("commit", "-q", "-m", f"[restore] uncommitted work of the {role} agent at {stamp}", cwd=tree)
            except LabError as err:
                done.append(f"warning: the uncommitted work in {tree} could not be kept ({str(err).splitlines()[-1]})")
        git("tag", f"kept/{stamp}/{role}", "HEAD", cwd=tree)
    git("tag", f"kept/{stamp}/trunk", TRUNK, cwd=lab.home)
    boards.commit(lab.board, f"[human] the board before restoring snapshot {name}")
    git("tag", f"kept/{stamp}/board", BOARD_BRANCH, cwd=lab.home)
    done.append(f"kept the state you are leaving as the tags kept/{stamp}/trunk, /board and one per agent (/eval, /data, ...)")

    # 2. the code: the trunk and every agent's branch
    git("update-ref", f"refs/heads/{TRUNK}", code_sha, cwd=lab.home)
    for role in roles:
        tree = lab.worktree(role)
        git("reset", "-q", "--hard", code_sha, cwd=tree)
        git("clean", "-q", "-fd", cwd=tree)
        settings = tree / ".claude" / "settings.local.json"  # ignored by git, so untouched; rewritten in case it was lost
        settings.parent.mkdir(exist_ok=True)
        settings.write_text(json.dumps(agent_settings(lab, role), indent=2) + "\n", encoding="utf-8", newline="\n")
    done.append(f"code: the trunk and the agents' branches are at {code_sha[:7]}")

    # 3. the board: exactly the snapshot's files, plus the ledger as it is now.  A new commit, so
    #    the board's history goes on unbroken.
    with locked(Ledger(lab.board).lock):
        had_ledger = git_ok("cat-file", "-e", "HEAD:ledger.jsonl", cwd=lab.board)
        git("rm", "-r", "-q", "--ignore-unmatch", ".", cwd=lab.board)
        git("checkout", board_sha, "--", ".", cwd=lab.board)
        if had_ledger:
            git("checkout", "HEAD", "--", "ledger.jsonl", cwd=lab.board)
        if git("status", "--porcelain", cwd=lab.board):
            git("commit", "-q", "-m", f"[human] restored snapshot {name}; the ledger is kept", cwd=lab.board)
    spent = Ledger(lab.board).summary()
    done.append(f"board: as in the snapshot ({board_sha[:7]}); the ledger is kept: {spent['spent']:.1f} of {spent['total']:g} GPU min used")
    done.append("not touched: the main branch, runs and datasets on the cloud volume, and fetched runs under .lab")
    done.append("the lab is still stopped.  To start again:  uv run python -m lab go   then   lab\\launch.ps1")
    return done
