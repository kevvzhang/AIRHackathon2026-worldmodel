"""Command line of the lab:   uv run python -m lab <command>

Agents run these from their own worktree, the human from the main checkout.

    status                         one screen: budget, trunk, runs, inboxes, who is alive, open gaps
    msg TO SUBJECT [--body TEXT]   leave a message for eval, data, train or human
    inbox                          list your unread messages
    ack [NAME ...]                 file messages as handled (all of yours if none is named)
    note TEXT                      add a line to your journal
    fresh HANDOVER                 end this conversation and start an empty one in its place (PROGRAM.md, "Fresh starts")
    save [-m TEXT]                 record hand edits to board files in the board's history
    sync                           merge the trunk into your branch
    publish                        merge the trunk, run the tests, move the trunk to your commit
    train NAME --data D -- ARGS    train on a cloud GPU within the budget, score it on the benchmark, fetch the results
    generate NAME -- ARGS          simulate a dataset in the cloud
    fetch NAME [--data] [--again]  copy a run (or a dataset) from the cloud volume into the shared folder
    runs                           what has been run, what it cost, how it scored
    budget                         GPU minutes spent, booked and left

For the human only:

    budget --new                   start a new budget period: the total counts from zero again (lab/launch.ps1 does
                                   this when it starts the lab from a standstill)
    setup                          create the board, the trunk and the three worktrees (safe to repeat)
    stop [--keep-cloud]            stop every agent at its next tool call, and the lab's cloud jobs
    go                             lift the stop, so that agents can be started again
    snapshot NAME                  name the lab as it is now (the trunk's code and the board)
    restore NAME                   go back to a snapshot; what you leave is kept under kept/<time>/... tags
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import board as boards
from . import scores, sessions
from .budget import Ledger
from .paths import (AREAS, BENCH_SEEDS, DEFAULT_BENCH, HUMAN_ONLY, ROLES, TRUNK, Lab, LabError, ago, current_role, git, git_ok,
                    now, parse_time, stamp)

SHIPPED = ("sandwm", "cloud.py", "lab", "pyproject.toml", "uv.lock")  # what a cloud job is built from
VOLUME = "sandwm"


def sh(cmd: list[str], cwd: Path | str | None = None) -> int:
    """Run an external command and let its output through.  In one place so that tests can replace it."""
    return subprocess.call(cmd, cwd=cwd)


def need_lab(lab: Lab) -> None:
    if not lab.ready:
        raise LabError("the lab is not set up here. From the main checkout run:  uv run python -m lab setup")


def need_agent_branch(cwd: Path) -> str:
    branch = git("branch", "--show-current", cwd=cwd)
    if not branch.startswith("lab-") or branch[4:] not in ROLES:
        raise LabError(f"this is branch {branch or '(none)'}; sync and publish work from an agent's branch (lab-eval, lab-data, lab-train)")
    return branch


def check_name(name: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,47}", name):
        raise LabError(f"{name!r}: names are 3-48 characters from a-z, 0-9, dot, dash and underscore")
    return name


# --- status -------------------------------------------------------------------------------------------
def job_state(job: dict) -> str:
    if job["end"]:
        return job["end"]["status"]
    age = (now() - parse_time(job["t"])).total_seconds() / 60
    return "running" if age < float(job["cap_min"]) + 10 else "lost"


def runs_text(lab: Lab, limit: int = 12) -> str:
    jobs = Ledger(lab.board).jobs()[-limit:]
    if not jobs:
        return "no cloud GPU job has been run yet"
    champion = boards.champion(lab.board)
    rows = [("run", "kind", "by", "started", "min", "state", "gates", "note")]
    for j in jobs:
        rows.append((j["run"] + (" *" if j["run"] == champion else ""), j.get("kind", "?"), j.get("by", "?"), ago(j["t"]),
                     f"{j['charged']:.1f}", job_state(j), scores.gate_line(lab.runs / j["run"], lab.board).replace("gates ", ""),
                     str(j.get("note", ""))[:60]))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows]
    return "\n".join(lines) + (f"\n* = champion ({champion})" if champion else "")


def budget_text(lab: Lab) -> str:
    s = Ledger(lab.board).summary()
    running = "; ".join(f"{j['run']} by {j.get('by', '?')}, started {ago(j['t'])}" for j in s["running"]) or "none"
    by = ", ".join(f"{who} {minutes:.1f}" for who, minutes in sorted(s["by"].items())) or "nothing yet"
    on_tracks = ", ".join(f"{track} {minutes:.1f}" for track, minutes in sorted(s["tracks"].items())) or "nothing yet"
    since = f"\ncounted since the lab was started {ago(s['since'])}; before that: {s['before']:.1f} min" if s["since"] else ""
    return (f"GPU minutes: {s['spent']:.1f} spent or booked of {s['total']:g}  ({max(s['total'] - s['spent'], 0):.1f} left){since}\n"
            f"limits: one job <= {s['per_run']:g} min, <= {s['concurrent']} at once\n"
            f"running: {running}\nby agent: {by}\n"
            f"research tracks: {sum(s['tracks'].values()):.1f} of the {s['tracks_cap']:g} min they may use together "
            f"({on_tracks}); that ceiling is inside the total above, not on top of it")


def status_text(lab: Lab, role: str) -> str:
    board = lab.board
    s = Ledger(board).summary()
    stopped = "   ** STOPPED by the human **" if (board / "STOP").exists() else ""
    lines = [f"sandwm lab  {stamp()}  you are: {role}{stopped}"]
    running = ", ".join(f"{j['run']} ({j.get('by', '?')}, {ago(j['t'])})" for j in s["running"]) or "none"
    since = f" since the lab was started {ago(s['since'])}" if s["since"] else ""
    lines.append(f"budget    {s['spent']:.1f} of {s['total']:g} GPU min spent or booked{since}; one job <= {s['per_run']:g} min, "
                 f"<= {s['concurrent']} at once; running: {running}")

    ahead = git("rev-list", "--count", f"main..{TRUNK}", cwd=lab.home, check=False)
    lines.append(f"trunk     {TRUNK} @ {git('log', '-1', '--format=%h %s', TRUNK, cwd=lab.home, check=False)}"
                 + (f"   ({ahead} commit(s) ahead of main)" if ahead not in ("", "0") else ""))
    for r in ROLES:
        counts = git("rev-list", "--left-right", "--count", f"{TRUNK}...lab-{r}", cwd=lab.home, check=False).split()
        seen = boards.last_seen(board, r)
        alive = f"last tool call {ago(seen['t'])}" if seen else "has not run yet"
        talk = sessions.describe(sessions.read(board, r))
        branch = f"branch {counts[1]} ahead / {counts[0]} behind trunk" if len(counts) == 2 else "no branch"
        lines.append(f"{r:9s} {alive}; {talk + '; ' if talk else ''}{branch}; inbox {len(boards.unread(board, r))}"
                     + (f"\n          last note: {boards.last_note(board, r)[:150]}" if boards.last_note(board, r) else ""))
    lines.append(f"human     inbox {len(boards.unread(board, 'human'))}" + ("" if role != "human" else "  (messages from the agents to you)"))

    champion = boards.champion(board)
    gate = scores.gate_line(lab.runs / champion, board) if champion else ""
    card, thresholds = (scores.card(lab.runs / champion) if champion else None), scores.gates(board)
    failing = f"; failing: {', '.join(scores.failing(card, thresholds))}" if card and thresholds and scores.failing(card, thresholds) else ""
    lines.append(f"champion  {champion or 'none yet'}" + (f"   {gate}{failing}" if gate else ""))

    found = boards.gaps(board)
    still_open = [g for g in found if g["status"] not in ("fixed", "closed", "wontfix")]
    worst = ", ".join(f"{g['id']} {g['title']}" for g in still_open if g["severity"] == "blocker")
    lines.append(f"gaps      {len(still_open)} open of {len(found)}" + (f"; blockers: {worst}" if worst else ""))
    unseen = [c for c in boards.coverage(board) if c["status"] != "covered"]
    if boards.coverage(board):
        first = ", ".join(f"{c['id']} {c['title']}" for c in unseen if c["priority"] == "high")
        lines.append(f"coverage  {len(unseen)} situation(s) the evaluation does not cover yet, of {len(boards.coverage(board))} listed"
                     + (f"; high: {first}" if first else ""))
    side = boards.tracks(board)
    if side:
        active = ", ".join(f"{t['id']} {t['title']}" for t in side if t["status"] == "active") or "none"
        lines.append(f"tracks    {len(side)} listed; active: {active}; GPU min used {sum(s['tracks'].values()):.1f} of {s['tracks_cap']:g}"
                     " (a ceiling inside the budget above, not extra)")
    lines.append("runs      " + runs_text(lab, 5).replace("\n", "\n          "))
    mine = boards.unread(board, role)
    if mine:
        lines.append(f"your inbox ({len(mine)}):")
        lines += [f"  {p.name}   {boards.subject(p)}" for p in mine[:10]]
        lines.append(f"  read them in {board / 'inbox' / role}; then  uv run python -m lab ack")
    own = sessions.read(board, role) if role in ROLES else None
    if own and own.due and own.mine:
        lines.append(sessions.due_notice(own))
    return "\n".join(lines)


# --- sharing code ----------------------------------------------------------------------------------------
def sync(cwd: Path) -> None:
    """Merge the trunk into the current agent branch."""
    need_agent_branch(cwd)
    if git("status", "--porcelain", "--untracked-files=no", cwd=cwd):
        raise LabError("you have uncommitted changes; commit them before taking in the trunk")
    if git_ok("merge-base", "--is-ancestor", TRUNK, "HEAD", cwd=cwd):
        return
    if subprocess.run(["git", "merge", "--no-edit", TRUNK], cwd=cwd, capture_output=True).returncode:
        files = git("diff", "--name-only", "--diff-filter=U", cwd=cwd, check=False)
        raise LabError("the trunk does not merge cleanly into your branch. Conflicts in:\n" + files +
                       "\nResolve them, `git add` the files and `git commit`; then run the command again.")


def publish(lab: Lab, cwd: Path, role: str) -> str:
    """Merge the trunk, run the tests, and move the trunk forward to this commit."""
    need_agent_branch(cwd)
    if git("status", "--porcelain", "--untracked-files=no", cwd=cwd):
        raise LabError("you have uncommitted changes; commit them first")
    for _ in range(4):
        old = git("rev-parse", TRUNK, cwd=cwd)
        sync(cwd)
        new = git("rev-parse", "HEAD", cwd=cwd)
        if new == old:
            return f"nothing to publish: the trunk is already at {new[:7]}"
        changed = git("diff", "--name-only", f"{old}..{new}", cwd=cwd).splitlines()
        if role != "human":
            tooling = [f for f in changed if f.startswith(HUMAN_ONLY)]
            if tooling:
                raise LabError("lab/ and cloud.py are the human's and are not published by agents:\n  " + "\n  ".join(tooling) +
                               f"\nTake them back (git checkout {TRUNK} -- <file>, then commit) and tell the human what you need: "
                               "uv run python -m lab msg human ...")
        if sh(["uv", "run", "pytest", "-q"], cwd=cwd):
            raise LabError("the tests fail, so nothing was published. Fix them (or the code) and publish again.")
        # move the trunk only if nobody else moved it meanwhile; otherwise merge again and retest
        if git_ok("update-ref", f"refs/heads/{TRUNK}", new, old, cwd=cwd):
            others = {who: [f for f in changed if f in files] for who, files in AREAS.items() if who != role}
            notice = "".join(f"\nnote: you changed {', '.join(files)} in the {who} agent's area; tell them what and why "
                             f"(uv run python -m lab msg {who} ...)" for who, files in others.items() if files)
            return f"published: the trunk is now at {new[:7]} ({len(changed)} file(s) changed){notice}"
    raise LabError("the trunk kept moving while the tests ran; try again")


# --- cloud jobs --------------------------------------------------------------------------------------------
def ready_to_ship(cwd: Path, role: str) -> None:
    dirty = git("status", "--porcelain", "--", *SHIPPED, cwd=cwd)
    if dirty:
        raise LabError("these files would go to the cloud but are not committed:\n" + dirty + "\nCommit them and publish first.")
    if role != "human" and not git_ok("merge-base", "--is-ancestor", "HEAD", TRUNK, cwd=cwd):
        raise LabError("this commit is not on the trunk yet. Publish it first:  uv run python -m lab publish\n"
                       "(cloud jobs run published code, so that the others can load what they produce)")


def fetch(lab: Lab, name: str, kind: str = "runs", again: bool = False) -> int:
    """Copy runs/<name> or data/<name> from the cloud volume into the lab's shared folder.

    The download is merged into what is already there: files that exist only locally (a scorecard
    computed on the laptop, notes) are kept, files that exist in both are replaced by the volume's.
    """
    target = (lab.runs if kind == "runs" else lab.datasets) / name
    if target.exists() and not again:
        print(f"already here: {target}   (--again downloads it again and keeps files that exist only here)")
        return 0
    lab.state.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=lab.state, prefix="fetch-") as tmp:
        code = sh(["uv", "run", "modal", "volume", "get", VOLUME, f"{kind}/{name}", tmp], cwd=Path.cwd())
        got = Path(tmp) / name if (Path(tmp) / name).is_dir() else Path(tmp)
        if code or not any(got.iterdir()):
            print(f"could not fetch {kind}/{name} from the volume")
            return code or 1
        shutil.copytree(got, target, dirs_exist_ok=True)
    return 0


def train(lab: Lab, cwd: Path, role: str, a, rest: list[str]) -> int:
    need_lab(lab)
    name = check_name(a.name)
    ready_to_ship(cwd, role)
    code = sh(["uv", "run", "modal", "run", "cloud.py::train", "--name", name, "--data", a.data, f"--args={shlex.join(rest)}",
               "--horizon", str(a.horizon), "--bench", a.bench, "--note", a.note, "--minutes", str(a.minutes)], cwd=cwd)
    job = next((j for j in reversed(Ledger(lab.board).jobs()) if j["run"] == name), None)
    if job is None:
        print(f"{name} was not started (see the message above).")
        return code or 1
    state = job_state(job)
    print(f"\n{name}: {state}, {job['charged']:.1f} GPU min charged.  {budget_text(lab).splitlines()[0]}")
    if state != "ok":
        return code or 1
    if fetch(lab, name) == 0:
        run, champion = lab.runs / name, boards.champion(lab.board)
        print(f"results are in {run}")
        if scores.card(run):
            sh(["uv", "run", "python", "-m", "sandwm.scorecard", "--run", str(run), "--gates", str(lab.board / "gates.toml")], cwd=cwd)
            if champion and champion != name and scores.card(lab.runs / champion):
                sh(["uv", "run", "python", "-m", "sandwm.scorecard", "--diff", str(lab.runs / champion), str(run)], cwd=cwd)
    return 0


def generate(lab: Lab, cwd: Path, role: str, a, rest: list[str]) -> int:
    need_lab(lab)
    name = check_name(a.name)
    ready_to_ship(cwd, role)
    given = [arg[7:] for arg in rest if arg.startswith("--seed=")] + [b for a, b in zip(rest, rest[1:]) if a == "--seed"]
    seed = int(given[-1]) if given else 0
    if name.startswith("bench-") != (seed >= BENCH_SEEDS):
        raise LabError(f"benchmarks are named bench-... and use seeds from {BENCH_SEEDS} up; training data uses neither. "
                       f"Got name {name!r} with seed {seed}.")
    return sh(["uv", "run", "modal", "run", "cloud.py::generate", "--name", name, f"--args={shlex.join(rest)}", "--note", a.note], cwd=cwd)


# --- stopping -----------------------------------------------------------------------------------------------
def stop_cloud(lab: Lab) -> list[str]:
    """Stop the lab's cloud app wherever it is running.  Other apps in the workspace are not touched."""
    listing = subprocess.run(["uv", "run", "modal", "app", "list", "--json"], cwd=lab.home, capture_output=True, text=True)
    stopped = []
    for entry in json.loads(listing.stdout or "[]"):
        if entry.get("description") == VOLUME and entry.get("state") not in ("stopped", "stopping"):
            if sh(["uv", "run", "modal", "app", "stop", "-y", entry["app_id"]], cwd=lab.home) == 0:
                stopped.append(entry["app_id"])
    return stopped


# --- the command line ------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True, metavar="command")
    sub.add_parser("status")
    m = sub.add_parser("msg")
    m.add_argument("to", choices=boards.RECIPIENTS)
    m.add_argument("subject")
    m.add_argument("--body", default="", help="the message text; '-' reads it from standard input")
    sub.add_parser("inbox")
    sub.add_parser("ack").add_argument("names", nargs="*")
    sub.add_parser("note").add_argument("text", nargs="+")
    fr = sub.add_parser("fresh")
    fr.add_argument("text", nargs="*", help="the handover: what the next conversation must know and the board does not say")
    fr.add_argument("--idle", action="store_true", help="for the launcher: ask only if this conversation is due and sits idle")
    sub.add_parser("save").add_argument("-m", "--message", default="edit")
    sub.add_parser("sync")
    sub.add_parser("publish")
    t = sub.add_parser("train")
    t.add_argument("name")
    t.add_argument("--data", required=True, help="dataset on the cloud volume to train on")
    t.add_argument("--bench", default=DEFAULT_BENCH, help="benchmark dataset to score on")
    t.add_argument("--minutes", type=float, default=0.0, help="GPU minutes to book for this run (default: the per-run limit)")
    t.add_argument("--horizon", type=int, default=16)
    t.add_argument("--note", default="", help="the hypothesis this run tests, in one line")
    g = sub.add_parser("generate")
    g.add_argument("name")
    g.add_argument("--note", default="", help="what this dataset is for, in one line")
    f = sub.add_parser("fetch")
    f.add_argument("name")
    f.add_argument("--data", action="store_true", help="fetch a dataset instead of a run")
    f.add_argument("--again", action="store_true", help="download it even if it is already here")
    sub.add_parser("runs").add_argument("-n", type=int, default=25)
    sub.add_parser("budget").add_argument("--new", action="store_true",
                                          help="for the human: start a new budget period, the total counts from zero again")
    s = sub.add_parser("setup")
    s.add_argument("--no-sync", action="store_true", help="do not install the Python environments")
    s.add_argument("--no-bench", action="store_true", help="do not generate the benchmark dataset")
    sub.add_parser("stop").add_argument("--keep-cloud", action="store_true", help="let cloud jobs that are running finish")
    sub.add_parser("go")
    sub.add_parser("snapshot").add_argument("name")
    sub.add_parser("restore").add_argument("name")
    return p


def human_only(role: str, what: str) -> None:
    if role != "human":
        raise LabError(f"`{what}` is for the human. If you think it is needed, say so:  uv run python -m lab msg human ...")


def run(argv: list[str]) -> int:
    rest = argv[argv.index("--") + 1 :] if "--" in argv else []
    a = build_parser().parse_args(argv[: argv.index("--")] if "--" in argv else argv)
    cwd = Path.cwd()
    lab, role = Lab.find(cwd), current_role(cwd)
    c = a.command
    if c == "setup":
        human_only(role, c)
        from .setup import setup

        setup(lab, sync=not a.no_sync, bench=not a.no_bench)
        return 0
    need_lab(lab)
    board = lab.board
    if c == "status":
        print(status_text(lab, role))
    elif c == "msg":
        body = sys.stdin.read() if a.body == "-" else a.body
        path = boards.send(board, role, a.to, a.subject, body)
        boards.commit(board, f"[{role}] to {a.to}: {a.subject}")
        print(f"left for {a.to}: {path}")
        if a.to in ROLES and role != "human":
            print(f"If the session lab-{a.to} is listed by your SendMessage tool, also send it one line saying a message is waiting.")
    elif c == "inbox":
        mine = boards.unread(board, role)
        print("\n".join(f"{p}   {boards.subject(p)}" for p in mine) or "no unread messages")
    elif c == "ack":
        moved = boards.acknowledge(board, role, a.names or None)
        boards.commit(board, f"[{role}] handled {len(moved)} message(s)")
        print(f"filed {len(moved)} message(s) under {board / 'inbox' / role / 'done'}")
    elif c == "note":
        boards.note(board, role, " ".join(a.text))
        boards.commit(board, f"[{role}] note")
    elif c == "fresh":
        if role not in ROLES:
            raise LabError("`fresh` replaces an agent's conversation and is run by that agent. To restart one yourself, close "
                           "its tab and run  lab\\launch.ps1 -Roles <role>")
        print((sessions.if_idle(lab, role) or "nothing to do") if a.idle else sessions.ask(lab, role, " ".join(a.text)))
    elif c == "save":
        print("recorded" if boards.commit(board, f"[{role}] {a.message}") else "nothing had changed")
    elif c == "sync":
        sync(cwd)
        print(f"your branch now contains the trunk ({git('rev-parse', '--short', TRUNK, cwd=cwd)})")
    elif c == "publish":
        print(publish(lab, cwd, role))
    elif c == "train":
        return train(lab, cwd, role, a, rest)
    elif c == "generate":
        return generate(lab, cwd, role, a, rest)
    elif c == "fetch":
        return fetch(lab, check_name(a.name), "data" if a.data else "runs", a.again)
    elif c == "runs":
        print(runs_text(lab, a.n))
    elif c == "budget":
        if a.new:
            human_only(role, "budget --new")
            used = Ledger(board).new_period(role)
            if used is None:
                print("Nothing has been spent in this budget period, so it goes on.")
            else:
                boards.commit(board, f"ledger: a new budget period (the last one used {used:.1f} GPU min)")
                print(f"A new budget period starts: the count is back at zero. The one that ends used {used:.1f} GPU min.")
        print(budget_text(lab))
    elif c == "stop":
        human_only(role, c)
        (board / "STOP").write_text(f"stopped by the human at {stamp()}\n", encoding="utf-8")
        print("STOP is on the board: every agent is refused at its next tool call, and no new cloud job can start.")
        if not a.keep_cloud:
            stopped = stop_cloud(lab)
            print(f"cloud: stopped {len(stopped)} running sandwm app(s)." if stopped else "cloud: no sandwm app was running.")
        print("Close the agents' terminal tabs when you like.  To allow a restart:  uv run python -m lab go")
    elif c == "go":
        human_only(role, c)
        (board / "STOP").unlink(missing_ok=True)
        print("STOP is lifted.  Start the agents again with  lab\\launch.ps1")
    elif c in ("snapshot", "restore"):
        human_only(role, c)
        from . import snapshots

        if c == "snapshot":
            print(snapshots.take(lab, check_name(a.name)))
        else:
            print("\n".join(snapshots.restore(lab, check_name(a.name))))
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(sys.argv[1:] if argv is None else argv)
    except LabError as err:
        print(f"lab: {err}", file=sys.stderr)
        return 1
