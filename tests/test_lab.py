"""The agent lab's tooling: budget, board, guard, sharing code.  Nothing here starts an agent or a cloud job."""

import json
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest

from lab import board as boards
from lab import cli, guard, jobs, scores, sessions, snapshots
from lab.budget import Ledger, Limits, Refused
from lab.paths import ROLES, TRUNK, Lab, LabError, git, locked, now, parse_time, stamp
from lab.setup import agent_settings, setup

LIMITS = "gpu_minutes_total = 60\ngpu_minutes_per_run = 20\nmax_concurrent_gpu_runs = 2\n"


@pytest.fixture
def board(tmp_path):
    (tmp_path / "board").mkdir()
    (tmp_path / "board" / "limits.toml").write_text(LIMITS)
    return tmp_path / "board"


# --- budget ---------------------------------------------------------------------------------------------
def test_limits_are_read_and_a_broken_file_refuses_everything(board):
    assert Ledger(board).limits == Limits(60.0, 20.0, 2, 20.0)
    (board / "limits.toml").write_text("gpu_minutes_total = 'lots'\n")
    with pytest.raises(Refused, match="no cloud job may start"):
        Ledger(board).reserve([{"run": "abc"}])
    (board / "limits.toml").unlink()
    with pytest.raises(Refused):
        Ledger(board).reserve([{"run": "abc"}])


def test_a_job_is_charged_its_allowance_until_it_reports_back(board):
    ledger = Ledger(board)
    (job,) = ledger.reserve([{"run": "r0001", "by": "train", "note": "baseline"}])
    assert ledger.spent() == 20 and [j["run"] for j in ledger.running()] == ["r0001"]
    ledger.settle(job, 6.25, "ok", gpu="A10G")
    assert ledger.spent() == 6.25 and ledger.running() == []
    record = ledger.jobs()[0]
    assert record["note"] == "baseline" and record["end"]["gpu"] == "A10G" and record["kind"] == "train"
    assert ledger.summary()["by"] == {"train": 6.25}


def test_the_budget_refuses_what_does_not_fit(board):
    ledger = Ledger(board)
    a, b = ledger.reserve([{"run": "a01"}, {"run": "b01"}])
    with pytest.raises(Refused, match="at most 2 may run at once"):
        ledger.reserve([{"run": "c01"}])
    ledger.settle(a, 20, "ok")
    ledger.settle(b, 20, "failed")
    assert ledger.spent() == 40
    with pytest.raises(Refused, match="GPU budget"):  # 40 spent + 2 x 20 > 60
        ledger.reserve([{"run": "c01"}, {"run": "d01"}])
    with pytest.raises(Refused, match="more than the 20"):
        ledger.reserve([{"run": "c01"}], minutes=30)
    (c,) = ledger.reserve([{"run": "c01"}], minutes=15)  # 55: fits
    with pytest.raises(Refused, match="GPU budget"):
        ledger.reserve([{"run": "e01"}], minutes=10)
    ledger.settle(c, 3, "ok")
    assert ledger.spent() == 43
    assert len(ledger.events()) == 6  # nothing that was refused left a line


def test_run_names_are_used_once_but_a_run_can_be_evaluated_again(board):
    ledger = Ledger(board)
    (job,) = ledger.reserve([{"run": "r0001"}], minutes=5)
    ledger.settle(job, 1, "ok")
    with pytest.raises(Refused, match="already used"):
        ledger.reserve([{"run": "r0001"}], minutes=5)
    with pytest.raises(Refused, match="must differ"):
        ledger.reserve([{"run": "x01"}, {"run": "x01"}], minutes=5)
    (again,) = ledger.reserve([{"run": "r0001"}], kind="evaluate", minutes=5)
    assert again == "r0001#2"


def test_nothing_starts_while_the_lab_is_stopped(board):
    (board / "STOP").write_text("stop")
    with pytest.raises(Refused, match="stopped the lab"):
        Ledger(board).reserve([{"run": "r0001"}])


def test_lock_is_exclusive_and_stale_locks_are_taken_over(tmp_path):
    path = tmp_path / "x.lock"
    with locked(path):
        with pytest.raises(LabError):
            with locked(path, wait=0.3):
                pass
    assert not path.exists()
    path.write_text("left behind by a dead process")
    with locked(path, wait=0.3, stale=0.0):
        pass


# --- board ------------------------------------------------------------------------------------------------
def test_messages_arrive_in_the_inbox_and_move_to_done(board):
    first = boards.send(board, "train", "eval", "Run r0007 is ready!", "It fixes G-001.\n")
    second = boards.send(board, "train", "eval", "Run r0007 is ready!")
    assert first != second and first.parent == board / "inbox" / "eval"
    assert "from: train" in first.read_text() and "It fixes G-001." in first.read_text()
    assert [boards.subject(p) for p in boards.unread(board, "eval")] == ["Run r0007 is ready!"] * 2
    assert boards.unread(board, "data") == []
    moved = boards.acknowledge(board, "eval", [first.name])
    assert [p.name for p in moved] == [first.name] and boards.unread(board, "eval") == [second]
    assert len(boards.acknowledge(board, "eval")) == 1 and boards.unread(board, "eval") == []


def test_journal_gaps_and_champion(board):
    boards.note(board, "data", "generated   thinwalls-v1\n and told train")
    boards.note(board, "data", "second note")
    assert boards.last_note(board, "data").endswith("second note") and boards.last_note(board, "eval") == ""
    assert "generated thinwalls-v1 and told train" in (board / "journal" / "data.md").read_text()
    (board / "gaps.md").write_text("# Gaps\n### G-001 | walls are lost | blocker | open\n- what\n"
                                   "### G-002 | sand multiplies | major | fixed\n### not a gap\n")
    assert boards.gaps(board) == [{"id": "G-001", "title": "walls are lost", "severity": "blocker", "status": "open"},
                                  {"id": "G-002", "title": "sand multiplies", "severity": "major", "status": "fixed"}]
    assert boards.champion(board) == ""
    (board / "champion.txt").write_text("r0007\n\nwhy: passes two more gates\n")
    assert boards.champion(board) == "r0007"


def test_gates_are_read_against_a_fetched_scorecard(board, tmp_path):
    run = tmp_path / "runs" / "r1"
    (run / "eval").mkdir(parents=True)
    (run / "eval" / "scorecard.json").write_text(json.dumps({"checks": {
        "walls/kept@48": {"value": 0.95, "better": "higher"}, "mass/sand@48": {"value": 0.3, "better": "lower"},
        "broken": {"value": None, "better": "higher"}}}))
    (board / "gates.toml").write_text('[gates]\n"walls/kept@48" = 0.9\n"mass/sand@48" = 0.1\n"broken" = 0.5\n"absent" = 1\n')
    card = scores.card(run)
    assert scores.failing(card, scores.gates(board)) == ["mass/sand@48", "broken", "absent"]
    assert scores.gate_line(run, board) == "gates 1/4"
    assert scores.gate_line(tmp_path / "nowhere", board) == "" and scores.card(tmp_path / "nowhere") is None


# --- guard --------------------------------------------------------------------------------------------------
def call(tool: str, home: Path, **tool_input) -> dict:
    return {"tool_name": tool, "tool_input": tool_input, "cwd": str(home / ".lab" / "train")}


def test_guard_fences_files(tmp_path):
    home = tmp_path
    lab = home / ".lab"

    def blocked(path) -> str | None:
        return guard.decide(call("Edit", home, file_path=str(path)), "train", str(home))

    assert blocked(lab / "train" / "sandwm" / "models.py") is None
    assert blocked("sandwm/models.py") is None  # relative to the agent's own worktree
    assert blocked(lab / "runs" / "r1" / "notes.md") is None
    assert blocked(lab / "datasets" / "probe-v1" / "notes.md") is None
    assert blocked(lab / "board" / "recipes.md") is None
    assert blocked(lab / "board" / "journal" / "train.md") is None
    assert blocked(tmp_path.parent / "scratch.txt") is None  # outside the repository: not the guard's business
    assert "human's" in blocked(lab / "board" / "limits.toml")
    assert "human's" in blocked(lab / "board" / "PROGRAM.md")
    assert "human's" in blocked(lab / "board" / "roles" / "train.md")
    assert "kept by the eval agent" in blocked(lab / "board" / "gaps.md")
    assert "another agent's journal" in blocked(lab / "board" / "journal" / "eval.md")
    assert "data agent's worktree" in blocked(lab / "data" / "sandwm" / "sim.py")
    assert "eval agent's worktree" in blocked(lab / "eval" / "sandwm" / "scorecard.py")
    assert "main checkout" in blocked(home / "sandwm" / "models.py")
    assert "settings" in blocked(lab / "train" / ".claude" / "settings.local.json")
    assert guard.decide(call("Write", home, file_path=str(lab / "board" / "gaps.md")), "eval", str(home)) is None
    assert guard.decide(call("Read", home, file_path=str(lab / "board" / "limits.toml")), "train", str(home)) is None


def test_guard_trips_on_commands_that_leave_the_lab(tmp_path):
    def blocked(command: str, tool: str = "Bash") -> str | None:
        return guard.decide(call(tool, tmp_path, command=command), "train", str(tmp_path))

    for fine in ("uv run python -m lab train r0001 --data basic64-10k -- --steps 100", "uv run pytest -q", "git status",
                 "git add sandwm/models.py && git commit -m 'spatial latent'", "git branch --show-current", "git branch -a",
                 "uv run modal app list", "uv run modal volume ls sandwm runs", "uv run python -m lab budget",
                 "git checkout lab-trunk -- sandwm/train.py", "git log main..lab-trunk --oneline"):
        assert blocked(fine) is None, fine
    assert "lab train" in blocked("uv run modal run cloud.py::train --name x")
    assert "lab train" in blocked("uv run modal run --detach cloud.py::sweep --spec 'a: '", "PowerShell")
    assert "limits and ledger" in blocked("sed -i 's/300/9000/' ../board/limits.toml")
    assert "limits and ledger" in blocked("Add-Content ..\\board\\ledger.jsonl '{}'", "PowerShell")
    assert "off limits" in blocked("cat ~/.modal.toml")
    assert "off limits" in blocked("uv run modal token new")
    assert "other Modal projects" in blocked("uv run modal app logs pwa-wsp-probe")
    assert "never deleted" in blocked("uv run modal volume rm -r sandwm runs/r0001")
    assert "pushes nowhere" in blocked("git push origin lab-trunk")
    assert "managed by" in blocked("git branch -D lab-eval")
    assert "managed by" in blocked("git update-ref refs/heads/lab-trunk HEAD")
    assert "own branch" in blocked("git checkout main")
    assert "own branch" in blocked("git switch lab-trunk && git merge lab-train")


def test_guard_stop_switch_blocks_everything_but_reading(tmp_path):
    (tmp_path / ".lab" / "board").mkdir(parents=True)
    (tmp_path / ".lab" / "board" / "STOP").write_text("stop")
    assert guard.decide(call("Bash", tmp_path, command="git status"), "eval", str(tmp_path)).startswith("STOP")
    assert guard.decide(call("ScheduleWakeup", tmp_path, delaySeconds=600), "eval", str(tmp_path)).startswith("STOP")
    assert guard.decide(call("Read", tmp_path, file_path="x"), "eval", str(tmp_path)) is None


def test_guard_as_claude_code_runs_it(tmp_path):
    """The hook is a script fed one JSON event on stdin: exit 2 and a reason on stderr block the call."""
    script = Path(guard.__file__)

    def run(event: dict) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(script), "--role", "data", "--home", str(tmp_path)],
                              input=json.dumps(event).encode(), capture_output=True)

    ok = run(call("Bash", tmp_path, command="uv run pytest"))
    assert ok.returncode == 0 and ok.stderr == b""
    seen = json.loads((tmp_path / ".lab" / "board" / "status" / "data.json").read_text())
    assert seen["tool"] == "Bash" and boards.last_seen(tmp_path / ".lab" / "board", "data") == seen
    no = run(call("Bash", tmp_path, command="git push"))
    assert no.returncode == 2 and b"pushes nowhere" in no.stderr
    assert subprocess.run([sys.executable, str(script), "--role", "data", "--home", str(tmp_path)], input=b"not json",
                          capture_output=True).returncode == 1  # a broken event never blocks the agent


# --- setup, and sharing code through the trunk -------------------------------------------------------------
@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A small repository that has the lab's tooling committed, set up as a lab (no environments, no benchmark)."""
    home = tmp_path / "project"
    (home / "lab").mkdir(parents=True)
    (home / "sandwm").mkdir()
    (home / "lab" / "guard.py").write_text("# stand-in\n")
    (home / "sandwm" / "models.py").write_text("WIDTH = 192\n")
    (home / "sandwm" / "sim.py").write_text("GRAVITY = 1\n")
    (home / "cloud.py").write_text("# cloud\n")
    (home / ".gitignore").write_text(".lab/\n.claude/settings.local.json\n/runs/\n")
    git("init", "-q", "-b", "main", cwd=home)
    for key, value in (("user.name", "Tester"), ("user.email", "tester@example.invalid"), ("core.autocrlf", "false")):
        git("config", key, value, cwd=home)
    git("add", "-A", cwd=home)
    git("commit", "-q", "-m", "start", cwd=home)
    lab = Lab.find(home)
    setup(lab, sync=False, bench=False, say=lambda *_: None)
    monkeypatch.delenv("LAB_ROLE", raising=False)
    monkeypatch.setattr(cli, "sh", lambda cmd, cwd=None: 0)  # "the tests pass", "the cloud job ran"
    return lab


def commit_in(tree: Path, name: str, text: str, message: str) -> None:
    (tree / name).write_text(text)
    git("add", name, cwd=tree)
    git("commit", "-q", "-m", message, cwd=tree)


def test_setup_builds_board_trunk_and_worktrees(repo):
    assert repo.home.name == "project" and repo.ready
    assert Lab.find(repo.worktree("eval")).home == repo.home, "a worktree finds the same lab"
    for name in ("PROGRAM.md", "limits.toml", "gates.toml", "gaps.md", "roles/eval.md", "roles/data.md", "roles/train.md",
                 ".gitignore", "inbox/human/.gitkeep"):
        assert (repo.board / name).exists(), name
    program = (repo.board / "PROGRAM.md").read_text(encoding="utf-8")
    assert "{{" not in program and str(repo.runs) in program
    assert git("branch", "--show-current", cwd=repo.board) == "lab-board" and git("status", "--porcelain", cwd=repo.board) == ""
    for role in ("eval", "data", "train"):
        tree = repo.worktree(role)
        assert git("branch", "--show-current", cwd=tree) == f"lab-{role}" and (tree / "sandwm" / "models.py").exists()
        settings = json.loads((tree / ".claude" / "settings.local.json").read_text())
        assert settings["env"]["LAB_ROLE"] == role
        hook = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        assert f"--role {role}" in hook and "guard.py" in hook and "\\" not in hook
        assert str(repo.board) in settings["permissions"]["additionalDirectories"]
        assert "Bash(uv run python -m lab *)" in settings["permissions"]["allow"]
        assert cli.current_role(tree) == role
    assert cli.current_role(repo.home) == "human"
    assert git("rev-parse", TRUNK, cwd=repo.home) == git("rev-parse", "main", cwd=repo.home)
    assert len(boards.gaps(repo.board)) >= 5 and boards.champion(repo.board) == "recon-k4"
    before = (repo.board / "gaps.md").read_text(encoding="utf-8")
    (repo.board / "gaps.md").write_text(before + "\nedited by eval\n", encoding="utf-8")
    setup(repo, sync=False, bench=False, say=lambda *_: None)  # repeating it changes nothing that exists
    assert (repo.board / "gaps.md").read_text(encoding="utf-8").endswith("edited by eval\n")


def test_agent_settings_quote_paths_with_spaces(tmp_path):
    hook = agent_settings(Lab(tmp_path / "my project"), "eval")["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert '/my project/lab/guard.py"' in hook and '--home "' in hook


def test_publish_moves_the_trunk_and_the_others_can_sync(repo):
    train, data = repo.worktree("train"), repo.worktree("data")
    commit_in(train, "sandwm/models.py", "WIDTH = 256\n", "[train] wider latent")
    out = cli.publish(repo, train, "train")
    assert "published" in out and git("rev-parse", TRUNK, cwd=train) == git("rev-parse", "HEAD", cwd=train)
    assert cli.publish(repo, train, "train").startswith("nothing to publish")
    commit_in(data, "sandwm/sim.py", "GRAVITY = 2\n", "[data] heavier")
    out = cli.publish(repo, data, "data")  # merges train's work first, then moves the trunk
    assert "published" in out and (data / "sandwm" / "models.py").read_text() == "WIDTH = 256\n"
    cli.sync(train)
    assert (train / "sandwm" / "sim.py").read_text() == "GRAVITY = 2\n"
    assert git("rev-parse", "main", cwd=repo.home) != git("rev-parse", TRUNK, cwd=repo.home), "main is never moved"
    assert (repo.home / "sandwm" / "models.py").read_text() == "WIDTH = 192\n", "the main checkout is never touched"


def test_publish_refuses_failing_tests_the_humans_files_and_dirty_trees(repo, monkeypatch):
    train = repo.worktree("train")
    commit_in(train, "sandwm/models.py", "WIDTH = 0\n", "[train] broken")
    monkeypatch.setattr(cli, "sh", lambda cmd, cwd=None: 1)
    trunk = git("rev-parse", TRUNK, cwd=train)
    with pytest.raises(LabError, match="tests fail"):
        cli.publish(repo, train, "train")
    assert git("rev-parse", TRUNK, cwd=train) == trunk
    monkeypatch.setattr(cli, "sh", lambda cmd, cwd=None: 0)
    commit_in(train, "cloud.py", "# no budget\n", "[train] edit cloud.py")
    with pytest.raises(LabError, match="are the human's"):
        cli.publish(repo, train, "train")
    assert "published" in cli.publish(repo, train, "human")
    (train / "sandwm" / "models.py").write_text("WIDTH = 1\n")
    with pytest.raises(LabError, match="uncommitted"):
        cli.publish(repo, train, "train")
    with pytest.raises(LabError, match="agent's branch"):
        cli.publish(repo, repo.home, "human")


def test_publish_reports_a_conflict_and_changes_in_anothers_area(repo):
    train, eval_ = repo.worktree("train"), repo.worktree("eval")
    commit_in(train, "sandwm/sim.py", "GRAVITY = 3\n", "[train] touch the simulator")
    assert "in the data agent's area" in cli.publish(repo, train, "train")
    commit_in(eval_, "sandwm/sim.py", "GRAVITY = 4\n", "[eval] also touch it")
    with pytest.raises(LabError, match="does not merge cleanly"):
        cli.publish(repo, eval_, "eval")


def test_cloud_jobs_need_published_committed_code(repo, monkeypatch):
    train = repo.worktree("train")
    args = cli.build_parser().parse_args(["train", "r0001-test", "--data", "basic64"])
    commit_in(train, "sandwm/models.py", "WIDTH = 64\n", "[train] narrow")
    with pytest.raises(LabError, match="not on the trunk"):
        cli.train(repo, train, "train", args, ["--steps", "10"])
    cli.publish(repo, train, "train")
    (train / "sandwm" / "models.py").write_text("WIDTH = 65\n")
    with pytest.raises(LabError, match="not committed"):
        cli.train(repo, train, "train", args, ["--steps", "10"])
    git("checkout", "--", "sandwm/models.py", cwd=train)
    commands = []
    monkeypatch.setattr(cli, "sh", lambda cmd, cwd=None: commands.append(cmd) or 0)
    assert cli.train(repo, train, "train", args, ["--steps", "10", "--model", "embed_dim=64"]) == 1  # the stand-in ran nothing
    assert commands[0][:5] == ["uv", "run", "modal", "run", "cloud.py::train"]
    assert "--args=--steps 10 --model embed_dim=64" in commands[0] and "bench-basic-v1" in commands[0]
    with pytest.raises(LabError, match="names are"):
        cli.train(repo, train, "train", cli.build_parser().parse_args(["train", "Bad Name", "--data", "x"]), [])
    bench = cli.build_parser().parse_args(["generate", "bench-oil-v1"])
    with pytest.raises(LabError, match="benchmarks are named"):
        cli.generate(repo, train, "data", bench, ["--seed", "5"])
    with pytest.raises(LabError, match="benchmarks are named"):
        cli.generate(repo, train, "data", cli.build_parser().parse_args(["generate", "basic-thin-v1"]), ["--seed", "9002"])
    assert cli.generate(repo, train, "data", bench, ["--episodes", "0", "--seed", "9002"]) == 0


def test_fetch_merges_into_what_is_already_there(repo, monkeypatch):
    def fake_download(cmd, cwd=None):  # `modal volume get sandwm runs/r1 <folder>` puts the run in <folder>/r1
        assert cmd[:5] == ["uv", "run", "modal", "volume", "get"] and cmd[6] == "runs/r1"
        (Path(cmd[-1]) / "r1" / "eval").mkdir(parents=True)
        (Path(cmd[-1]) / "r1" / "ckpt.pt").write_text("weights")
        (Path(cmd[-1]) / "r1" / "eval" / "metrics.json").write_text("from the volume")
        return 0

    monkeypatch.setattr(cli, "sh", fake_download)
    assert cli.fetch(repo, "r1") == 0
    run = repo.runs / "r1"
    assert (run / "ckpt.pt").read_text() == "weights"
    (run / "eval" / "scorecard.json").write_text("scored on the laptop")
    (run / "eval" / "metrics.json").write_text("edited here")
    monkeypatch.setattr(cli, "sh", lambda cmd, cwd=None: pytest.fail("must not download what is already here"))
    assert cli.fetch(repo, "r1") == 0
    monkeypatch.setattr(cli, "sh", fake_download)
    assert cli.fetch(repo, "r1", again=True) == 0
    assert (run / "eval" / "scorecard.json").read_text() == "scored on the laptop", "local-only files survive"
    assert (run / "eval" / "metrics.json").read_text() == "from the volume"
    assert not list(repo.state.glob("fetch-*")), "the temporary download folder is removed"
    monkeypatch.setattr(cli, "sh", lambda cmd, cwd=None: 1)
    assert cli.fetch(repo, "missing") == 1 and not (repo.runs / "missing").exists()


def test_booking_through_the_lab_keeps_the_ledger_and_tags_the_code(repo):
    train = repo.worktree("train")
    assert jobs.find(train).home == repo.home and jobs.limits(train).gpu_minutes_per_run == 20
    booking = jobs.book([{"run": "r0001", "data": "basic64", "note": "baseline"}, {"run": "r0002", "data": "basic64"}],
                        "train", 5, cwd=train)
    assert booking.minutes == 5 and git("rev-parse", "run/r0001", cwd=repo.home) == git("rev-parse", "HEAD", cwd=train)
    ledger = Ledger(repo.board)
    assert ledger.spent() == 10 and ledger.jobs()[0]["by"] == "train" and ledger.jobs()[0]["note"] == "baseline"
    booking.done("r0001", {"gpu min": 1.5, "gpu": "A10G", "steps": 100})
    booking.done("r0002", RuntimeError("out of memory"))
    booking.close()
    r1, r2 = ledger.jobs()
    assert r1["end"]["status"] == "ok" and r1["charged"] == 1.5
    assert r2["end"]["status"] == "failed" and r2["charged"] < 1 and "out of memory" in r2["end"]["error"]
    assert git("status", "--porcelain", cwd=repo.board) == "", "the ledger is recorded in the board's history"
    (repo.board / "STOP").write_text("stop")
    with pytest.raises(Refused):
        jobs.book([{"run": "r0003"}], cwd=train)
    with pytest.raises(Refused):
        jobs.record("generate", cwd=train, dataset="x")


def test_snapshot_and_restore(repo, monkeypatch, capsys):
    train, data = repo.worktree("train"), repo.worktree("data")
    start, main = git("rev-parse", TRUNK, cwd=repo.home), git("rev-parse", "main", cwd=repo.home)
    monkeypatch.chdir(repo.home)
    assert cli.main(["snapshot", "pre"]) == 0 and "snapshot pre: code at" in capsys.readouterr().out
    with pytest.raises(LabError, match="already exists"):
        snapshots.take(repo, "pre")

    # the lab gets messy: published code, uncommitted and untracked work, a rewritten gap list, messages, spending
    commit_in(train, "sandwm/models.py", "WIDTH = 7\n", "[train] an idea")
    cli.publish(repo, train, "train")
    (data / "sandwm" / "sim.py").write_text("GRAVITY = -1\n")
    (data / "notes.txt").write_text("half an idea")
    (data / "runs").mkdir()
    (data / "runs" / "scratch.bin").write_text("ignored scratch")
    boards.send(repo.board, "train", "eval", "run ready")
    boards.note(repo.board, "train", "did things")
    (repo.board / "gaps.md").write_text("# everything is fine\n", encoding="utf-8")
    (job,) = Ledger(repo.board).reserve([{"run": "r0001"}], minutes=5)
    Ledger(repo.board).settle(job, 3, "ok")
    (repo.board / "limits.toml").write_text(LIMITS)  # the human also changed the budget meanwhile

    with pytest.raises(LabError, match="no snapshot called nope"):
        snapshots.restore(repo, "nope")
    with pytest.raises(LabError, match="stop the lab first"):
        snapshots.restore(repo, "pre")
    (repo.board / "STOP").write_text("stop")
    assert cli.main(["restore", "pre"]) == 0
    out = capsys.readouterr().out
    assert "kept the state you are leaving" in out and "the ledger is kept: 3.0 of 300" in out

    assert git("rev-parse", TRUNK, cwd=repo.home) == start and git("rev-parse", "main", cwd=repo.home) == main
    for role in ROLES:
        tree = repo.worktree(role)
        assert git("rev-parse", "HEAD", cwd=tree) == start and git("status", "--porcelain", cwd=tree) == ""
        assert (tree / "sandwm" / "models.py").read_text() == "WIDTH = 192\n" and cli.current_role(tree) == role
        assert json.loads((tree / ".claude" / "settings.local.json").read_text())["env"]["LAB_ROLE"] == role
    assert not (data / "notes.txt").exists() and (data / "sandwm" / "sim.py").read_text() == "GRAVITY = 1\n"
    assert (data / "runs" / "scratch.bin").exists(), "ignored scratch files are left alone"

    assert boards.unread(repo.board, "eval") == [] and boards.last_note(repo.board, "train") == ""
    assert len(boards.gaps(repo.board)) >= 5 and Ledger(repo.board).limits.gpu_minutes_total == 300
    assert Ledger(repo.board).spent() == 3, "money already spent stays spent"
    assert (repo.board / "STOP").exists() and git("status", "--porcelain", cwd=repo.board) == ""
    assert "restored snapshot pre" in git("log", "-1", "--format=%s", cwd=repo.board)

    kept = git("tag", "--list", "kept/*", cwd=repo.home).split()
    assert sorted(t.rsplit("/", 1)[1] for t in kept) == ["board", "data", "eval", "research", "train", "trunk"]
    tag = {t.rsplit("/", 1)[1]: t for t in kept}
    assert git("show", f"{tag['data']}:notes.txt", cwd=repo.home) == "half an idea"
    assert "GRAVITY = -1" in git("show", f"{tag['data']}:sandwm/sim.py", cwd=repo.home)
    assert "WIDTH = 7" in git("show", f"{tag['trunk']}:sandwm/models.py", cwd=repo.home)
    assert "everything is fine" in git("show", f"{tag['board']}:gaps.md", cwd=repo.home)
    assert "run ready" in git("ls-tree", "-r", "--name-only", tag["board"], cwd=repo.home).replace("-", " ")

    monkeypatch.setenv("LAB_ROLE", "train")
    assert cli.main(["restore", "pre"]) == 1 and "for the human" in capsys.readouterr().err


def test_no_lab_means_no_bookkeeping(tmp_path):
    assert jobs.find(tmp_path) is None and jobs.limits(tmp_path) is None and jobs.book([{"run": "abc"}], cwd=tmp_path) is None
    jobs.record("generate", cwd=tmp_path, dataset="x")


def test_command_line_end_to_end(repo, monkeypatch, capsys):
    train = repo.worktree("train")
    monkeypatch.chdir(train)
    assert cli.main(["msg", "eval", "r0001 is ready", "--body", "passes walls/kept@48"]) == 0
    assert cli.main(["note", "launched", "r0001"]) == 0
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "you are: train" in out and "0.0 of 300 GPU min" in out and "champion  recon-k4" in out
    assert f"trunk     {TRUNK} @ {git('rev-parse', '--short', TRUNK, cwd=train)} start" in out
    assert "last note:" in out and "launched r0001" in out and "blockers: G-001" in out
    monkeypatch.setenv("LAB_ROLE", "eval")
    assert cli.main(["inbox"]) == 0 and "r0001 is ready" in capsys.readouterr().out
    assert cli.main(["ack"]) == 0 and cli.main(["inbox"]) == 0
    assert "no unread messages" in capsys.readouterr().out
    assert cli.main(["stop"]) == 1 and "for the human" in capsys.readouterr().err
    assert cli.main(["budget"]) == 0 and "300.0 left" in capsys.readouterr().out
    assert cli.main(["runs"]) == 0
    monkeypatch.setenv("LAB_ROLE", "human")
    monkeypatch.setattr(cli, "stop_cloud", lambda lab: [])
    assert cli.main(["stop"]) == 0 and (repo.board / "STOP").exists()
    assert "STOPPED" in cli.status_text(repo, "human")
    assert cli.main(["go"]) == 0 and not (repo.board / "STOP").exists()
    assert git("status", "--porcelain", cwd=repo.board) == ""


# --- the research agent and tracks ---------------------------------------------------------------------------
def test_tracks_have_a_budget_of_their_own(board):
    (board / "limits.toml").write_text(LIMITS + "gpu_minutes_tracks = 12\n")
    ledger = Ledger(board)
    assert ledger.limits.tracks_cap == 12 and Limits(90.0, 20.0, 2).tracks_cap == 30, "a third of the total unless set"
    (a,) = ledger.reserve([{"run": "t001-rl-task"}], minutes=5)
    (b,) = ledger.reserve([{"run": "r0001-main"}], minutes=5)
    with pytest.raises(Refused, match="track budget"):  # 5 booked + 10 more > 12
        ledger.reserve([{"run": "t001-rl-policy"}], minutes=10)
    ledger.settle(a, 2, "ok")
    ledger.settle(b, 5, "ok")
    (c,) = ledger.reserve([{"run": "t002-diffusion"}], minutes=10)  # 2 used + 10 = 12: fits exactly
    summary = ledger.summary()
    assert summary["tracks"] == {"T-001": 2.0, "T-002": 10.0} and summary["tracks_cap"] == 12 and summary["spent"] == 17
    with pytest.raises(Refused, match="track budget"):
        ledger.reserve([{"run": "t001-more"}], minutes=1)
    ledger.reserve([{"run": "r0002-main"}], minutes=20)  # the main line is not held back by the tracks' cap
    ledger.settle(c, 10, "ok")


def test_tracks_draw_on_the_same_total_as_everything_else(board):
    (board / "limits.toml").write_text(LIMITS + "gpu_minutes_tracks = 30\n")  # the total is 60
    ledger = Ledger(board)
    for name, minutes in (("r0001-main", 20), ("r0002-main", 20), ("r0003-main", 15)):
        (job,) = ledger.reserve([{"run": name}], minutes=minutes)
        ledger.settle(job, minutes, "ok")
    with pytest.raises(Refused, match="GPU budget"):  # 5 left of the total, though tracks have used none of their 30
        ledger.reserve([{"run": "t001-first"}], minutes=10)
    ledger.reserve([{"run": "t001-first"}], minutes=5)
    assert ledger.summary()["tracks_cap"] == 30 and ledger.summary()["spent"] == 60


def test_track_list_is_read_from_the_board(board):
    assert boards.tracks(board) == []
    (board / "tracks.md").write_text("# Tracks\n    ### T-009 | an example in the header | active\n"
                                     "### T-002 | a diffusion predictor | parked\n- idea\n### T-001 | an agent that acts | active\n")
    assert boards.tracks(board) == [{"id": "T-002", "title": "a diffusion predictor", "status": "parked"},
                                    {"id": "T-001", "title": "an agent that acts", "status": "active"}]


def test_guard_knows_the_research_agent(tmp_path):
    lab = tmp_path / ".lab"

    def blocked(role: str, path) -> str | None:
        return guard.decide(call("Edit", tmp_path, file_path=str(path)), role, str(tmp_path))

    assert blocked("research", lab / "board" / "tracks.md") is None
    assert blocked("research", lab / "board" / "tracks" / "T-001-rl.md") is None
    assert blocked("research", lab / "board" / "journal" / "research.md") is None
    assert blocked("research", lab / "research" / "notes.md") is None
    assert "kept by the research agent" in blocked("train", lab / "board" / "tracks.md")
    assert "kept by the research agent" in blocked("eval", lab / "board" / "tracks" / "T-001-rl.md")
    assert "research agent's worktree" in blocked("data", lab / "research" / "notes.md")
    assert "kept by the eval agent" in blocked("research", lab / "board" / "gaps.md")
    assert "kept by the train agent" in blocked("research", lab / "board" / "recipes.md")
    assert "train agent's worktree" in blocked("research", lab / "train" / "sandwm" / "models.py")
    assert "human's" in blocked("research", lab / "board" / "limits.toml")


def test_research_is_set_up_and_reachable(repo, monkeypatch, capsys):
    tree = repo.worktree("research")
    assert git("branch", "--show-current", cwd=tree) == "lab-research" and cli.current_role(tree) == "research"
    assert json.loads((tree / ".claude" / "settings.local.json").read_text())["env"]["LAB_ROLE"] == "research"
    for name in ("roles/research.md", "tracks.md", "tracks/.gitkeep", "inbox/research/.gitkeep"):
        assert (repo.board / name).exists(), name
    program = (repo.board / "PROGRAM.md").read_text(encoding="utf-8")
    assert "## Tracks" in program and "lab msg <eval|data|train|research|human>" in program
    assert Ledger(repo.board).limits.tracks_cap == 100
    monkeypatch.chdir(repo.worktree("train"))
    assert cli.main(["msg", "research", "idea: an agent that learns to act in the world", "--body", "with RL"]) == 0
    assert "lab-research" in capsys.readouterr().out
    assert [boards.subject(p) for p in boards.unread(repo.board, "research")] == ["idea: an agent that learns to act in the world"]
    (repo.board / "tracks.md").write_text("### T-001 | an agent that acts | active\n", encoding="utf-8")
    (job,) = Ledger(repo.board).reserve([{"run": "t001-task", "by": "train"}], minutes=5)
    Ledger(repo.board).settle(job, 4, "ok")
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "research  has not run yet" in out and "active: T-001 an agent that acts; GPU min used 4.0 of 100" in out
    assert cli.main(["budget"]) == 0 and "research tracks: 4.0 of the 100 min" in capsys.readouterr().out
    settings = tree / ".claude" / "settings.local.json"
    before = settings.stat().st_mtime_ns
    setup(repo, sync=False, bench=False, say=lambda *_: None)
    assert settings.stat().st_mtime_ns == before, "an unchanged settings file is not rewritten under a running agent"


# --- coverage: what the evaluation does not see -----------------------------------------------------------------
def test_coverage_list_is_read_shown_and_kept_by_eval(repo, monkeypatch, capsys):
    listed = boards.coverage(repo.board)  # the template seeds the human's examples
    assert [c["id"] for c in listed] == ["C-001", "C-002", "C-003", "C-004"]
    assert listed[0] == {"id": "C-001", "title": "very long rollouts: hundreds to thousands of frames", "priority": "high",
                         "status": "open"}
    text = (repo.board / "coverage.md").read_text(encoding="utf-8")
    (repo.board / "coverage.md").write_text(text.replace("frames | high | open", "frames | high | covered"), encoding="utf-8")
    monkeypatch.chdir(repo.home)
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "coverage  3 situation(s) the evaluation does not cover yet, of 4 listed; high: C-002 intermittent actions" in out
    assert "## What the evaluation does not see" in (repo.board / "PROGRAM.md").read_text(encoding="utf-8")
    assert "## What the evaluation does not see" in (repo.board / "roles" / "eval.md").read_text(encoding="utf-8")

    def blocked(role: str) -> str | None:
        return guard.decide(call("Edit", repo.home, file_path=str(repo.board / "coverage.md")), role, str(repo.home))

    assert blocked("eval") is None and "kept by the eval agent" in blocked("train") and "kept by the eval agent" in blocked("data")


# --- fresh starts: a conversation is replaced by an empty one from time to time -----------------------------------
def seen_since(board: Path, role: str, hours: float, idle: float = 0.0, calls: int = 80, launch: str = "L1") -> None:
    """The guard's record of `role`'s conversation: it began `hours` ago and made its last tool call `idle` minutes ago."""
    t = now()
    (board / "status").mkdir(exist_ok=True)
    (board / "status" / f"{role}.json").write_text(json.dumps({
        "t": stamp(t - timedelta(minutes=idle)), "tool": "Bash", "session": "s1", "since": stamp(t - timedelta(hours=hours)),
        "calls": calls, "launch": launch}))


def test_guard_keeps_the_clock_of_a_conversation(tmp_path, monkeypatch):
    monkeypatch.delenv("LAB_SESSION", raising=False)
    home, board = str(tmp_path), tmp_path / ".lab" / "board"
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("{}")
    event = {"tool_name": "Bash", "session_id": "s1", "transcript_path": str(transcript)}
    guard.beat("eval", home, event)
    guard.beat("eval", home, {**event, "tool_name": "Read"})
    first = boards.last_seen(board, "eval")
    assert (first["calls"], first["tool"], first["session"], first["launch"]) == (2, "Read", "s1", "")
    assert abs((now() - parse_time(first["since"])).total_seconds()) < 120, "it began when its transcript was created"
    monkeypatch.setenv("LAB_SESSION", "L1")  # the same conversation, now run by a launcher that can replace it
    guard.beat("eval", home, event)
    again = boards.last_seen(board, "eval")
    assert (again["calls"], again["since"], again["launch"]) == (3, first["since"], "L1")
    guard.beat("eval", home, {"tool_name": "Glob", "session_id": "s2"})  # another conversation: the clock starts again
    assert boards.last_seen(board, "eval")["calls"] == 1 and boards.last_seen(board, "eval")["session"] == "s2"
    (board / "status" / "eval.json").write_text('{"t": "20')  # a file cut short does not stop the heartbeat
    guard.beat("eval", home, event)
    assert boards.last_seen(board, "eval")["calls"] == 1
    assert sorted(p.name for p in (board / "status").iterdir()) == ["eval.json"], "nothing is left beside the record"


def test_a_conversation_is_due_after_its_hours_and_some_work(board, monkeypatch):
    monkeypatch.delenv("LAB_SESSION", raising=False)
    nobody = sessions.read(board, "eval")
    assert nobody.since is None and not nobody.due and sessions.describe(nobody) == ""
    seen_since(board, "eval", hours=5)
    young = sessions.read(board, "eval")
    assert young.limit == 6 and not young.due and sessions.describe(young) == "conversation 5.0 h old, fresh start in 1.0 h"
    seen_since(board, "eval", hours=6.5)
    old = sessions.read(board, "eval")
    assert old.due and not old.mine and sessions.describe(old) == "conversation 6.5 h old, fresh start due"
    monkeypatch.setenv("LAB_SESSION", "L1")
    assert sessions.read(board, "eval").mine
    seen_since(board, "eval", hours=6.5, calls=10)
    assert not sessions.read(board, "eval").due and "once it has done more work" in sessions.describe(sessions.read(board, "eval"))
    seen_since(board, "train", hours=6.5)  # data and train come a little later, so that the three do not start afresh together
    assert sessions.read(board, "train").limit == pytest.approx(7.2) and not sessions.read(board, "train").due
    seen_since(board, "train", hours=7.3)
    assert sessions.read(board, "train").due
    seen_since(board, "eval", hours=6.5, launch="")
    stuck = sessions.read(board, "eval")
    assert stuck.due and not stuck.mine and "old launcher" in sessions.describe(stuck)
    (board / "limits.toml").write_text(LIMITS + "session_hours = 2\n")
    seen_since(board, "eval", hours=2.5)
    assert Ledger(board).limits.session_hours == 2 and sessions.read(board, "eval").due
    (board / "limits.toml").write_text(LIMITS + "session_hours = 0\n")
    seen_since(board, "eval", hours=99)
    assert not sessions.read(board, "eval").due and sessions.describe(sessions.read(board, "eval")) == "conversation 99.0 h old"


def test_an_agent_asks_for_its_fresh_start_and_leaves_a_handover(repo, monkeypatch, capsys):
    train, request = repo.worktree("train"), sessions.request_file(repo.board, "train")
    monkeypatch.chdir(train)
    monkeypatch.setenv("LAB_ROLE", "train")
    monkeypatch.delenv("LAB_SESSION", raising=False)
    seen_since(repo.board, "train", hours=8)
    assert cli.main(["fresh", "nothing open"]) == 1 and "cannot be replaced from here" in capsys.readouterr().err
    jobs.book([{"run": "r0001"}], minutes=5, cwd=train).close()  # no launcher to replace it: nothing is held back
    monkeypatch.setenv("LAB_SESSION", "L1")
    assert cli.main(["fresh"]) == 1 and "must know" in capsys.readouterr().err
    seen_since(repo.board, "train", hours=0.2)
    assert cli.main(["fresh", "too", "early"]) == 1 and "only 12 minutes old" in capsys.readouterr().err
    seen_since(repo.board, "train", hours=8)
    (job,) = Ledger(repo.board).reserve([{"run": "r0002", "by": "train"}], minutes=5)
    assert cli.main(["fresh", "a job is running"]) == 1 and "still running: r0002" in capsys.readouterr().err
    Ledger(repo.board).settle(job, 1, "ok")
    (train / "sandwm" / "models.py").write_text("WIDTH = 256\n")
    assert cli.main(["fresh", "work is uncommitted"]) == 1 and "commit what is in your worktree" in capsys.readouterr().err
    git("checkout", "--", "sandwm/models.py", cwd=train)
    assert not request.exists(), "nothing that was refused asked the launcher for anything"

    assert cli.main(["status"]) == 0  # a due conversation is told so, and starts no new cloud job
    out = capsys.readouterr().out
    assert "train     last tool call just now; conversation 8.0 h old, fresh start due; branch" in out and "FRESH START DUE" in out
    with pytest.raises(Refused, match="due for a fresh start"):
        jobs.book([{"run": "r0003"}], cwd=train)
    with pytest.raises(Refused, match="due for a fresh start"):
        jobs.record("generate", cwd=train, dataset="x")

    assert cli.main(["fresh", "half done: the spatial latent in sandwm/models.py;", "eval waits for a verdict on r0002"]) == 0
    assert "Call no other tool" in capsys.readouterr().out and json.loads(request.read_text())["why"] == "asked"
    note = boards.last_note(repo.board, "train")
    assert "HANDOVER before a fresh start (conversation 8.0 h old, 80 tool calls): half done: the spatial latent" in note
    assert git("status", "--porcelain", cwd=repo.board) == "", "the handover is in the board's history"
    monkeypatch.setenv("LAB_ROLE", "human")
    monkeypatch.chdir(repo.home)
    assert cli.main(["fresh", "x"]) == 1 and "run by that agent" in capsys.readouterr().err
    assert cli.main(["status"]) == 0 and "FRESH START DUE" not in capsys.readouterr().out


def test_the_launcher_replaces_a_due_conversation_that_sits_idle(repo, monkeypatch, capsys):
    monkeypatch.chdir(repo.home)  # the launcher runs `python -m lab fresh --idle` from the main checkout, as the agent
    monkeypatch.setenv("LAB_ROLE", "data")
    monkeypatch.setenv("LAB_SESSION", "L7")
    request = sessions.request_file(repo.board, "data")

    def asked(**record) -> bool:
        seen_since(repo.board, "data", **{"hours": 7, "idle": 45, "launch": "L7", **record})
        assert cli.main(["fresh", "--idle"]) == 0
        capsys.readouterr()
        return request.exists()

    assert not asked(launch="L6"), "the record is of an earlier conversation: the new one has made no tool call yet"
    assert not asked(idle=5), "it is in the middle of something"
    assert not asked(hours=3) and not asked(calls=10)
    (job,) = Ledger(repo.board).reserve([{"run": "r0001", "by": "data"}], minutes=5)
    assert not asked(), "its cloud job would die with it"
    Ledger(repo.board).settle(job, 1, "ok")
    (repo.board / "STOP").write_text("stop")
    assert not asked()
    (repo.board / "STOP").unlink()
    assert asked() and json.loads(request.read_text())["why"] == "idle"
    assert "the launcher ended the conversation" in boards.last_note(repo.board, "data")
    notes = (repo.board / "journal" / "data.md").read_text()
    assert asked() and (repo.board / "journal" / "data.md").read_text() == notes, "one request is enough"


def test_the_launcher_passes_the_compaction_limit_and_resumes_by_id(repo):
    """lab/agent.ps1 -DryRun prints the claude command it would run; nothing is started."""
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("the launcher is a Windows PowerShell script")
    script = repo.home / "lab" / "agent.ps1"
    shutil.copy(Path(cli.__file__).parent / "agent.ps1", script)

    def dry_run(*more: str) -> str:
        done = subprocess.run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-Role", "eval",
                               "-DryRun", *more], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        return done.stdout

    limits = repo.board / "limits.toml"
    out = dry_run()
    assert "--name lab-eval" in out and "--autocompact 500000" in out, "the template's compact_tokens"
    assert "--resume" not in out and "--continue" not in out
    limits.write_text(limits.read_text().replace("compact_tokens = 500000", "compact_tokens = 250_000"))
    assert "--autocompact 250000" in dry_run()
    limits.write_text(limits.read_text().replace("compact_tokens = 250_000", "compact_tokens = 0"))
    assert "--autocompact" not in dry_run(), "0 leaves Claude Code's own default"
    assert "--continue" in dry_run("-Continue"), "no conversation on record yet"
    (repo.board / "status").mkdir(exist_ok=True)
    (repo.board / "status" / "eval.json").write_text(json.dumps({"session": "0a1b2c3d-1111-2222-3333-444455556666"}))
    assert "--resume 0a1b2c3d-1111-2222-3333-444455556666" in dry_run("-Continue"), "--continue skips /loop conversations"
