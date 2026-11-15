"""The Modal app must at least import: it is only exercised for real in the cloud."""

from pathlib import Path

import pytest

pytest.importorskip("modal")


def test_cloud_app_is_well_formed():
    import cloud

    assert cloud.app.name == "sandwm"
    for name in ("generate", "train", "sweep", "reevaluate", "generate_job", "train_job", "evaluate_job"):
        assert hasattr(cloud, name), f"cloud.py lost its {name!r} entry point"
    assert isinstance(cloud.GPU, list) and cloud.GPU
    assert cloud.lab_jobs() is not None, "the lab's bookkeeping must be importable on this machine"
    assert 0 < cloud.RUN_SECONDS <= 8 * 3600 and cloud.MAX_GPUS >= 1


def test_log_lines_are_tagged_with_the_run_name(capsys):
    import cloud

    with cloud.tagged("run-a"):
        print("one")
        print("two\nthree", end="")
        print(" four")
    assert capsys.readouterr().out == "[run-a] one\n[run-a] two\n[run-a] three four\n"


class FakeCall:
    """Stands in for a Modal function call: `get` gives the prepared result, or raises it if it is an exception."""

    def __init__(self, result):
        self.result = result

    def get(self, timeout=None):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeJob:
    """Stands in for a Modal function: `spawn` hands back a call with a prepared result instead of running anything."""

    def __init__(self, results):
        self.results, self.calls = results, []

    def spawn(self, *call):
        self.calls.append(call)
        return FakeCall(self.results[len(self.calls) - 1])


def test_launch_books_and_settles_every_job(monkeypatch, capsys):
    import cloud

    class Booking:
        minutes, settled, closed = 5.0, [], False

        def poll(self, every):  # the lab has a GPU slot for both at once
            news, Booking.news = getattr(Booking, "news", {"a": "go", "b": "go"}), {}
            return news

        def waits(self):
            return ""

        def done(self, name, result):
            self.settled.append((name, result))

        def close(self):
            Booking.closed = True

    class Jobs:
        Refused = RuntimeError
        asked = None

        @staticmethod
        def book(record, kind, minutes):
            Jobs.asked = (record, kind, minutes)
            return Booking()

    summary = {"steps": 10, "gpu": "A10G", "gpu min": 1.2}
    monkeypatch.setattr(cloud, "lab_jobs", lambda: Jobs)
    job = FakeJob([summary, ValueError("boom")])
    with pytest.raises(SystemExit):  # one of the two failed
        cloud.launch(job, ["a", "b"], [("a",), ("b",)], "train", 5.0, [{"run": "a"}, {"run": "b"}])
    assert Jobs.asked == ([{"run": "a"}, {"run": "b"}], "train", 5.0) and job.calls == [("a",), ("b",)]
    assert [name for name, _ in Booking.settled] == ["a", "b"] and Booking.closed
    out = capsys.readouterr().out
    assert "5 GPU min booked for each" in out and "b FAILED" in out and "A10G" in out


def test_launch_does_not_start_what_the_budget_refuses(monkeypatch):
    import cloud

    class Jobs:
        class Refused(Exception):
            pass

        @staticmethod
        def book(record, kind, minutes):
            raise Jobs.Refused("GPU budget: nothing left")

    monkeypatch.setattr(cloud, "lab_jobs", lambda: Jobs)
    job = FakeJob([])
    with pytest.raises(SystemExit, match="not started: GPU budget"):
        cloud.launch(job, ["a"], [("a",)], "train", 0.0, [{"run": "a"}])
    assert job.calls == [], "nothing may reach the cloud after a refusal"


@pytest.mark.filterwarnings("ignore:The .* function is executing locally")
def test_score_job_finds_the_benchmarks_on_the_volume_and_reports_what_it_wrote(tmp_path, monkeypatch):
    import torch

    import cloud
    from sandwm import scorecard

    class Volume:
        commits = 0

        def commit(self):
            Volume.commits += 1

    seen = []

    def fake_scorecard(argv):  # stands in for sandwm.scorecard.main: writes the card it is asked for
        seen.append(argv)
        card = Path(argv[argv.index("--run") + 1]) / "eval" / (argv[argv.index("--card") + 1] + ".json")
        card.parent.mkdir(parents=True, exist_ok=True)
        card.write_text("{}")

    monkeypatch.setattr(cloud, "VOL", tmp_path)
    monkeypatch.setattr(cloud, "volume", Volume())
    monkeypatch.setattr(scorecard, "main", fake_scorecard)
    for bench in ("bench-a", "bench-b"):
        (tmp_path / "data" / bench).mkdir(parents=True)
        (tmp_path / "data" / bench / "meta.json").write_text("{}")
    kept = tmp_path / "runs" / "r1" / "eval" / "scorecard.json"  # the run's own card: not touched, so not reported
    kept.parent.mkdir(parents=True)
    kept.write_text("{}")
    threads = torch.get_num_threads()
    try:
        written = cloud.score_job.local("r1", "bench-a", ["bench-b=walls,mass"], "probes", "--no-pictures --twin 1")
        assert written == ["runs/r1/eval/probes.json"] and Volume.commits == 1
        assert seen == [["--run", str(tmp_path / "runs" / "r1"), "--bench", str(tmp_path / "data" / "bench-a"), "--also",
                         f"{tmp_path / 'data' / 'bench-b'}=walls,mass", "--card", "probes", "--no-pictures", "--twin", "1"]]
        # a "run" without a model, for the world scored against itself
        assert cloud.score_job.local("truth-twin", "bench-a", [], "coverage") == ["runs/truth-twin/eval/coverage.json"]
        with pytest.raises(FileNotFoundError, match="data/nowhere is not on the volume"):
            cloud.score_job.local("r1", "nowhere", [], "probes")
    finally:
        torch.set_num_threads(threads)


def test_score_uploads_a_missing_benchmark_and_copies_the_card_home(tmp_path, monkeypatch, capsys):
    import cloud

    class Entry:
        def __init__(self, path):
            self.path = path

    class Batch:
        put = []

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def put_directory(self, local, remote):
            Batch.put.append((Path(local).name, remote))

    class Volume:
        def listdir(self, path):
            return [Entry("data/bench-a")]

        def batch_upload(self):
            return Batch()

        def read_file(self, rel):
            yield b'{"walls/kept@48": '
            yield b"0.9}"

    class Job:
        calls = []

        def remote(self, *call):
            Job.calls.append(call)
            return ["runs/r1/eval/probes.json", "runs/r1/eval/probes/sheet.png"]

    class Jobs:
        Refused = RuntimeError
        noted = []

        @staticmethod
        def record(what, **fields):
            Jobs.noted.append((what, fields["run"], fields["card"]))

    monkeypatch.setattr(cloud, "volume", Volume())
    monkeypatch.setattr(cloud, "score_job", Job())
    monkeypatch.setattr(cloud, "lab_jobs", lambda: Jobs)
    data, out = tmp_path / "datasets", tmp_path / "runs"
    (data / "bench-b").mkdir(parents=True)
    (data / "bench-b" / "meta.json").write_text("{}")
    written = cloud.score_run("r1", "bench-a", "probes", ["bench-b=walls"], "--no-pictures", str(data), str(out), "why")
    assert written == ["runs/r1/eval/probes.json", "runs/r1/eval/probes/sheet.png"]
    assert Batch.put == [("bench-b", "data/bench-b")], "only what the volume lacks is uploaded"
    assert Job.calls == [("r1", "bench-a", ["bench-b=walls"], "probes", "--no-pictures")] and Jobs.noted == [("score", "r1", "probes")]
    assert (out / "r1" / "eval" / "probes.json").read_text() == '{"walls/kept@48": 0.9}' and (out / "r1" / "eval" / "probes" / "sheet.png").exists()
    assert "scored in" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="data/bench-z is not on the volume, and not in"):
        cloud.score_run("r1", "bench-z", "probes", [], "", str(data), str(out))
    assert len(Job.calls) == 1, "nothing is scored when a benchmark is missing"


@pytest.mark.filterwarnings("ignore:The .* function is executing locally")
def test_job_bodies_run_locally(tmp_path, monkeypatch):
    """What runs in the cloud container, run here instead: generate -> train -> evaluate -> scorecard on a "volume"."""
    import cloud

    class Volume:
        commits = 0

        def commit(self):
            Volume.commits += 1

    monkeypatch.setattr(cloud, "VOL", tmp_path)
    monkeypatch.setattr(cloud, "volume", Volume())
    shape = "--steps 32 --size 32"
    cloud.generate_job.local("tiny", f"--episodes 16 --val-episodes 4 {shape}")
    cloud.generate_job.local("bench-tiny", f"--episodes 0 --val-episodes 6 --seed 9001 {shape}")
    cloud.generate_job.local("tiny", "--episodes 999")  # already there: left alone
    assert (tmp_path / "data" / "tiny" / "meta.json").exists() and Volume.commits == 2

    tiny = "--batch 8 --frame-skip 2 --eval-every 2 --log-every 1 --num-proj 32 --threads 2 --model embed_dim=32 proj_hidden=64 pred_depth=2"
    summary = cloud.train_job.local("r1", "tiny", f"--steps 400 {tiny}", 6, "bench-tiny", 0.004)
    assert summary["gpu"] == "cpu" and summary["gpu min"] >= 0 and summary["skip"] == 2
    assert 1 <= summary["steps"] < 400, "training must have stopped at 80% of the booked minutes"
    run = tmp_path / "runs" / "r1"
    for name in ("ckpt.pt", "metrics.jsonl", "eval/metrics.json", "eval/scorecard.json", "eval/bench/fixed-1.png"):
        assert (run / name).exists(), name

    again = cloud.evaluate_job.local("r1", 6, "")
    assert again["steps"] == summary["steps"] and "gpu min" in again

    with pytest.raises(FileExistsError, match="never overwritten"):  # a finished run is not trained over
        cloud.train_job.local("r1", "tiny", f"--steps 4 {tiny}", 6, "", 0.0)
