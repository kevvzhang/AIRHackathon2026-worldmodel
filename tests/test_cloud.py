"""The Modal app must at least import: it is only exercised for real in the cloud."""

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


class FakeJob:
    """Stands in for a Modal function: `starmap` hands back prepared results instead of running anything."""

    def __init__(self, results):
        self.results, self.calls = results, None

    def starmap(self, calls, return_exceptions=False):
        self.calls = list(calls)
        yield from self.results


def test_launch_books_and_settles_every_job(monkeypatch, capsys):
    import cloud

    class Booking:
        minutes, settled, closed = 5.0, [], False

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
    assert job.calls is None, "nothing may reach the cloud after a refusal"


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
