"""The scorecard: each check on a hand-made rollout, then the whole thing on a tiny trained run."""

import json

import numpy as np
import pytest

from sandwm import generate, scorecard, train
from sandwm.scorecard import Rollout, diff, gate_results, run_checks

WALL, SAND = 1, 2


def hand_made() -> Rollout:
    """One 6x6 window, read 16 frames after the context.

    Truth: a wall border, two inner wall cells, and one grain of sand that falls one cell.
    Imagination: the grain arrives but also stays where it was; one inner wall cell is lost and
    one is invented.  With the brush hidden nothing moves.  The latent loses one inner wall cell.
    """
    before = np.zeros((6, 6), np.uint8)
    before[[0, -1], :] = before[:, [0, -1]] = WALL
    before[3, 2] = before[3, 3] = WALL
    after = before.copy()
    before[1, 1] = SAND
    after[2, 1] = SAND
    imagined = after.copy()
    imagined[1, 1] = SAND  # the grain is duplicated
    imagined[3, 3] = 0  # a wall cell is lost
    imagined[4, 4] = WALL  # and one appears
    seen = before.copy()
    seen[3, 3] = 0  # what the decoder shows for the last frame the model saw

    def stack(a, b):
        return np.stack([a, b])[None]

    return Rollout(truth=stack(before, after), imagined=stack(seen, imagined), idle=stack(seen, before),
                   recon=stack(seen, after), frame_skip=16, names=("empty", "wall", "sand", "water"))


def test_each_check_on_a_hand_made_rollout():
    r = hand_made()
    assert r.at(16) == 1 and r.at(48) is None and r.at(8) is None and r.horizons == [16]
    c = {name: entry["value"] for name, entry in run_checks(r).items()}
    assert c["walls/kept@16"] == 0.5  # one of the two inner wall cells survives
    assert c["walls/invented@16"] == 0.5  # one invented cell against two true inner cells
    assert c["fidelity/walls"] == 0.5
    assert c["dynamics/changed@16"] == 0.5  # the grain arrived, but its old cell was not cleared
    assert c["dynamics/unchanged@16"] == pytest.approx(32 / 34, abs=1e-4)
    assert c["actions/gain@16"] == 0.5  # with the brush hidden, neither changed cell is right
    assert c["mass/sand@16"] == pytest.approx(1 / 17, abs=1e-4) == c["bias/sand@16"]  # two grains imagined, one true
    assert c["mass/water@16"] == 0 and c["fidelity/iou/water"] is None  # no water anywhere: nothing to measure
    assert c["iou/sand@16"] == 0.5 and c["iou/wall@16"] == pytest.approx(21 / 23, abs=1e-4)
    assert c["rollout/miou@16"] == pytest.approx((21 / 23 + 0.5) / 2, abs=1e-4)
    assert c["rollout/beats_copy@16"] == pytest.approx((21 / 23 + 0.5) / 2 - 0.5, abs=1e-4)
    assert c["fidelity/miou"] == pytest.approx((21 / 22 + 1) / 2, abs=1e-4)
    better = {name: entry["better"] for name, entry in run_checks(r).items()}
    assert better["walls/kept@16"] == "higher" and better["walls/invented@16"] == "lower" and better["bias/sand@16"] == "info"


def card(**checks) -> dict:
    return {"run": "x", "bench": "b", "windows": 1, "frame_skip": 4, "step": 1,
            "checks": {name: {"value": value, "better": better} for name, (value, better) in checks.items()}}


def test_diff_puts_regressions_first_and_ignores_noise():
    old = card(a=(0.50, "higher"), b=(0.20, "lower"), c=(0.70, "higher"), d=(0.1, "info"), gone=(1.0, "higher"), e=(None, "higher"))
    new = card(a=(0.30, "higher"), b=(0.10, "lower"), c=(0.71, "higher"), d=(0.9, "info"), added=(1.0, "higher"), e=(0.5, "higher"))
    rows = diff(old, new)
    assert rows[0] == ("a", 0.5, 0.3, "REGRESSED") and rows[-1] == ("b", 0.2, 0.1, "improved")
    verdicts = {name: verdict for name, _, _, verdict in rows}
    assert verdicts == {"a": "REGRESSED", "b": "improved", "c": "same", "d": "info", "gone": "gone", "added": "new", "e": "n/a"}
    assert {name: v for name, _, _, v in diff(old, new, tol=0.005)}["c"] == "improved"
    text = scorecard.format_diff(old, new)
    assert "1 regressed, 1 improved, 1 unchanged" in text and "\n  c " not in text


def test_gates_use_the_direction_of_the_check():
    c = card(up=(0.9, "higher"), down=(0.3, "lower"), missing=(None, "higher"))
    results = gate_results(c, {"up": 0.9, "down": 0.2, "missing": 0.1, "absent": 0.5})
    assert [(name, ok) for name, _, _, ok in results] == [("up", True), ("down", False), ("missing", False), ("absent", False)]
    text = scorecard.format_card(c, {"up": 0.9, "down": 0.2, "absent": 0.5})
    assert "gate 0.9: pass" in text and "gate 0.2: FAIL" in text and "no such check" in text and "gates passed: 1 of 3" in text


def test_scorecard_on_a_tiny_run(tmp_path, capsys):
    data, bench, run = tmp_path / "train", tmp_path / "bench-tiny", tmp_path / "run"
    shape = ["--steps", "32", "--size", "32"]
    generate.main(["--out", str(data), "--episodes", "16", "--val-episodes", "4", *shape])
    generate.main(["--out", str(bench), "--episodes", "0", "--val-episodes", "6", "--seed", "9001", *shape])
    assert not (bench / "train").exists() and (bench / "preview.png").exists()

    # a time limit ends training early and still leaves a checkpoint behind
    train.main(["--data", str(data), "--out", str(run), "--steps", "500", "--max-minutes", "0.002", "--batch", "8", "--frame-skip", "2",
                "--eval-every", "250", "--log-every", "1", "--num-proj", "32", "--threads", "2",
                "--model", "embed_dim=32", "proj_hidden=64", "pred_depth=2"])
    assert "the limit of 0.002 minutes is reached" in capsys.readouterr().out
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert 1 <= rows[-1]["step"] < 500 and "val_pred" in rows[-1]

    scorecard.main(["--run", str(run), "--bench", str(bench), "--frames", "8", "16"])
    result = scorecard.load_card(run)
    assert result["bench"] == "bench-tiny" and result["windows"] == 6 and result["frame_skip"] == 2
    assert result["read_at_frames"] == [8, 16] and result["step"] == rows[-1]["step"]
    for name in ("fidelity/miou", "fidelity/walls", "rollout/miou@8", "rollout/beats_copy@16", "dynamics/changed@16",
                 "actions/gain@8", "walls/kept@16", "walls/invented@16", "mass/sand@16", "mass/water@8", "bias/sand@16"):
        assert name in result["checks"], name
    assert all(e["value"] is None or -1e6 < e["value"] < 1e6 for e in result["checks"].values())
    for name in ("fixed-1.png", "fixed-2.png", "worst-1.png", "worst-2.png", "windows.json"):
        assert (run / "eval" / "bench" / name).stat().st_size > 0
    shown = json.loads((run / "eval" / "bench" / "windows.json").read_text())
    assert shown["shown_fixed"] == [0, 1, 2, 3, 4, 5] and len(shown["miou_per_window"]) == 6

    gates = tmp_path / "gates.toml"
    gates.write_text('[gates]\n"walls/invented@16" = 1000.0\n"fidelity/miou" = 2.0\n')
    capsys.readouterr()
    scorecard.main(["--run", str(run), "--gates", str(gates)])  # without --bench: print the card the run already has
    out = capsys.readouterr().out
    assert "gate 1000: pass" in out and "gate 2: FAIL" in out and "gates passed: 1 of 2" in out
    scorecard.main(["--diff", str(run), str(run)])
    assert "0 regressed, 0 improved" in capsys.readouterr().out

    with pytest.raises(ValueError, match="none of the read-out points"):  # the benchmark is too short for 48 frames
        scorecard.score(run, bench, frames=(48,), pictures=False)
