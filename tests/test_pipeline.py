"""Data round-trip and a tiny end-to-end run: generate -> train -> evaluate."""

import json

import numpy as np
import pytest
import torch

from sandwm import compare, evaluate, generate, train
from sandwm.data import EpisodeDataset, encode_actions
from sandwm.materials import BASIC, WALL
from sandwm.sim import SandSim


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "tiny"
    generate.main(["--out", str(out), "--episodes", "24", "--val-episodes", "8", "--steps", "24", "--size", "32", "--shard", "16"])
    return out


def test_dataset_layout_and_contents(dataset_dir):
    meta = json.loads((dataset_dir / "meta.json").read_text())
    assert meta["materials"] == "basic" and meta["tools"] == list(BASIC.tools)
    assert len(list((dataset_dir / "train").glob("shard-*.npz"))) == 2
    assert (dataset_dir / "preview.png").exists() and (dataset_dir / "preview.gif").exists()
    ds = EpisodeDataset(dataset_dir, "train")
    assert ds.frames.shape == (24, 25, 32, 32) and ds.frames.dtype == np.uint8
    assert ds.actions.shape == (24, 24, 3) and ds.frames.max() < len(BASIC)
    assert (ds.frames[:, :, 0] == WALL).all()
    assert len(EpisodeDataset(dataset_dir, "train", max_episodes=5)) == 5
    val = EpisodeDataset(dataset_dir, "val")
    assert len(val) == 8 and not (val.frames[:8] == ds.frames[:8]).all(), "val must not repeat train"


def test_generation_is_reproducible(dataset_dir, tmp_path):
    generate.main(["--out", str(tmp_path / "again"), "--episodes", "24", "--val-episodes", "8", "--steps", "24", "--size", "32", "--shard", "16"])
    a, b = EpisodeDataset(dataset_dir, "train"), EpisodeDataset(tmp_path / "again", "train")
    assert (a.frames == b.frames).all() and (a.actions == b.actions).all()
    with pytest.raises(SystemExit):  # refuses to overwrite an existing dataset
        generate.main(["--out", str(tmp_path / "again")])


def test_stored_actions_reproduce_the_stored_frames_in_deterministic_mode(tmp_path):
    out = tmp_path / "det"
    generate.main(["--out", str(out), "--episodes", "4", "--val-episodes", "1", "--steps", "20", "--size", "32", "--deterministic"])
    ds = EpisodeDataset(out, "train")
    sim = SandSim(len(ds), 32, BASIC, **ds.meta["sim"])
    sim.reset(ds.frames[:, 0])
    for t in range(ds.episode_len):
        assert (sim.step(ds.actions[:, t]) == ds.frames[:, t + 1]).all(), f"replay diverged at step {t}"


def test_windows_and_frame_skip(dataset_dir):
    ds = EpisodeDataset(dataset_dir, "train")
    frames, actions = ds.window(np.array([3, 7]), np.array([2, 5]), n_frames=4, frame_skip=3)
    assert frames.shape == (2, 4, 32, 32) and actions.shape == (2, 3, 3, 3)
    assert (frames[1] == ds.frames[7, [5, 8, 11, 14]]).all()
    assert (actions[1] == ds.actions[7, 5:14].reshape(3, 3, 3)).all()
    f, a = ds.sample(16, 4, 2, np.random.default_rng(0))
    assert f.shape == (16, 4, 32, 32) and a.shape == (16, 3, 2, 3)
    with pytest.raises(ValueError):
        ds.sample(1, 30, 1, np.random.default_rng(0))


def test_grouped_batches_and_spread_frames(dataset_dir):
    ds = EpisodeDataset(dataset_dir, "train")

    def source_episodes(windows):  # brute force: which episode each 4-frame window was cut from
        return [next(e for e in range(len(ds)) for t in range(ds.episode_len - 2) if (ds.frames[e, t : t + 4] == w).all())
                for w in windows]

    f, a = ds.sample(32, 4, 1, np.random.default_rng(0), episodes_per_batch=4)
    assert f.shape == (32, 4, 32, 32) and a.shape == (32, 3, 1, 3)
    counts = np.bincount(source_episodes(f), minlength=len(ds))
    assert sorted(counts[counts > 0]) == [8, 8, 8, 8], "expected 4 episodes contributing 8 windows each"
    assert len(set(source_episodes(ds.sample(32, 4, 1, np.random.default_rng(0))[0]))) > 8  # default: spread widely

    spread = ds.spread(6, 5, np.random.default_rng(0))
    assert spread.shape == (6, 5, 32, 32)
    episode = next(e for e in range(len(ds)) if (ds.frames[e, 0] == spread[0, 0]).all())
    assert (spread[0] == ds.frames[episode, [0, 6, 12, 18, 24]]).all()


def test_observations_and_action_encoding(dataset_dir):
    ds = EpisodeDataset(dataset_dir, "train")
    obs = ds.observe(ds.frames[:2, :3])
    assert obs.shape == (2, 3, 3, 32, 32) and -1 <= obs.min() and obs.max() <= 1
    assert torch.allclose(obs[0, 0, :, 0, 0], torch.tensor(BASIC.palette[WALL] / 127.5 - 1, dtype=torch.float32), atol=1e-6)
    raw = torch.tensor([[[0, 0, 0]], [[31, 0, 2]], [[5, 9, 0]]])  # (3 steps, frame_skip 1, 3)
    enc = encode_actions(raw, size=32, n_tools=5)
    assert enc.shape == (3, 7)
    assert enc[1].tolist() == [1, -1, 0, 0, 1, 0, 0]
    assert torch.equal(enc[0], enc[2]), "'do nothing' must encode identically wherever the cursor is"
    assert encode_actions(torch.zeros(4, 3, 2, 3, dtype=torch.long), 32, 5).shape == (4, 3, 14)


@pytest.mark.parametrize("frame_skip", [1, 2])
def test_train_then_evaluate(dataset_dir, tmp_path, frame_skip, capsys):
    run = tmp_path / "run"
    extra = ["--episodes-per-batch", "2", "--recon-weight", "0.5"] if frame_skip == 2 else []
    train.main(["--data", str(dataset_dir), "--out", str(run), "--steps", "4", "--batch", "8", "--eval-every", "2",
                "--log-every", "1", "--num-proj", "32", "--frame-skip", str(frame_skip), "--threads", "2",
                "--model", "embed_dim=32", "proj_hidden=64", "pred_depth=2", *extra])
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert sum("val_pred" in r for r in rows) == 2 and all(np.isfinite(r["pred"]) for r in rows if "pred" in r)
    assert all(0 <= r["emb_within"] <= 1 and 0 < r["pixel_within"] < 1 for r in rows if "val_pred" in r)
    assert (run / "ckpt.pt").exists() and (run / "curves.png").exists()

    evaluate.main(["--run", str(run), "--windows", "16", "--horizon", "6", "--show", "2"])
    result = json.loads((run / "eval" / "metrics.json").read_text())
    assert result["frame_skip"] == frame_skip and len(result["rollout"]["horizon"]) == 6
    assert set(result["probes"]) >= {"sand: amount", "water: centre y", "wall: centre x"}
    for name in ("rollout_curves.png", "rollout_sheet.png", "rollout.gif"):
        assert (run / "eval" / name).stat().st_size > 0
    for series in ("model", "frozen", "ceiling", "pixel_copy"):
        assert len(result["rollout"]["miou"][series]) == 6

    summary = compare.summarise(run, frames_ahead=4)
    assert summary["skip"] == frame_skip and summary["steps"] == 4
    capsys.readouterr()
    compare.main([str(run), "--frames-ahead", "4"])
    table = capsys.readouterr().out.splitlines()
    assert table[0].startswith("run") and table[2].startswith("run ") and "within" in table[0]
    assert 0 <= result["within_episode_share"]["embedding"] <= 1
