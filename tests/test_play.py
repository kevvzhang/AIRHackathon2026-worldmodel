"""The interactive player: its bookkeeping must match the reference rollout, every run must be played
on its own code, and the server must answer."""

import json
import subprocess
import threading
from pathlib import Path
import urllib.error
import urllib.request

import numpy as np
import pytest
import torch

from sandwm.data import encode_actions, observe
from sandwm.materials import BASIC
from sandwm.models import JEPA, Decoder, ModelConfig
from sandwm import play, playsession
from sandwm.play import Player, Server, code_snapshot, find_runs, make_handler, run_code
from sandwm.playsession import PlaySession, handle
from sandwm.scenes import random_scene
from sandwm.sim import SandSim

SIM = {"stochastic": True, "sink_prob": 0.5, "slide_prob": 1.0, "spread_prob": 0.9, "brush_radius": 1}


def make_run(path, frame_skip: int = 1, seed: int = 0):
    """A tiny randomly initialised checkpoint in the format written by sandwm.train."""
    torch.manual_seed(seed)
    n_tools = len(BASIC.tools)
    cfg = ModelConfig(image_size=32, action_dim=(2 + n_tools) * frame_skip, n_classes=len(BASIC), embed_dim=32,
                      proj_hidden=64, cnn_channels=(8, 16), pred_depth=2, pred_heads=2, pred_dim_head=8, pred_mlp=32,
                      dec_channels=(16, 8))
    model, decoder = JEPA(cfg), Decoder(cfg)
    with torch.no_grad():  # move away from the zero-initialised identity predictor, so the test means something
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    path.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "decoder": decoder.state_dict(), "config": cfg.to_dict(), "step": 7,
                "args": {"frame_skip": frame_skip, "data": "data/tiny", "recon_weight": 0.0},
                "data_meta": {"materials": "basic", "size": 32, "sim": SIM}}, path / "ckpt.pt")
    return path


def strokes(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(1, 31, n), rng.integers(1, 31, n), rng.integers(0, len(BASIC.tools), n)], 1).astype(np.int16)


@pytest.mark.parametrize("frame_skip", [1, 2])
def test_open_loop_play_matches_reality_and_the_reference_rollout(tmp_path, frame_skip):
    session = PlaySession(make_run(tmp_path / "run", frame_skip))
    session.reset("random", seed=5)
    start = session.sim.grid[0].copy()
    acts = strokes(6 * frame_skip)
    for a in acts:
        state = session.step(a)
    assert state["step"] == 6 and state["tick"] == 6 * frame_skip and state["since_sync"] == 6

    # reality: the same simulator, the same seed, the same actions
    sim = SandSim(1, 32, BASIC, seed=[5, 1], **SIM)
    sim.reset(start[None])
    for a in acts:
        sim.step(a[None])
    assert (sim.grid[0] == session.sim.grid[0]).all()

    # imagination: JEPA.rollout from the first frame, given the same action blocks
    model = session.model
    with torch.no_grad():
        z0 = model.encode(observe(torch.from_numpy(start[None, None]), BASIC.palette))
        blocks = encode_actions(torch.from_numpy(acts.reshape(1, 6, frame_skip, 3)), 32, len(BASIC.tools))
        expected = model.rollout(z0, blocks)[0]
    torch.testing.assert_close(session.latents, expected[-session.history:])


def test_sync_auto_sync_and_hidden_brush(tmp_path):
    session = PlaySession(make_run(tmp_path / "run"))
    session.reset("empty", seed=1)
    for a in strokes(4):
        session.step(a)
    session.sync()
    with torch.no_grad():
        truth = session._encode(np.stack(session.frames))[0]
    torch.testing.assert_close(session.latents, truth)
    assert session.since_sync == 0 and session.synced_at == 4

    session.auto_sync = 3
    flags = [session.step(a).get("synced_after", False) for a in strokes(7, seed=1)]
    assert flags == [False, False, True, False, False, True, False]

    # with the brush hidden, the model's action is "none" whatever is painted
    session.model_sees_brush = False
    paint = np.array([10, 10, BASIC.tool("sand")], np.int16)
    session.step(paint)
    assert session.actions[-1][2] == 1 and session.actions[-1][:2].abs().sum() == 0  # one-hot "none", cursor zeroed
    assert (session.sim.grid[0][9:12, 9:12] != 0).any(), "reality still gets painted"


def test_switching_models_keeps_the_world(tmp_path):
    session = PlaySession(make_run(tmp_path / "runs" / "a"))
    make_run(tmp_path / "runs" / "b", frame_skip=2, seed=1)
    assert [r["name"] for r in find_runs(tmp_path / "runs")] == ["a", "b"]
    for a in strokes(3):
        session.step(a)
    before = session.sim.grid[0].copy()
    session.load(tmp_path / "runs" / "b")
    assert session.skip == 2 and (session.sim.grid[0] == before).all() and session.since_sync == 0
    session.step(strokes(1)[0])
    assert session.step(strokes(1)[0])["model_stepped"]


def test_run_menu_spans_folders_and_marks_the_lab_champion(tmp_path):
    make_run(tmp_path / "runs" / "a")
    make_run(tmp_path / ".lab" / "runs" / "b")
    make_run(tmp_path / ".lab" / "runs" / "c")
    (tmp_path / ".lab" / "runs" / "no-ckpt").mkdir()
    (tmp_path / ".lab" / "board").mkdir()
    (tmp_path / ".lab" / "board" / "champion.txt").write_text("c\n\nThe first word of this file is the champion.\n")
    runs = find_runs([tmp_path / "runs", tmp_path / ".lab" / "runs"])
    assert [(r["name"], r["champion"]) for r in runs] == [("a", False), ("b", False), ("c", True)]
    assert len({r["group"] for r in runs}) == 2 and runs[1]["group"] == runs[2]["group"]


def test_a_run_from_newer_code_is_refused_clearly(tmp_path):
    session = PlaySession(make_run(tmp_path / "runs" / "a"))
    new = make_run(tmp_path / "runs" / "grid", seed=1)
    ckpt = torch.load(new / "ckpt.pt", weights_only=True)
    ckpt["config"]["latent_grid"] = 8  # an option this code does not know
    torch.save(ckpt, new / "ckpt.pt")
    with pytest.raises(ValueError, match="options this code does not have"):
        session.load(new)
    assert session.run.name == "a" and session.step(strokes(1)[0])["model_stepped"]  # still playing the old model


def test_new_scenes_are_drawn_like_the_training_scenes(tmp_path, monkeypatch):
    """The run's terrain options reach random_scene, as far as the code it is played on has them."""
    calls = []

    def scene(rng, size, ms, cups=(0, 2), ramps=(0, 2)):  # a random_scene that knows two terrain options
        calls.append({"cups": cups, "ramps": ramps})
        return random_scene(rng, size, ms)

    monkeypatch.setattr(playsession, "random_scene", scene)
    run = make_run(tmp_path / "runs" / "a")
    ckpt = torch.load(run / "ckpt.pt", weights_only=True)
    ckpt["data_meta"]["terrain"] = {"cups": [3, 3], "wood": [1, 4]}  # wood: an option this code does not have
    torch.save(ckpt, run / "ckpt.pt")
    session = PlaySession(run)
    session.reset("random", seed=4)
    assert session.terrain == {"cups": (3, 3)} and calls[-1] == {"cups": (3, 3), "ramps": (0, 2)}


def test_the_world_carries_over_to_another_session(tmp_path):
    a = PlaySession(make_run(tmp_path / "runs" / "a"))
    for act in strokes(5):
        a.step(act)
    world = a.world()
    b = PlaySession(make_run(tmp_path / "runs" / "b", frame_skip=2, seed=1))
    reply = handle(b, "adopt", world)
    assert reply["carried"] and (b.sim.grid[0] == a.sim.grid[0]).all() and b.tick == a.tick and b.since_sync == 0
    assert not b.adopt({**world, "grid": world["grid"][:8]}), "a world of another size is not adopted"


def test_runs_without_a_tag_use_this_checkout():
    assert run_code("no-such-run-0000") == (None, "this checkout")


def _head_tree():
    tree = play._git("rev-parse", "-q", "--verify", "HEAD:sandwm")
    if tree is None:
        pytest.skip("not in a git checkout")
    return tree


def test_code_snapshot_holds_the_package_at_a_commit():
    tree = _head_tree()
    root = code_snapshot(tree)
    at_commit = subprocess.run(["git", "show", f"{tree}:models.py"], cwd=play.CHECKOUT, capture_output=True, check=True).stdout
    lf = lambda b: b.replace(b"\r\n", b"\n")  # git archive may write Windows line endings
    assert lf((root / "sandwm" / "models.py").read_bytes()) == lf(at_commit)
    assert (root / "sandwm" / "playsession.py").read_bytes() == play.SESSION_MODULE.read_bytes()
    assert code_snapshot(tree) == root  # copied out once


def test_server_and_model_processes(tmp_path, monkeypatch):
    """The page's API end to end, with real model processes: same-code switches stay in one process,
    a run tagged with other code gets a new process on its own copy of the code, the world carries over."""
    runs = tmp_path / "runs"
    make_run(runs / "a")
    make_run(runs / "b", frame_skip=2, seed=1)
    make_run(runs / "c", seed=2)
    tree = _head_tree()
    real_run_code = play.run_code
    monkeypatch.setattr(play, "run_code", lambda run: (tree, "test1234") if Path(run).name == "c" else real_run_code(run))

    player = Player()
    server = Server(("127.0.0.1", 0), make_handler(player, runs))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path, body):
        req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

    try:
        ids = [r["id"] for r in find_runs(runs)]
        info = player.load(ids[0])["info"]
        assert info["code"] == "this checkout" and Path(info["code_dir"]) == Path(play.__file__).resolve().parent
        with urllib.request.urlopen(url + "/", timeout=30) as r:
            assert b"<canvas" in r.read()
        with urllib.request.urlopen(url + "/api/state", timeout=30) as r:
            state = json.loads(r.read())
        assert [r["name"] for r in state["runs"]] == ["a", "b", "c"] and state["info"]["tools"] == list(BASIC.tools)
        assert state["info"]["run"] == ids[0]
        s = post("/api/step", {"action": [5, 5, 3], "auto_sync": 0, "model_sees_brush": True})
        assert s["model_stepped"] and len(s["truth"]) == len(s["imagined"]) > 0 and set(s["metrics"]) >= {"imagined", "frozen", "ceiling"}
        assert post("/api/reset", {"scene": "empty", "seed": 3})["step"] == 0
        post("/api/step", {"action": [16, 16, 3]})

        first = player.worker.proc.pid
        loaded = post("/api/load", {"run": ids[1]})  # same code: same process
        assert loaded["info"]["name"] == "b" and loaded["info"]["frame_skip"] == 2 and player.worker.proc.pid == first
        truth = loaded["state"]["truth"]

        loaded = post("/api/load", {"run": ids[2]})  # other code: a new process on its own copy, same world
        assert player.worker.proc.pid != first and loaded["info"]["code"] == "test1234"
        assert Path(loaded["info"]["code_dir"]) == code_snapshot(tree) / "sandwm"
        assert loaded["carried"] and loaded["state"]["truth"] == truth

        with pytest.raises(urllib.error.HTTPError) as err:
            post("/api/load", {"run": "../../etc"})
        assert err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        player.close()
