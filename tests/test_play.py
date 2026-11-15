"""The interactive player: its bookkeeping must match the reference rollout, and the server must answer."""

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
import torch

from sandwm.data import encode_actions, observe
from sandwm.materials import BASIC
from sandwm.models import JEPA, Decoder, ModelConfig
from sandwm.play import PlaySession, find_runs, make_handler
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
    assert find_runs(tmp_path / "runs") == ["a", "b"]
    for a in strokes(3):
        session.step(a)
    before = session.sim.grid[0].copy()
    session.load(tmp_path / "runs" / "b")
    assert session.skip == 2 and (session.sim.grid[0] == before).all() and session.since_sync == 0
    session.step(strokes(1)[0])
    assert session.step(strokes(1)[0])["model_stepped"]


def test_server_endpoints(tmp_path):
    from http.server import ThreadingHTTPServer

    runs = tmp_path / "runs"
    session = PlaySession(make_run(runs / "a"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(session, runs))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path, body):
        req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    try:
        with urllib.request.urlopen(url + "/", timeout=30) as r:
            assert b"<canvas" in r.read()
        with urllib.request.urlopen(url + "/api/state", timeout=30) as r:
            state = json.loads(r.read())
        assert state["runs"] == ["a"] and state["info"]["tools"] == list(BASIC.tools)
        s = post("/api/step", {"action": [5, 5, 3], "auto_sync": 0, "model_sees_brush": True})
        assert s["model_stepped"] and len(s["truth"]) == len(s["imagined"]) > 0 and set(s["metrics"]) >= {"imagined", "frozen", "ceiling"}
        assert post("/api/reset", {"scene": "empty", "seed": 3})["step"] == 0
        with pytest.raises(urllib.error.HTTPError) as err:
            post("/api/load", {"run": "../../etc"})
        assert err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
