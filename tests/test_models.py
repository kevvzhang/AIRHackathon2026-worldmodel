import pytest
import torch

from sandwm.models import PRESETS, JEPA, Decoder, ModelConfig, SIGReg, count_params, jepa_loss, parse_overrides


def small_cfg(**kw):
    base = dict(image_size=32, action_dim=7, embed_dim=32, proj_hidden=64, cnn_channels=(8, 16, 32), pred_depth=2,
                pred_heads=2, pred_dim_head=8, pred_mlp=64, dec_channels=(16, 8, 8), vit_dim=32, vit_depth=2, vit_heads=2)
    return ModelConfig(**{**base, **kw})


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.mark.parametrize("encoder", ["cnn", "vit"])
def test_shapes(encoder):
    cfg = small_cfg(encoder=encoder)
    model, decoder = JEPA(cfg), Decoder(cfg)
    obs, act = torch.randn(4, 4, 3, 32, 32), torch.randn(4, 3, 7)
    emb = model.encode(obs)
    assert emb.shape == (4, 4, 32)
    assert model.predict(emb[:, :3], act).shape == (4, 3, 32)
    assert decoder(emb[:, 0]).shape == (4, cfg.n_classes, 32, 32)
    model.eval()
    assert model.rollout(emb[:, :2], torch.randn(4, 2 + 5 - 1, 7)).shape == (4, 5, 32)
    assert model.rollout(emb[:, :1], torch.randn(4, 9, 7)).shape == (4, 9, 32)  # context shorter than the history


def test_config_round_trips_through_a_checkpoint_dict():
    cfg = small_cfg(pred_residual=False)
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg


def test_presets_and_overrides():
    assert parse_overrides(["vit_patch=4", "pred_residual=false", "cnn_channels=8,16", "encoder=vit", "pred_dropout=0.0"]) == dict(
        vit_patch=4, pred_residual=False, cnn_channels=(8, 16), encoder="vit", pred_dropout=0.0)
    with pytest.raises(ValueError, match="unknown model option"):
        parse_overrides(["patch=4"])
    assert ModelConfig(**PRESETS["cpu"]) == ModelConfig()
    lewm = JEPA(ModelConfig(**PRESETS["lewm"]))
    # the paper quotes ~5M parameters for the ViT-Tiny encoder and ~10M for the predictor
    assert 5.0e6 < count_params(lewm.encoder) < 6.0e6
    assert 9.0e6 < count_params(lewm.predictor) < 11.0e6
    assert lewm.encode(torch.randn(2, 1, 3, 64, 64)).shape == (2, 1, 192)


def test_predictor_is_causal():
    model = JEPA(small_cfg()).eval()
    for p in model.parameters():  # un-zero the AdaLN gates so that attention is actually in play
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.5)
    emb, act = torch.randn(2, 3, 32), torch.randn(2, 3, 7)
    base = model.predict(emb, act)
    emb2, act2 = emb.clone(), act.clone()
    emb2[:, 2] += 1.0
    act2[:, 2] += 1.0
    out = model.predict(emb2, act2)
    assert torch.allclose(out[:, :2], base[:, :2], atol=1e-5), "an output depends on a later frame or action"
    assert not torch.allclose(out[:, 2], base[:, 2], atol=1e-3)
    act3 = act.clone()
    act3[:, 0] += 1.0
    assert not torch.allclose(model.predict(emb, act3)[:, 2], base[:, 2], atol=1e-4), "actions are ignored"


def test_residual_predictor_starts_as_the_copy_baseline():
    emb, act = torch.randn(2, 3, 32), torch.randn(2, 3, 7)
    assert torch.equal(JEPA(small_cfg()).eval().predict(emb, act), emb)
    assert not torch.allclose(JEPA(small_cfg(pred_residual=False)).eval().predict(emb, act), emb, atol=1e-2)


def test_rollout_matches_teacher_forcing_on_the_first_step():
    model = JEPA(small_cfg()).eval()
    torch.nn.init.normal_(model.pred_proj.net[-1].weight, std=0.1)
    emb, act = torch.randn(2, 3, 32), torch.randn(2, 6, 7)
    rolled = model.rollout(emb, act)
    assert torch.allclose(rolled[:, 0], model.predict(emb, act[:, :3])[:, -1], atol=1e-6)
    second = model.predict(torch.cat([emb[:, 1:], rolled[:, :1]], 1), act[:, 1:4])[:, -1]
    assert torch.allclose(rolled[:, 1], second, atol=1e-6)


def test_sigreg_is_minimal_for_an_isotropic_gaussian():
    sigreg = SIGReg(num_proj=256)
    n, d = 512, 16
    gauss = sigreg(torch.randn(n, d)).item()
    # under the null the Epps-Pulley statistic has expectation int (1 - exp(-t^2)) exp(-t^2/2) dt ~= 1.05
    assert 0.7 < gauss < 1.5
    collapsed = sigreg(torch.zeros(n, d)).item()
    low_rank = sigreg(torch.randn(n, 2) @ torch.randn(2, d)).item()
    scaled = sigreg(3 * torch.randn(n, d)).item()
    shifted = sigreg(torch.randn(n, d) + 2).item()
    assert min(collapsed, low_rank, scaled, shifted) > 5 * gauss
    # Right mean and variance but the wrong shape.  In one dimension this is caught ...
    two_points = sigreg(torch.randn(n, 1).sign()).item()
    assert two_points > 5 * sigreg(torch.randn(n, 1)).item()
    # ... but NOT in many dimensions when the coordinates are independent: a random projection is
    # then a weighted sum of them, which the central limit theorem makes look Gaussian.  SIGReg
    # constrains random projections only -- worth remembering when reading its value.
    assert sigreg(torch.randn(n, d).sign()).item() < 1.5 * gauss


def test_sigreg_scores_each_leading_group_separately():
    sigreg = SIGReg(num_proj=64)
    z = torch.randn(3, 128, 8)
    z[1] += 5  # one time step is far from Gaussian(0, 1)

    def score(x):
        torch.manual_seed(1)  # same random directions on every call
        return sigreg(x)

    assert torch.allclose(score(z), torch.stack([score(z[i]) for i in range(3)]).mean(), rtol=1e-5)
    assert score(z) > 3 * score(z[0])


def test_loss_trains_encoder_and_predictor_end_to_end():
    cfg = small_cfg()
    model, sigreg = JEPA(cfg), SIGReg(64)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    obs, act = torch.randn(16, 4, 3, 32, 32), torch.randn(16, 3, 7)

    def step():
        opt.zero_grad()
        loss, out = jepa_loss(model, sigreg, obs, act, lambd=0.1)
        loss.backward()
        opt.step()
        return out

    def has_grad(module):
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

    out = step()
    assert out["emb"].shape == (16, 4, 32) and out["pred_emb"].shape == (16, 3, 32)
    assert has_grad(model.encoder) and has_grad(model.projector) and has_grad(model.pred_proj)
    # By design the zero-initialised layers open up one after another: the residual output layer
    # on step 1, the AdaLN modulation on step 2, and through it the action encoder on step 3.
    assert not has_grad(model.predictor)
    step()
    assert has_grad(model.predictor) and not has_grad(model.action_encoder)
    step()
    assert has_grad(model.action_encoder)

    # no stop-gradient: the prediction loss alone must reach the encoder through the targets too
    model.zero_grad()
    emb = model.encode(obs)
    (model.predict(emb[:, :-1].detach(), act) - emb[:, 1:]).square().mean().backward()
    assert has_grad(model.encoder)
