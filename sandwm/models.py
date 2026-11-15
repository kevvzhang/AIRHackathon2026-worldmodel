"""A LeWM-style JEPA world model, scaled down so that it trains on a laptop CPU.

Re-implementation of the ideas in LeWorldModel (Maes et al., 2026; github.com/lucas-maes/le-wm):

* an encoder maps each frame to ONE embedding vector, through a projector containing BatchNorm;
* an autoregressive causal transformer predicts the next embedding from a short history, with
  the action injected through AdaLN-zero (so the predictor starts out as the identity);
* the loss is  MSE(predicted next embedding, actual next embedding) + lambda * SIGReg(embeddings),
  trained end-to-end: no stop-gradient, no EMA target network, no reconstruction term.

The pixel decoder defined here is NOT part of the world model: it is trained on detached
embeddings, purely so that we can look at what the latent state contains.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    image_size: int = 64
    action_dim: int = 7  # per latent step = frame_skip * (2 + number of tools)
    n_classes: int = 4  # number of materials (decoder output classes)
    embed_dim: int = 192
    history: int = 3  # context length of the predictor
    proj_hidden: int = 1024
    encoder: str = "cnn"  # "cnn" (fast on CPU) or "vit" (as in the paper)
    cnn_channels: tuple[int, ...] = (32, 64, 128, 128)
    vit_patch: int = 8
    vit_dim: int = 192
    vit_depth: int = 6
    vit_heads: int = 3
    pred_depth: int = 4
    pred_heads: int = 8
    pred_dim_head: int = 32
    pred_mlp: int = 768
    pred_dropout: float = 0.1
    # True: the predictor outputs the CHANGE in embedding (zero-initialised), so it starts out as
    # the "nothing changes" baseline.  False: it outputs the next embedding directly, as in LeWM.
    pred_residual: bool = True
    dec_channels: tuple[int, ...] = (96, 48, 24, 16)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = dict(d)
        for key in ("cnn_channels", "dec_channels"):
            d[key] = tuple(d[key])
        return cls(**d)


# --- anti-collapse regulariser ----------------------------------------------------------------
class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regulariser (LeJEPA / LeWM).

    Projects the embeddings onto random unit directions and scores every 1-D projection with the
    Epps-Pulley test: the squared distance between its empirical characteristic function and that
    of N(0, 1), integrated over t in [-t_max, t_max] under a Gaussian window.  Zero iff every
    projection is standard normal, i.e. iff the embeddings are isotropic Gaussian -- which rules
    out both collapse to a point and collapse onto a subspace.
    """

    def __init__(self, num_proj: int = 1024, knots: int = 17, t_max: float = 3.0) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, t_max, knots)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt)  # trapezoid rule, doubled: the integrand is even in t
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)  # characteristic function of N(0, 1)
        self.register_buffer("weights", weights * window)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., N, D).  The statistic is computed across the N samples and then averaged."""
        directions = torch.randn(z.size(-1), self.num_proj, device=z.device, dtype=z.dtype)
        directions = directions / directions.norm(dim=0)
        x_t = (z @ directions).unsqueeze(-1) * self.t  # (..., N, M, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights) * z.size(-2)).mean()


# --- building blocks ----------------------------------------------------------------------------
class Projector(nn.Module):
    """Linear -> BatchNorm -> GELU -> Linear.  The BatchNorm matters: LeWM reports that SIGReg
    cannot be optimised well through an encoder that ends in LayerNorm."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(dim, heads * dim_head * 3, bias=False)
        self.out = nn.Linear(heads * dim_head, dim)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=causal)
        return self.out(y.transpose(1, 2).reshape(B, T, -1))


def mlp(dim: int, hidden: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))


class Block(nn.Module):
    """Plain pre-norm transformer block (ViT encoder)."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn, self.mlp = Attention(dim, heads, dim_head), mlp(dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class AdaLNBlock(nn.Module):
    """Causal transformer block conditioned through AdaLN-zero (as in DiT / LeWM).

    The condition produces a shift and scale for each normalised input and a gate for each
    residual branch.  All of them start at zero, so every block starts as the identity.
    """

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn, self.mlp = Attention(dim, heads, dim_head, dropout), mlp(dim, mlp_dim, dropout)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(c).chunk(6, dim=-1)
        x = x + gate1 * self.attn(self.norm1(x) * (1 + scale1) + shift1, causal=True)
        return x + gate2 * self.mlp(self.norm2(x) * (1 + scale2) + shift2)


# --- encoders -----------------------------------------------------------------------------------
class CNNEncoder(nn.Module):
    """Stride-2 conv stack; the final feature map is flattened (not pooled) to keep 'where'."""

    def __init__(self, image_size: int, channels: tuple[int, ...]) -> None:
        super().__init__()
        layers, c_in, size = [], 3, image_size
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1), nn.GroupNorm(8, c), nn.GELU()]
            c_in, size = c, size // 2
        self.net = nn.Sequential(*layers, nn.Flatten())
        self.out_dim = c_in * size * size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ViTEncoder(nn.Module):
    """Small ViT; returns the CLS token after the final LayerNorm."""

    def __init__(self, image_size: int, patch: int, dim: int, depth: int, heads: int) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, dim, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, (image_size // patch) ** 2 + 1, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, heads, dim // heads, 4 * dim) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]


# --- predictor ----------------------------------------------------------------------------------
class Predictor(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.embed_dim
        self.pos = nn.Parameter(torch.randn(1, cfg.history, d) * 0.02)
        self.blocks = nn.ModuleList(
            AdaLNBlock(d, cfg.pred_heads, cfg.pred_dim_head, cfg.pred_mlp, cfg.pred_dropout) for _ in range(cfg.pred_depth)
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = z + self.pos[:, : z.size(1)]
        for block in self.blocks:
            x = block(x, c)
        return self.norm(x)


# --- the world model ----------------------------------------------------------------------------
class JEPA(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim
        if cfg.encoder == "cnn":
            self.encoder = CNNEncoder(cfg.image_size, cfg.cnn_channels)
        elif cfg.encoder == "vit":
            self.encoder = ViTEncoder(cfg.image_size, cfg.vit_patch, cfg.vit_dim, cfg.vit_depth, cfg.vit_heads)
        else:
            raise ValueError(f"unknown encoder {cfg.encoder!r}")
        self.projector = Projector(self.encoder.out_dim, cfg.proj_hidden, d)
        self.action_encoder = nn.Sequential(nn.Linear(cfg.action_dim, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))
        self.predictor = Predictor(cfg)
        self.pred_proj = Projector(d, cfg.proj_hidden, d)
        if cfg.pred_residual:
            nn.init.zeros_(self.pred_proj.net[-1].weight)
            nn.init.zeros_(self.pred_proj.net[-1].bias)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Observations (B, T, 3, H, W) -> embeddings (B, T, D)."""
        B, T = obs.shape[:2]
        return self.projector(self.encoder(obs.flatten(0, 1))).view(B, T, -1)

    def predict(self, emb: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Teacher-forced next-step prediction.

        emb (B, T, D) and actions (B, T, A), where actions[:, i] is taken at frame i.
        Output[:, i] predicts the embedding of frame i+1 from frames <= i (T <= history).
        """
        x = self.predictor(emb, self.action_encoder(actions))
        out = self.pred_proj(x.flatten(0, 1)).view_as(x)
        return emb + out if self.cfg.pred_residual else out

    @torch.no_grad()
    def rollout(self, emb: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Open-loop imagination.

        emb: (B, C, D) embeddings of C >= 1 context frames.  actions: (B, C + N - 1, A), the
        actions taken at frames 0 .. C+N-2.  Returns (B, N, D): predicted embeddings of frames
        C .. C+N-1, each one predicted from the model's own previous predictions.
        """
        n = actions.size(1) - emb.size(1) + 1
        for _ in range(n):
            t = emb.size(1)
            ctx = emb[:, -self.cfg.history :]
            nxt = self.predict(ctx, actions[:, t - ctx.size(1) : t])[:, -1:]
            emb = torch.cat([emb, nxt], dim=1)
        return emb[:, -n:]


def jepa_loss(model: JEPA, sigreg: SIGReg, obs: torch.Tensor, actions: torch.Tensor, lambd: float):
    """LeWM objective on a batch of windows: obs (B, T+1, 3, H, W), actions (B, T, A)."""
    emb = model.encode(obs)
    pred = model.predict(emb[:, :-1], actions)
    pred_loss = (pred - emb[:, 1:]).square().mean()
    sig_loss = sigreg(emb.transpose(0, 1))  # Gaussianity across the batch, at every time step
    return pred_loss + lambd * sig_loss, {"pred": pred_loss.detach(), "sigreg": sig_loss.detach(), "emb": emb, "pred_emb": pred}


# --- visualisation-only decoder -----------------------------------------------------------------
class Decoder(nn.Module):
    """Embedding -> per-cell material logits (B, K, H, W).  A probe, not part of the world model."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        ch = cfg.dec_channels
        self.size0 = cfg.image_size // 2 ** len(ch)  # resolution of the first feature map
        self.fc = nn.Linear(cfg.embed_dim, ch[0] * self.size0**2)
        layers = []
        for c_in, c_out in zip(ch[:-1], ch[1:]):
            layers += [nn.GELU(), nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c_in, c_out, 3, padding=1)]
        # last doubling by sub-pixel convolution: cheaper than a conv at full resolution
        layers += [nn.GELU(), nn.Conv2d(ch[-1], 4 * cfg.n_classes, 3, padding=1), nn.PixelShuffle(2)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(len(z), -1, self.size0, self.size0)
        return self.net(x)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
