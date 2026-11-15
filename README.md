# sandwm

A small **action-conditioned world model for a falling-sand pixel world** (Noita / The Powder Toy
style), sized so that the whole loop -- generate data, train, evaluate -- runs on a laptop CPU.

* `sandwm.generate` -- a fast batched falling-sand simulator plus a scripted random "brush" agent
  that pours sand and water, draws walls and erases things.
* `sandwm.train` -- a LeWM-style JEPA world model: an encoder turns each frame into one embedding
  vector, and a small causal transformer predicts the next embedding from the action.
* `sandwm.evaluate` -- open-loop imagination, linear probes of physical quantities, and pictures.

## Quick start

```bash
uv sync                                                    # creates .venv with CPU PyTorch
uv run python -m sandwm.generate --out data/basic64        # 2000 + 200 episodes, ~35 s, 15 MB
uv run python -m sandwm.train --data data/basic64 --out runs/lewm
uv run python -m sandwm.evaluate --run runs/lewm
uv run pytest                                              # 28 tests, ~45 s
```

Things to look at afterwards:

| file | what it shows |
| --- | --- |
| `data/basic64/preview.png`, `preview.gif` | what the simulated world looks like |
| `runs/lewm/curves.png` | training curves, incl. the copy-last baseline and a collapse monitor |
| `runs/lewm/eval/rollout_sheet.png`, `rollout.gif` | truth vs. reconstruction vs. imagination |
| `runs/lewm/eval/rollout_curves.png`, `metrics.json` | imagination error against horizon, probe R^2 |

Every script has `--help`. All of them take `--device cuda` unchanged if a GPU is available.

## The world (`sim.py`, `materials.py`, `scenes.py`)

A 64x64 grid of material indices with a one-cell wall border. The default material set is
empty / wall / sand / water; `--materials extended` adds oil (floats on water) and smoke (rises).

The simulator is a **Margolus block cellular automaton**: the grid is cut into 2x2 blocks, every
block is updated independently by one small rule function (`block_rule`), and the block partition
shifts between sub-steps. That gives three properties that matter for world-model research:

* **Exact conservation.** Every move is a swap, so physics never creates or destroys matter; only
  the brush does. (Checked exhaustively over the whole rule table in the tests.)
* **Markov, time-homogeneous frames.** One frame is one full cycle of the four partition offsets,
  so the next frame depends only on (frame, action, noise) -- there is no hidden phase.
* **Speed.** The rule is tabulated once; stepping a batch of worlds is one table lookup. About
  5,600 frames/s on a laptop CPU.

Rules, in order, inside each block: (1) gravity -- denser over lighter swaps, free fall is
certain, sinking through a medium happens with probability `sink_prob`; (2) diagonal slide off
ledges for powders and liquids, mirrored upwards for gases; (3) sideways spread for liquids
resting on something, with probability `spread_prob`. `--deterministic` removes all randomness.
A free-falling grain drops 4 cells per frame.

Adding a material is one line in `materials.py` (a kind and a density); the physics follows.

One quirk to know about: dense falling masses thin out into stripes (see `preview.png`). This is
inherent to synchronous block updates -- a solid column cannot fall as a unit -- and is
deterministic, so it is something the world model simply has to learn.

**Actions** are `(x, y, tool)`: a 3x3 brush at a cell. Tool 0 does nothing, 1 erases, the others
paint a material into empty cells. `StrokePolicy` produces the training behaviour: idle periods
alternating with strokes in which one tool is held for 3-16 frames while the cursor drifts.

**Dataset format** (`data.py`): `frames (N, T+1, H, W) uint8` of material indices and
`actions (N, T, 3) int16`, in `.npz` shards with a `meta.json`. Storing materials rather than
pixels is lossless and tiny (143k frames = 15 MB), and gives both the RGB observation the model
sees and exact class targets for metrics.

## The model (`models.py`)

A re-implementation of the ideas in **LeWorldModel** (LeWM; Maes, Le Lidec, Scieur, LeCun,
Balestriero, 2026, [arXiv:2603.19312](https://arxiv.org/abs/2603.19312),
[code](https://github.com/lucas-maes/le-wm)), scaled down roughly 3x in parameters:

```
frame (RGB 64x64) --encoder--> projector (with BatchNorm) --> z_t  in R^192
(z_{t-2}, z_{t-1}, z_t) + actions --causal transformer, AdaLN-zero--> prediction of z_{t+1}

loss = MSE(prediction, z_{t+1}) + lambda * SIGReg(z)        (lambda = 0.1)
```

Trained end to end with no stop-gradient, no EMA target network and no reconstruction loss.
SIGReg is what prevents collapse: it pushes the embeddings of a batch towards an isotropic
Gaussian by testing random 1-D projections against N(0, 1).

Differences from the paper, and why:

| | LeWM | here | reason |
| --- | --- | --- | --- |
| encoder | ViT-Tiny, 224 px, patch 14 | 4-layer CNN, 64 px (`--encoder vit` available) | on this CPU the CNN trains at ~1,600 frames/s, a comparable ViT at ~160-450 |
| predictor | 6 layers, 16 heads | 4 layers, 8 heads | size |
| prediction target | next embedding | **change** in embedding, zero-initialised (`--no-residual` restores LeWM) | see results below |
| learning rate | 5e-5, 100 epochs | 5e-4, warm-up + cosine, a few thousand steps | tiny budget |
| decoder | trained after the fact | trained alongside, on detached embeddings | one command; it still cannot influence the world model |

The decoder deserves a sentence: a JEPA has no way to show you a picture. The decoder here is a
probe that maps an embedding to per-cell material logits. Its gradient is cut before the world
model, so it only reveals what the latent state already contains.
