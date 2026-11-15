# sandwm

A small **action-conditioned world model for a falling-sand pixel world** (Noita / The Powder Toy
style). The whole loop -- generate data, train, evaluate -- runs on a laptop CPU in about twenty
minutes, and the same code runs on cloud GPUs through [Modal](https://modal.com) for anything bigger.

* `sandwm.generate` -- a fast batched falling-sand simulator plus a scripted random "brush" agent
  that pours sand and water, draws walls and erases things.
* `sandwm.train` -- a LeWM-style JEPA world model: an encoder turns each frame into one embedding
  vector, and a small causal transformer predicts the next embedding from the action.
* `sandwm.evaluate` -- open-loop imagination, linear probes of physical quantities, and pictures.
* `cloud.py` -- the same three steps on Modal GPUs, with datasets and runs kept on a Modal volume.

## Quick start (laptop)

```bash
uv sync                                                    # creates .venv (CPU PyTorch on Windows)
uv run python -m sandwm.generate --out data/basic64        # 2000 + 200 episodes, ~35 s, 15 MB
uv run python -m sandwm.train --data data/basic64 --out runs/lewm     # ~20 min for 3000 steps
uv run python -m sandwm.evaluate --run runs/lewm
uv run pytest                                              # 39 tests, under a minute
```

Things to look at afterwards:

| file | what it shows |
| --- | --- |
| `data/basic64/preview.png`, `preview.gif` | what the simulated world looks like |
| `runs/lewm/curves.png` | training curves, incl. the copy-last baseline and a collapse monitor |
| `runs/lewm/eval/rollout_sheet.png`, `rollout.gif` | truth vs. reconstruction vs. imagination |
| `runs/lewm/eval/rollout_curves.png`, `metrics.json` | imagination error against horizon, probe R^2 |

Every script has `--help`.

## Playing with a trained model (`sandwm.play`)

```bash
uv run python -m sandwm.play --run runs/recon-k2        # opens http://127.0.0.1:8765/
```

A local page (the server only listens on 127.0.0.1) with three views of the same world:

* **Reality** -- the simulator.
* **Imagination** -- the world model dreaming: at a *sync* its encoder sees the true frames once,
  after that the predictor runs on its own latent states plus your brush actions. Shown through
  the decoder.
* **Reconstruction** -- decode(encode(reality)), the most one latent vector can show -- or
  **Mistakes**, the cells where imagination and reality differ.

Your brush drives both worlds: left-drag paints with the selected tool (keys 1-4), right-drag
erases. `Space` pauses, `N` steps once, `S` syncs, `R` / `E` start a new random scene or an empty
box, `B` hides the brush from the model (the no-action ablation, live). Auto-sync re-shows reality
every N model steps (every step = one-step prediction). A chart under the panels tracks the mIoU of
the imagination against reality, next to the "nothing changes since the sync" baseline and the
ceiling, so you can watch the dream drift. The model menu switches between any `runs/*/ckpt.pt`
without resetting the world.

The default is one CPU thread: at one frame at a time the small CNN model runs fastest that way
(about 35 ms per step including the decoder, versus about 47 ms on four threads).

## Training on Modal GPUs (`cloud.py`)

One-time setup: `uv run modal setup` (stores a token in `~/.modal.toml`).

```bash
# simulate a dataset in the cloud (CPU only; 10,000 episodes take ~100 s)
uv run modal run cloud.py::generate --name basic64-10k --args "--episodes 10000 --val-episodes 500"

# train one model on a GPU, then evaluate it; --args goes to `python -m sandwm.train`
uv run modal run --detach cloud.py::train --name vit-k2 --data basic64-10k \
    --args "--preset lewm --frame-skip 2 --steps 20000 --batch 128"

# train several models side by side, one GPU each, and print a comparison table
uv run modal run --detach cloud.py::sweep --data basic64-10k \
    --spec "cnn: --steps 20000 --batch 128; vit: --preset lewm --steps 20000 --batch 128"

# copy a finished run (curves, pictures, metrics, checkpoint) back into ./runs/
uv run modal volume get sandwm runs/vit-k2 runs/
```

* Everything lives on a Modal volume named `sandwm`, as `data/<dataset>/` and `runs/<run>/`
  (`uv run modal volume ls sandwm runs`).
* `--detach` keeps a job running if the terminal closes.
* `SANDWM_GPU` is an order of preference, default `L4,A10G,T4`: Modal takes the first type it can
  schedule. (A single fixed type sometimes waits many minutes for capacity.) `SANDWM_MAX_GPUS`
  caps how many GPUs a sweep holds at once (default 4).
* The image pins the same package versions as the local environment, so a dataset generated in
  the cloud with a given seed is identical to the one generated locally.
* Speed on an A10G at batch 128: the small CNN model about 0.02 s/step, the paper-sized ViT model
  about 0.10 s/step -- versus 0.4 s/step for the small model at batch 64 on the laptop CPU.

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
[code](https://github.com/lucas-maes/le-wm)):

```
frame (RGB 64x64) --encoder--> projector (with BatchNorm) --> z_t  in R^192
(z_{t-2}, z_{t-1}, z_t) + actions --causal transformer, AdaLN-zero--> prediction of z_{t+1}

loss = MSE(prediction, z_{t+1}) + lambda * SIGReg(z)        (lambda = 0.1)
```

Trained end to end with no stop-gradient, no EMA target network and no reconstruction loss.
SIGReg is what prevents collapse: it pushes the embeddings of a batch towards an isotropic
Gaussian by testing random 1-D projections against N(0, 1).

Two sizes are built in, and any field of `ModelConfig` can be overridden from the command line:

| `--preset` | encoder | predictor | parameters | intended for |
| --- | --- | --- | --- | --- |
| `cpu` (default) | 4-layer CNN | 4 layers, 8 heads | 6M | laptop |
| `lewm` | ViT-Tiny: 12 layers, 3 heads, width 192, patch 8 | 6 layers, 16 heads | 18M | GPU; the paper's sizes |

```bash
--model vit_patch=4 pred_depth=8 pred_residual=false cnn_channels=48,96,192,192   # for example
```

`--frame-skip k` makes one latent step span k simulator frames; the k actions are concatenated
(the paper uses 5). The dataset stores every frame, so this is a training-time choice.

Differences from the paper, and why:

| | LeWM | here | reason |
| --- | --- | --- | --- |
| image size | 224 px, patch 14 (256 tokens) | 64 px, patch 8 (64 tokens) | one pixel is one cell of the world |
| prediction target | next embedding | **change** in embedding, zero-initialised (`--model pred_residual=false` restores LeWM) | see results |
| learning rate | 5e-5, 100 epochs | 3e-4 to 5e-4, warm-up + cosine | shorter schedules |
| mixed precision | bf16 everywhere | bf16 in the encoder trunk only | consecutive embeddings differ by a few percent, about the resolution of bf16 |
| decoder | trained after the fact | trained alongside, on detached embeddings | one command; it still cannot influence the world model |

The decoder deserves a sentence: a JEPA has no way to show you a picture. The decoder here is a
probe that maps an embedding to per-cell material logits. Its gradient is cut before the world
model, so it only reveals what the latent state already contains.

## How to read the evaluation

`sandwm.evaluate` encodes 3 context frames and then rolls the predictor forward on its own
outputs, given only the actions. Each number is reported next to the references that give it
meaning:

* **frozen** -- the latent state never changes after the context. This is the baseline a world
  model has to beat, and in a world where 99% of cells are unchanged from one frame to the next
  it is a strong one.
* **ceiling** -- encode and decode the *true* future frame. This is the most the latent and the
  probe decoder could possibly show; the gap between ceiling and 1 is information the single
  embedding vector does not carry.
* **pixel** -- repeat the last true frame. Not limited by the latent at all; shown for scale.
* **no-action** -- the model's imagination with the brush hidden, to check that actions are used.

"Accuracy on changed cells" restricts the comparison to cells whose material differs from the
last context frame, i.e. to where the dynamics actually are.
