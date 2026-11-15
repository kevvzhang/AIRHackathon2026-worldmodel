# sandwm

A small **action-conditioned world model for a falling-sand pixel world** (Noita / The Powder Toy
style). The whole loop -- generate data, train, evaluate -- runs on a laptop CPU in about twenty
minutes, and the same code runs on cloud GPUs through [Modal](https://modal.com) for anything bigger.

* `sandwm.generate` -- a fast batched falling-sand simulator plus a scripted random "brush" agent
  that pours sand and water, draws walls and erases things.
* `sandwm.train` -- a LeWM-style latent world model: an encoder turns each frame into one embedding
  vector, and a small causal transformer predicts the next embedding from the action.
* `sandwm.evaluate` -- open-loop imagination, linear probes of physical quantities, and pictures.
* `cloud.py` -- the same three steps on Modal GPUs, with datasets and runs kept on a Modal volume.
* `sandwm.scorecard` -- named checks of what a model gets right (walls kept, mass conserved, ...) on a
  frozen benchmark, so that any two runs can be compared check by check.
* `lab/` -- three Claude Code agents (eval, data, train) that improve the model in a loop: see
  [lab/README.md](lab/README.md).

## Quick start (laptop)

```bash
uv sync                                                    # creates .venv (CPU PyTorch on Windows)
uv run python -m sandwm.generate --out data/basic64        # 2000 + 200 episodes, ~35 s, 15 MB
uv run python -m sandwm.train --data data/basic64 --out runs/lewm     # ~20 min for 3000 steps
uv run python -m sandwm.evaluate --run runs/lewm
uv run pytest                                              # 77 tests
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
uv run python -m sandwm.play --run runs/recon-k2        # opens http://127.0.0.1:8765/ (or the next free port)
uv run python -m sandwm.play --run .lab/runs/r0003-grid8-15k   # a run from the agent lab
```

A local page (the server only listens on 127.0.0.1) with three views of the same world:

* **Reality** -- the simulator.
* **Imagination** -- the world model dreaming: at a *sync* its encoder sees the true frames once,
  after that the predictor runs on its own latent states plus your brush actions. Shown through
  the decoder.
* **Reconstruction** -- decode(encode(reality)), the most one latent vector can show -- or
  **Mistakes**, the cells where imagination and reality differ.

Your brush drives both worlds: left-drag paints with the selected tool (number keys), right-drag
erases. `Space` pauses, `N` steps once, `S` syncs, `R` / `E` start a new random scene or an empty
box, `B` hides the brush from the model (the no-action ablation, live). Auto-sync re-shows reality
every N model steps (every step = one-step prediction). A chart under the panels tracks the mIoU of
the imagination against reality, next to the "nothing changes since the sync" baseline and the
ceiling, so you can watch the dream drift. The model menu switches between any `runs/*/ckpt.pt`
and `.lab/runs/*/ckpt.pt` (the agent lab's runs, its champion marked; `--runs-dir`, repeatable,
picks other folders) without resetting the world, and picks up new runs as they arrive.

Every run is played with the code it was trained with, simulator included. The lab keeps adding
and removing model options and changing the simulator, so no single version of the code loads all
of its runs. It tags each run's commit as `run/<name>`: for such a run the player copies the
`sandwm` package at that commit out of git (once, into the system temp folder, `sandwm-play/`)
and runs the simulator and the model in a separate process on that copy
(`sandwm/playsession.py`, which is added to the copy). Runs without a tag use this checkout's
code. The info line names the code (`code fc963d3` or `code this checkout`). Switching to a run on
other code starts a new process (a few seconds) and carries the world over; between runs on the
same code it takes a fraction of a second. New random scenes are drawn with the run's training
terrain (its data's `terrain` options, as far as that code has them).

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
* `--bench <dataset>` on `train`, `sweep` or `reevaluate` also writes a scorecard on that benchmark
  dataset (`python -m sandwm.scorecard`).
* Once the agent lab is set up (`python -m lab setup`), every GPU job started through `cloud.py` is
  booked against the lab's budget first and refused if it does not fit: see `lab/README.md`.
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

loss = MSE(prediction, z_{t+1}) + lambda * SIGReg(z)        (lambda = 0.1)      <- LeWM
     + recon_weight * cross-entropy(decoder(z_t), frame_t)                        <- added here, on by default
```

The first line is LeWM: trained end to end with no stop-gradient and no EMA target network.
SIGReg is what prevents collapse there: it pushes the embeddings of a batch towards an isotropic
Gaussian by testing random 1-D projections against N(0, 1).

The second line is not in LeWM. It is on by default (`--recon-weight 1`) because in this world the
first line alone learns to encode the scenery and little else -- the main finding so far, see
[Results](#results). `--recon-weight 0` gives the pure LeWM objective, with the decoder reduced to
a probe trained on detached embeddings; `--lambd 0` drops SIGReg instead.

Two sizes are built in, and any field of `ModelConfig` can be overridden from the command line:

| `--preset` | encoder | predictor | parameters | intended for |
| --- | --- | --- | --- | --- |
| `cpu` (default) | 4-layer CNN | 4 layers, 8 heads | 6M | laptop |
| `lewm` | ViT-Tiny: 12 layers, 3 heads, width 192, patch 8 | 6 layers, 16 heads | 18M | GPU; the paper's sizes |

```bash
--model vit_patch=4 pred_depth=8 pred_residual=false cnn_channels=48,96,192,192   # for example
```

`--frame-skip k` makes one latent step span k simulator frames; the k actions are concatenated
(default 4; the paper uses 5). The dataset stores every frame, so this is a training-time choice.

Two further switches exist for experiments: `--episodes-per-batch E` builds each batch from only E
episodes, and `--model pred_residual=false` makes the predictor output the next embedding directly
rather than the change in embedding.

Differences from the paper, and why:

| | LeWM | here | reason |
| --- | --- | --- | --- |
| image size | 224 px, patch 14 (256 tokens) | 64 px, patch 8 (64 tokens) | one pixel is one cell of the world |
| objective | prediction + SIGReg | prediction + SIGReg + reconstruction | the pure objective takes a shortcut in this world ([Results](#results)) |
| prediction target | next embedding | **change** in embedding, zero-initialised | starts at the copy-last baseline; matters at small budgets only ([Results](#results)) |
| learning rate | 5e-5, 100 epochs | 5e-4 (CNN), 1e-4 (ViT), warm-up + cosine | shorter schedules |
| mixed precision | bf16 everywhere | bf16 in the encoder trunk only | consecutive embeddings differ by a few percent, about the resolution of bf16 |
| decoder | trained after the fact, for pictures only | trained alongside; part of the loss unless `--recon-weight 0` | see above |

The decoder maps an embedding to per-cell material logits. A JEPA has no other way to show you a
picture, so it is also what every "decoded" number and image in the evaluation goes through.

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

`uv run python -m sandwm.compare runs/a runs/b` prints the key numbers of several evaluated runs
side by side, at a fixed number of simulator frames ahead so that different frame skips compare.
It also reports the **within-episode share**: the fraction of the embedding's variance that lies
within episodes rather than between them, next to the same fraction for raw pixels. An embedding
far below the pixel value mostly says *which episode* a frame is from, not what is happening in it.

## Results

Status as of 2026-11-14. Every row is a single run with a single seed, so differences of a few
hundredths are not established. Unless stated: 10,000 training episodes (650,000 frames), small
CNN model (`--preset cpu`), 20,000 steps at batch 128 (about 6 minutes on an A10G), evaluated on
256 held-out windows, open-loop imagination judged 16 simulator frames ahead. "Changed cells" is
accuracy on the ~5% of cells that differ from the last context frame, as model / frozen / ceiling.

### 1. The pure LeWM objective encodes the scenery, not the dynamics

| encoder (`--recon-weight 0`) | frame skip | within-episode share | decode mIoU ceiling | changed cells | probe R^2 amount / position |
| --- | --- | --- | --- | --- | --- |
| CNN | 1 | 0.07 | 0.64 | 0.43 / 0.35 / 0.55 | 0.63 / 0.40 |
| CNN | 2 | 0.03 | 0.58 | 0.43 / 0.39 / 0.49 | 0.55 / 0.28 |
| CNN | 4 | 0.01 | 0.49 | 0.46 / 0.45 / 0.48 | 0.45 / 0.19 |
| ViT (`--preset lewm`), lr 3e-4 | 2 | 0.17 | 0.49 | 0.40 / 0.38 / 0.45 | 0.84 / 0.10 |
| ViT, lr 1e-4 | 2 | 0.15 | 0.51 | 0.42 / 0.36 / 0.45 | 0.88 / 0.21 |
| ViT, lr 3e-4, predictor as in the paper | 2 | 0.14 | 0.49 | 0.40 / 0.36 / 0.41 | 0.83 / 0.06 |

For raw pixels the within-episode share is 0.29. The embeddings stay high-rank and SIGReg is
satisfied, but the share falls throughout training (0.15 at step 2,000, 0.07 at 20,000), the
frozen-latent error shrinks 4-10x, and at frame skip 4 the decode ceiling itself gets worse with
training (0.55 to 0.48). The likely reason: every episode has its own random terrain. Which scene
a frame belongs to differs across a batch, so it can satisfy SIGReg, and is constant in time, so
it costs nothing to predict. LeWM's own benchmarks have one fixed background and do not offer
this shortcut.

### 2. A reconstruction term fixes it

| variant (CNN, frame skip 1) | within-episode share | decode mIoU ceiling | changed cells | probe R^2 amount / position |
| --- | --- | --- | --- | --- |
| pure objective | 0.07 | 0.64 | 0.43 / 0.35 / 0.55 | 0.63 / 0.40 |
| batches built from 8 episodes (`--episodes-per-batch 8`) | 0.81 | 0.44 | 0.46 / 0.26 / 0.52 | 0.24 / 0.22 |
| batches built from 32 episodes | 0.66 | 0.55 | 0.42 / 0.20 / 0.62 | 0.40 / 0.31 |
| reconstruction, weight 0.1 | 0.15 | 0.77 | 0.52 / 0.24 / 0.72 | 0.86 / 0.68 |
| **reconstruction, weight 1** | 0.21 | 0.81 | 0.54 / 0.16 / 0.80 | 0.91 / 0.72 |
| reconstruction + 8 episodes per batch | 0.75 | 0.74 | 0.49 / 0.18 / 0.74 | 0.78 / 0.64 |
| reconstruction, SIGReg off (`--lambd 0`) | 0.20 | 0.84 | 0.56 / 0.14 / 0.82 | 0.92 / 0.73 |

Grouped batches do force the dynamics in, but at the cost of the static layout. With the
reconstruction term, SIGReg no longer contributes anything measurable.

### 3. With reconstruction: frame skip, encoder, width

| variant (`--recon-weight 1`) | decode mIoU ceiling | changed cells | imagined mIoU vs. repeating the last true frame | train time |
| --- | --- | --- | --- | --- |
| CNN, frame skip 1 | 0.81 | 0.54 / 0.16 / 0.80 | 0.75 vs 0.78 | 6 min |
| CNN, frame skip 2 | 0.81 | 0.59 / 0.17 / 0.78 | 0.76 vs 0.77 | 6 min |
| **CNN, frame skip 4 (the default)** | 0.80 | 0.66 / 0.20 / 0.78 | 0.75 vs 0.71 | 6 min |
| CNN, frame skip 1, 512-d embedding | 0.81 | 0.57 / 0.17 / 0.80 | 0.75 vs 0.78 | 7 min |
| ViT (`--preset lewm`), frame skip 1, lr 1e-4 | 0.74 | 0.55 / 0.23 / 0.70 | 0.70 vs 0.78 | 32 min |

At frame skip 4 the imagination stays ahead of "nothing changes" as the horizon grows: mIoU
0.72 vs 0.64 at 32 frames and 0.68 vs 0.58 at 56 frames. The paper-sized ViT did not beat the
small CNN here, with or without reconstruction, and a wider embedding did not raise the ceiling.

The residual predictor matters only for short runs: after 1,000 CPU steps it was at 0.96x the
copy-last error where the paper's direct form was at 2.45x, but after 20,000 steps the two were
indistinguishable.

### 4. On the laptop

2,000 training episodes (130,000 frames), 3,000 steps at batch 64, `--recon-weight 1
--frame-skip 1`: 29 minutes on an i7-1265U with the CPU shared (about 20 when idle). Decode mIoU
ceiling 0.72, changed cells 0.46 / 0.23 / 0.70, probes 0.60-0.94. The current defaults (frame
skip 4) have been verified in the cloud but not yet run to completion on the laptop.

### 5. Known limits

* One vector per frame caps what can be shown at a decode mIoU of about 0.8: thin walls, single
  grains and falling streams are blurred or lost, and a wider vector did not help.
* The predictor is deterministic, so where the physics is random (water surfaces, sinking sand)
  it predicts an average, and imagined frames lose detail as they roll forward.
* Trained only on the scripted random brush; no agent, no task, 64x64, four materials.
* The decoder is small (0.35M parameters) and every decoded number goes through it.

## The agent lab (`lab/`)

`lab/` sets up three Claude Code agents that keep improving the model on their own: one judges
every run and keeps the list of what is wrong, one works on the simulator and the data, one on
the model and the training runs. They share a board of plain files, a GPU budget enforced in
code, and a scorecard that turns every discovered failure into a regression check.
[lab/README.md](lab/README.md) says how to start, watch, steer and stop it.

## Not started

Ideas, none of them begun: a grid of latent vectors with a local predictor instead of one vector;
a pixel-space local baseline (a small convolutional net predicting the next frame) as a yardstick;
a stochastic predictor; planning towards a goal frame in latent space; reactions (fire, wood,
acid), larger grids and an embodied agent.
