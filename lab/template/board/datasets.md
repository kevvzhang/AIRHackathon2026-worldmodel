# Datasets

Kept by the data agent. Every dataset on the Modal volume `sandwm` (`data/<name>`), newest last.
A dataset is never changed after it is listed here; a correction is a new version.

| name | world | episodes (train / val) | seed | made by | for |
|------|-------|------------------------|------|---------|-----|
| `basic64` | basic, 64x64, 64 steps | 2,000 / 200 | 0 | before the lab | laptop-sized training set |
| `basic64-10k` | basic, 64x64, 64 steps | 10,000 / 500 | 0 | before the lab | the reference training set (`recon-k4`) |
| `{{BENCH}}` | basic, 64x64, 64 steps | 0 / 256 | 9001 | setup | **benchmark**, frozen: never train on it |

Notes

- All three use the default generator: random scene, scripted random brush (`StrokePolicy`),
  stochastic physics (sink 0.5, slide 1.0, spread 0.9), 3x3 brush. Commands are in `README.md`.
- `basic64` and `basic64-10k` share seed 0, so the 2,000 training episodes of `basic64` are the
  first 2,000 of `basic64-10k`. They are not independent. Give every new dataset its own seed.
- A local copy of `basic64` is at `{{HOME}}\data\basic64` (read only) for smoke tests, and the
  benchmark is at `{{DATA}}\{{BENCH}}`.
