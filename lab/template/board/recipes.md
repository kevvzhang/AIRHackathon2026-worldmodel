# Recipes and runs

Kept by the train agent: the reference recipe, one line for every run tried against it, and
what is planned.

## Reference recipe (`recon-k4`, the first champion)

    uv run python -m lab train <name> --data basic64-10k --note "..." -- --steps 20000 --batch 128 --lr 5e-4 --warmup 500 --frame-skip 4 --recon-weight 1 --eval-every 1000 --log-every 200

Small CNN encoder into one 192-number vector per frame, 4-layer causal transformer predictor,
latent prediction + 0.1 x SIGReg + reconstruction, seed 0. About 6 minutes on an A10G. It was
trained before the lab existed, so it has no line in the ledger; its scorecard on the benchmark
was computed on the laptop and is in the shared runs folder.

## Runs

| run | changed against the reference | result (checks that moved) | kept? |
|-----|-------------------------------|----------------------------|-------|
| `recon-k4` | nothing: it is the reference | gates 1 of 10; see `gaps.md` | champion |

## Planned

(train agent: your plan for the remaining budget goes here)
