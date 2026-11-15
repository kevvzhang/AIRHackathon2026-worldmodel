"""Train the world model.

    uv run python -m sandwm.train --data data/basic64 --out runs/lewm

The defaults train the LeWM objective (latent prediction + SIGReg) plus a reconstruction term
(--recon-weight 1), with one latent step spanning 4 simulator frames.  `--recon-weight 0` is the
pure LeWM objective; `--lambd 0` drops SIGReg.

Writes to the run directory: ckpt.pt (model + decoder + config), metrics.jsonl, curves.png.
Ctrl+C stops early and still saves a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import EpisodeDataset
from .metrics import embedding_stats, seg_metrics, within_episode_share
from .models import PRESETS, JEPA, Decoder, ModelConfig, SIGReg, count_params, jepa_loss, parse_overrides


@torch.no_grad()
def validate(model: JEPA, decoder: Decoder, sigreg: SIGReg, obs, act, grid, spread_obs) -> dict:
    """Metrics on a fixed batch of validation windows: obs (B, T+1, ...), act (B, T, A), grid (B, T+1, H, W).

    spread_obs (E, M, 3, H, W) holds M frames spaced through each of E episodes.
    """
    was_training = model.training
    model.eval(), decoder.eval()
    emb = model.encode(obs)
    pred = model.predict(emb[:, :-1], act)
    shuffled = model.predict(emb[:, :-1], act[torch.randperm(len(act), generator=torch.Generator().manual_seed(0))])
    K = model.cfg.n_classes
    recon = decoder(emb.flatten(0, 1)).argmax(1)
    imagined = decoder(pred.flatten(0, 1)).argmax(1)
    std, rank = embedding_stats(emb)
    recon_acc, recon_iou = seg_metrics(recon, grid.flatten(0, 1), K)
    next_acc, next_iou = seg_metrics(imagined, grid[:, 1:].flatten(0, 1), K)
    out = {
        "val_pred": (pred - emb[:, 1:]).square().mean().item(),
        "val_copy": (emb[:, :-1] - emb[:, 1:]).square().mean().item(),  # baseline: "nothing changes"
        "val_pred_shuffled_actions": (shuffled - emb[:, 1:]).square().mean().item(),
        "val_sigreg": sigreg(emb.transpose(0, 1)).item(),
        "emb_std": std,
        "emb_rank": rank,
        "emb_within": within_episode_share(model.encode(spread_obs)),
        "recon_acc": recon_acc,
        "recon_miou": recon_iou,
        "next_acc": next_acc,
        "next_miou": next_iou,
    }
    model.train(was_training), decoder.train(was_training)
    return out


def plot_curves(metrics_path: Path, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    train = [r for r in rows if "pred" in r]
    val = [r for r in rows if "val_pred" in r]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot([r["step"] for r in train], [r["pred"] for r in train], label="train prediction", alpha=0.6)
    if val:
        s = [r["step"] for r in val]
        ax[0].plot(s, [r["val_pred"] for r in val], "o-", label="val prediction")
        ax[0].plot(s, [r["val_copy"] for r in val], "s--", label="val copy-last baseline")
        ax[0].plot(s, [r["val_pred_shuffled_actions"] for r in val], "^:", label="val, shuffled actions")
        ax[2].plot(s, [r["recon_miou"] for r in val], "o-", label="decode(encode(frame)) mIoU")
        ax[2].plot(s, [r["next_miou"] for r in val], "s-", label="decode(1-step prediction) mIoU")
        ax[2].plot(s, [r["emb_rank"] / max(r["emb_rank"] for r in val) for r in val], ":", label="effective rank (rel.)")
        if "emb_within" in val[0]:
            ax[2].plot(s, [r["emb_within"] for r in val], "^-", label="within-episode share of embedding variance")
            ax[2].axhline(val[0]["pixel_within"], color="gray", ls="--", lw=1, label="... the same share for raw pixels")
    ax[0].set(yscale="log", xlabel="step", title="latent prediction MSE")
    ax[1].plot([r["step"] for r in train], [r["sigreg"] for r in train], label="train SIGReg")
    ax[1].set(yscale="log", xlabel="step", title="SIGReg")
    ax[2].set(xlabel="step", title="what the latent holds (probe decoder)", ylim=(0, 1))
    for a in ax:
        a.legend(fontsize=8), a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64, help="windows per step (each is history+1 frames)")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--lambd", type=float, default=0.1, help="weight of the SIGReg term")
    p.add_argument("--num-proj", type=int, default=1024, help="random projections in SIGReg")
    p.add_argument("--frame-skip", type=int, default=4, help="raw frames per latent step; their actions are concatenated")
    p.add_argument("--preset", choices=sorted(PRESETS), default="cpu",
                   help="architecture size: 'cpu' = small CNN encoder, 'lewm' = ViT-Tiny + 6-layer predictor as in the paper")
    p.add_argument("--model", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override any ModelConfig field, e.g. --model encoder=vit vit_patch=4 pred_residual=false")
    p.add_argument("--no-amp", action="store_true", help="disable bfloat16 mixed precision on GPU")
    p.add_argument("--episodes-per-batch", type=int, default=0,
                   help="build each batch from this many episodes (0 = one episode per window); see EpisodeDataset.sample")
    p.add_argument("--recon-weight", type=float, default=1.0,
                   help="fraction of the decoder's gradient that reaches the encoder.  0 = the pure LeWM objective with the "
                        "decoder as a mere probe; in this world that encodes the scenery and little else (see README)")
    p.add_argument("--decoder-batch", type=int, default=64, help="frames per step used to train the probe decoder")
    p.add_argument("--max-episodes", type=int, default=None, help="use only the first N training episodes")
    p.add_argument("--max-minutes", type=float, default=0.0,
                   help="stop training after this many minutes and save, even if --steps is not reached (0 = no limit)")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = PyTorch default)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    train = EpisodeDataset(args.data, "train", args.max_episodes)
    val = EpisodeDataset(args.data, "val")
    cfg = ModelConfig(**{**PRESETS[args.preset], **parse_overrides(args.model), "image_size": train.size,
                         "action_dim": train.action_dim * args.frame_skip, "n_classes": len(train.materials)})
    model, decoder = JEPA(cfg).to(device), Decoder(cfg).to(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        model.amp = not args.no_amp and torch.cuda.is_bf16_supported(including_emulation=False)
    sigreg = SIGReg(args.num_proj).to(device)
    opt = torch.optim.AdamW([
        {"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": decoder.parameters(), "lr": 1e-3, "weight_decay": 0.0},
    ])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)) * (
        0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, args.steps))))))

    n_frames = cfg.history + 1
    vf, va = val.sample(256, n_frames, args.frame_skip, np.random.default_rng(1234))
    val_obs, val_act = val.observe(vf, device), val.encode_actions(va, device)
    val_grid = torch.from_numpy(vf).long().to(device)
    spread = val.spread(64, 8, np.random.default_rng(4321))  # 8 frames through each of 64 episodes
    spread_obs = val.observe(spread, device)
    pixel_within = within_episode_share(spread_obs.flatten(2))

    print(f"data: {len(train)} train / {len(val)} val episodes of {train.episode_len + 1} frames, {train.size}x{train.size}, "
          f"materials={train.materials.name}")
    print(f"model: encoder {count_params(model.encoder) + count_params(model.projector):,} | predictor "
          f"{count_params(model.predictor) + count_params(model.pred_proj) + count_params(model.action_encoder):,} | "
          f"probe decoder {count_params(decoder):,} params")
    where = f"{torch.cuda.get_device_name()}, bfloat16 {'on' if model.amp else 'off'}" if device.type == "cuda" else f"{torch.get_num_threads()} CPU threads"
    print(f"config: preset {args.preset}, encoder {cfg.encoder}, frame skip {args.frame_skip}, history {cfg.history}, "
          f"residual prediction {cfg.pred_residual}, episodes per batch {args.episodes_per_batch or args.batch}, "
          f"recon weight {args.recon_weight} | {where}", flush=True)

    def save(step: int) -> None:
        torch.save({"model": model.state_dict(), "decoder": decoder.state_dict(), "config": cfg.to_dict(), "step": step,
                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "data_meta": train.meta}, args.out / "ckpt.pt")

    metrics_file = (args.out / "metrics.jsonl").open("w")

    def log(row: dict) -> None:
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()

    def run_validation(step: int) -> None:
        v = validate(model, decoder, sigreg, val_obs, val_act, val_grid, spread_obs)
        v["pixel_within"] = pixel_within
        log({"step": step, **v})
        print(f"  val @ {step}: pred {v['val_pred']:.4f} | copy-last {v['val_copy']:.4f} | shuffled-actions "
              f"{v['val_pred_shuffled_actions']:.4f} | rank {v['emb_rank']:.0f}/{cfg.embed_dim} | within-episode "
              f"{v['emb_within']:.3f} (pixels {pixel_within:.3f}) | "
              f"recon acc {v['recon_acc']:.3f} mIoU {v['recon_miou']:.3f} | 1-step acc {v['next_acc']:.3f} mIoU {v['next_miou']:.3f}",
              flush=True)

    model.train(), decoder.train()
    t_start, step = time.perf_counter(), 0
    try:
        for step in range(1, args.steps + 1):
            frames, actions = train.sample(args.batch, n_frames, args.frame_skip, rng, args.episodes_per_batch)
            obs, act = train.observe(frames, device), train.encode_actions(actions, device)
            loss, out = jepa_loss(model, sigreg, obs, act, args.lambd)

            # Decoder.  With --recon-weight 0 it sees detached embeddings and is a pure probe: its gradient
            # never reaches the world model.  Otherwise that fraction of its gradient flows into the encoder.
            pick = torch.randperm(args.batch * n_frames, device=device)[: args.decoder_batch]
            target = torch.from_numpy(frames).to(device).flatten(0, 1)[pick].long()
            emb = out["emb"]
            dec_in = emb.detach() + args.recon_weight * (emb - emb.detach())
            dec_loss = F.cross_entropy(decoder(dec_in.flatten(0, 1)[pick]), target)

            opt.zero_grad(set_to_none=True)
            (loss + dec_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            if step % args.log_every == 0 or step == 1:
                per_step = (time.perf_counter() - t_start) / step
                row = {"step": step, "pred": out["pred"].item(), "sigreg": out["sigreg"].item(), "dec": dec_loss.item(),
                       "lr": sched.get_last_lr()[0], "sec_per_step": per_step}
                log(row)
                print(f"step {step:5d}/{args.steps} | pred {row['pred']:.4f} sigreg {row['sigreg']:.3f} dec {row['dec']:.3f} | "
                      f"{per_step:.2f} s/step, eta {(args.steps - step) * per_step / 60:.1f} min", flush=True)
            out_of_time = args.max_minutes > 0 and time.perf_counter() - t_start > 60 * args.max_minutes
            if step % args.eval_every == 0 or step == args.steps or out_of_time:
                run_validation(step)
                save(step)
            if out_of_time and step < args.steps:
                print(f"stopping at step {step} of {args.steps}: the limit of {args.max_minutes:g} minutes is reached", flush=True)
                break
    except KeyboardInterrupt:
        print("interrupted -- saving checkpoint")
        save(step)
    metrics_file.close()
    plot_curves(args.out / "metrics.jsonl", args.out / "curves.png")
    print(f"done: {step} steps in {(time.perf_counter() - t_start) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
