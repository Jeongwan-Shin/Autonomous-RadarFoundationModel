#!/usr/bin/env python3
"""Pretrain the radar encoder on signals that cost nothing to compute.

Three objectives, none of which needs an annotator:

  moving      per point, whether the ego-compensated Doppler residual exceeds
              1 m/s. Physics, not labels.
  box_class   which labelled object a point sits inside. Sparse and highly
              variable between clips -- measured between 0.4% and 3.9% of points
              depending on the sample -- so it corrects rather than drives.
  ego         speed, acceleration and yaw rate regressed from the points alone.
              Forces the encoder to represent geometry and its own motion instead
              of memorising clutter.

Deliberately not included yet: distillation from the vision tower. That needs the
Qwen3-VL encoder in the loop and belongs to the next stage; adding it here would
confound the comparison this run exists to make.

Torchrun over the 5 devices with DDP. The encoder is 16.3 M parameters, so
sharding would be pointless -- the whole model replicates in well under a
gigabyte and the batch is what fills memory.

    torchrun --nproc_per_node=5 -m training.pretrain_radar --clips qa --epochs 3
    python -m training.pretrain_radar --clips qa --steps 30 --batch 4   # smoke
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from datatools import paths
from training.radar_data import RadarClipDataset, collate
from training.radar_encoder import RadarEncoder

# The losses live on different scales: cross-entropy on a 2-way problem,
# cross-entropy on a 13-way problem dominated by one class, and smooth L1 terms
# on quantities of order 10. Left unweighted, the ego term swamps the rest.
#
# `stats` carries weight 1.0 because it is the only term that reaches the tokens
# the language model is given. Until it existed the temporal stack and the
# readout were never trained at all -- the other three losses stop at
# `per_point` and `frame_tokens` -- so the encoder shipped a random temporal
# mixer, and a linear probe recovered a frame's detection count from its output
# at R^2 0.31 against 0.97 from the raw points.
LOSS_WEIGHTS = {"moving": 1.0, "box_class": 0.5, "ego": 0.1, "stats": 1.0,
                # The one objective that asks the emitted tokens where things
                # are. Weighted with `moving` because the detection tasks stand
                # on it -- a linear probe recovered the bearing histogram from
                # those tokens at only R^2 0.517 when nothing asked for it.
                "bearing": 1.0}


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


def setup():
    if is_distributed():
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank, dist.get_world_size()
    torch.cuda.set_device(0)
    return 0, 1


def log(rank, msg):
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clip_split(which, split, radars=("lrr1",)):
    """Clip ids per radar, as {short_name: [clip_id, ...]}.

    A rig carries either the SRR or the MRR+LRR pair, never both, so "which
    clips" is not separable from "which radar". Pretraining on `lrr1` alone --
    which is what this did until now -- leaves the encoder unaware of 51% of the
    clips the instruction model is trained on, and gives the routed experts a
    constant sensor to route on.
    """
    columns = ["split", "has_radar_extrinsics", "has_egomotion", "has_obstacle"]
    columns += [f"has_{r}" for r in radars]
    clips = __import__("pandas").read_parquet(
        os.path.join(paths.COMMON_DIR, "nvidia_clips.parquet"),
        columns=sorted(set(columns)))
    usable = clips[clips["has_radar_extrinsics"] & clips["has_egomotion"]
                   & clips["has_obstacle"] & (clips["split"] == split)]
    if which == "qa":
        import glob
        ids = {os.path.basename(f)[:-5] for f in glob.glob(
            os.path.join(paths.SPLIT_ROOT, "10_radar_vision_qa/qa/*.json"))}
        usable = usable[usable.index.isin(ids)]

    out, claimed = {}, set()
    for radar in radars:
        # First radar in the list wins a clip that has several, so no clip is
        # seen twice per epoch.
        picked = usable[usable[f"has_{radar}"].fillna(False)
                        & ~usable.index.isin(claimed)]
        out[radar] = list(picked.index)
        claimed.update(picked.index)
    return out


@torch.no_grad()
def evaluate(model, loader, device, max_batches=20):
    model.eval()
    totals = {"moving_correct": 0, "moving_total": 0,
              "box_correct": 0, "box_total": 0, "ego_mae": 0.0, "ego_n": 0,
              "stats_loss": 0.0, "stats_n": 0,
              "bearing_loss": 0.0, "bearing_n": 0}
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        points = batch["points"].to(device, torch.bfloat16)
        mask = batch["mask"].to(device)
        out = model.module.forward(points, mask) if hasattr(model, "module") \
            else model.forward(points, mask)
        per_point = out["per_point"]
        head = model.module if hasattr(model, "module") else model

        moving = batch["is_moving"].to(device)[mask]
        pred = head.head_moving(per_point)[mask].argmax(-1)
        totals["moving_correct"] += (pred == moving).sum().item()
        totals["moving_total"] += moving.numel()

        # Accuracy over the boxed points only: predicting "no box" everywhere
        # would otherwise score 97%.
        target = batch["box_class"].to(device)
        boxed = mask & (target >= 0)
        if boxed.any():
            pred_cls = head.head_class(per_point)[boxed].argmax(-1)
            totals["box_correct"] += (pred_cls == target[boxed]).sum().item()
            totals["box_total"] += int(boxed.sum().item())

        frame_repr = out["frame_tokens"].mean(dim=2)
        has_points = mask.any(dim=2)
        if has_points.any():
            predicted = head.head_ego(frame_repr)[has_points]
            truth = batch["ego_state"].to(device, torch.bfloat16)[has_points]
            totals["ego_mae"] += (predicted - truth).abs().mean().item()
            totals["ego_n"] += 1

        # How well the emitted tokens still describe their own frame. This is
        # the number to watch: it is the only metric computed from what the
        # language model is handed rather than from an intermediate.
        gpu_batch = {"points": points, "mask": mask,
                     "is_moving": batch["is_moving"].to(device),
                     "occupancy": batch["occupancy"].to(device)}
        stats = head._statistics_loss(out["tokens"], gpu_batch, has_points)
        totals["stats_loss"] += float(stats)
        totals["stats_n"] += 1
        # Binary cross-entropy against the occupancy grid. Predicting the
        # marginal occupancy of every cell costs 14.18 nats summed, 0.295 per
        # cell; anything well under that is the tokens carrying where the road
        # users are.
        totals["bearing_loss"] += float(
            head._bearing_loss(out["tokens"], gpu_batch, has_points))
        totals["bearing_n"] += 1
    model.train()
    return {
        "moving_acc": totals["moving_correct"] / max(totals["moving_total"], 1),
        "box_acc": totals["box_correct"] / max(totals["box_total"], 1),
        "box_n": totals["box_total"],
        "ego_mae": totals["ego_mae"] / max(totals["ego_n"], 1),
        "stats_loss": totals["stats_loss"] / max(totals["stats_n"], 1),
        "bearing_loss": totals["bearing_loss"] / max(totals["bearing_n"], 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clips", default="qa", choices=("qa", "all"))
    ap.add_argument("--radar", default="lrr1",
                    help="comma-separated; a clip is assigned to the first radar "
                         "in the list that it carries. 'lrr1,srr0' covers both "
                         "vehicle generations, which is what the instruction "
                         "model sees and what the routed experts split on")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--max-points", type=int, default=1024)
    # nuScenes' ARS408 gives 125 returns per scan against this rig's median
    # 815, so an encoder meant to read both has to stop treating density as a
    # signal. 80 is below the sparse end; 1024 is `max_points`, so the dense
    # end is untouched.
    ap.add_argument("--thin-low", type=int, default=80)
    ap.add_argument("--thin-high", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=0, help="stop early; for smoke tests")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--readout", default="frame", choices=("frame", "global", "polar"),
                    help="'frame' emits frame-aligned tokens plus a sum and a "
                         "max token per frame; 'global' is the original "
                         "256-query pool, kept only to reload old checkpoints")
    ap.add_argument("--experts", type=int, default=0,
                    help="routed experts in the per-point path; 0 keeps a dense "
                         "MLP. The three front radars differ by a factor of "
                         "three in return density, which is what these split on")
    ap.add_argument("--router-weight", type=float, default=0.01)
    ap.add_argument("--frame-queries", type=int, default=10,
                    help="resampled tokens per frame; the readout emits "
                         "n_frames x (this + 2) tokens in frame order")
    ap.add_argument("--out", default="/NHNHOME/workspace/checkpoints/radar_encoder")
    args = ap.parse_args(argv)

    rank, world = setup()
    device = torch.device("cuda", torch.cuda.current_device())

    radars = tuple(r.strip() for r in args.radar.split(",") if r.strip())
    train_ids = clip_split(args.clips, "train", radars)
    val_ids = clip_split(args.clips, "val", radars)
    log(rank, f"world size {world}, radars {'+'.join(radars)}")
    for radar in radars:
        log(rank, f"  {radar}: train {len(train_ids[radar]):,}  "
                  f"val {len(val_ids[radar]):,}")

    common = dict(n_frames=args.frames, max_points=args.max_points)
    # Thinning is training only. Validation has to be measured on the scans the
    # sensor actually produced, or the number moves with the augmentation
    # rather than with the model.
    train_common = dict(common, thin_to=(args.thin_low, args.thin_high))
    # One reader per radar, concatenated. RadarClipDataset keys its sensor id off
    # its own `radar`, so each half of the batch reports the sensor it came from
    # and the routed experts have something real to route on.
    parts = [RadarClipDataset(train_ids[r], radar=r, **train_common)
             for r in radars]
    val_parts = [RadarClipDataset(val_ids[r][:400 // len(radars)], radar=r,
                                  **common) for r in radars]
    train_ds = torch.utils.data.ConcatDataset(parts)
    val_ds = torch.utils.data.ConcatDataset(val_parts)

    sampler = DistributedSampler(train_ds) if world > 1 else None
    train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                          shuffle=sampler is None, num_workers=args.workers,
                          collate_fn=collate, drop_last=True, pin_memory=True,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=max(2, args.workers // 2), collate_fn=collate)

    model = RadarEncoder(dim=args.dim, n_frames=args.frames,
                         frame_queries=args.frame_queries,
                         readout=args.readout,
                         n_experts=args.experts).to(device).to(torch.bfloat16)
    log(rank, f"readout {args.readout}: {model.n_tokens} tokens to the language model")
    n_params = sum(p.numel() for p in model.parameters())
    log(rank, f"encoder {n_params/1e6:.1f} M parameters")
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[device.index])

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.05)
    total_steps = args.steps or args.epochs * len(train_dl)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=total_steps, pct_start=0.05)

    os.makedirs(args.out, exist_ok=True)
    step = 0
    started = time.monotonic()
    history = []
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in train_dl:
            gpu = {"points": batch["points"].to(device, torch.bfloat16,
                                                non_blocking=True),
                   "mask": batch["mask"].to(device, non_blocking=True),
                   "is_moving": batch["is_moving"].to(device, non_blocking=True),
                   "box_class": batch["box_class"].to(device, non_blocking=True),
                   "occupancy": batch["occupancy"].to(device, non_blocking=True),
                   "ego_state": batch["ego_state"].to(device, torch.bfloat16,
                                                      non_blocking=True)}
            inner = model.module if hasattr(model, "module") else model
            _, losses = inner.losses(gpu)
            total = sum(LOSS_WEIGHTS[k] * v for k, v in losses.items())
            balance = inner.router_loss()
            if balance is not None:
                total = total + args.router_weight * balance

            optimiser.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            step += 1

            if step % 20 == 0 or step == 1:
                rate = step * args.batch * world / (time.monotonic() - started)
                log(rank, f"step {step}/{total_steps}  total {total.item():.4f}  "
                          + "  ".join(f"{k} {v.item():.4f}" for k, v in losses.items())
                          + f"  {rate:.0f} clip/s")
            if args.steps and step >= args.steps:
                break
        if args.steps and step >= args.steps:
            break

        if rank == 0:
            metrics = evaluate(model, val_dl, device)
            history.append({"epoch": epoch, "step": step, **metrics})
            log(rank, f"epoch {epoch}: moving acc {metrics['moving_acc']*100:.1f}%  "
                      f"box acc {metrics['box_acc']*100:.1f}% (n={metrics['box_n']:,})  "
                      f"ego MAE {metrics['ego_mae']:.3f}  "
                      f"stats {metrics['stats_loss']:.4f}  "
                  f"bearing {metrics['bearing_loss']:.4f}")
            inner = model.module if hasattr(model, "module") else model
            torch.save({"model": inner.state_dict(), "args": vars(args),
                        "epoch": epoch, "history": history},
                       os.path.join(args.out, "encoder.pt"))

    if rank == 0:
        metrics = evaluate(model, val_dl, device)
        log(rank, f"final: moving acc {metrics['moving_acc']*100:.1f}%  "
                  f"box acc {metrics['box_acc']*100:.1f}%  "
                  f"ego MAE {metrics['ego_mae']:.3f}  "
                  f"stats {metrics['stats_loss']:.4f}  "
                  f"bearing {metrics['bearing_loss']:.4f}")
        history.append({"epoch": "final", "step": step, **metrics})
        with open(os.path.join(args.out, "history.json"), "w") as fh:
            json.dump(history, fh, indent=2)
        log(rank, f"saved to {args.out}")
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
