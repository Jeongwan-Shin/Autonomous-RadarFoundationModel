#!/usr/bin/env python3
"""Does the radar encoder's output still carry how many returns there were?

Shuffling the radar between clips costs the aligned 8B model only 0.03 nll on
`radar_probe`, whose answer *is* the detection count. Either the language model
ignores a signal it was given, or the signal never survived the encoder. This
decides which, before any capacity is added to fix the wrong one.

The suspicion is structural rather than a matter of capacity. Both pooling steps
in `RadarEncoder` are softmax cross-attention against learned queries, and a
softmax's weights sum to one, so its output is a weighted *average* of the point
features and is very nearly invariant to how many points there were. Counting is
the textbook case that mean-style aggregation cannot do and sum aggregation can.

Four representations are probed against the same targets with the same ridge
regression, so the comparison isolates the aggregation and nothing else:

  global      the 256 tokens the language model actually receives, mean-pooled
  frame       the 48 per-frame queries, mean-pooled -- one pooling step earlier
  sumpool     per-point features summed under the mask. Not part of the model;
              this is the control that says whether the information exists at
              the point level and is destroyed downstream
  count       log(1 + n) alone, an oracle upper bound on a count-only feature

If `sumpool` predicts and `global` does not, the encoder is discarding
cardinality by construction and no number of extra layers or experts downstream
can recover it.

    python -m training.probe_radar_tokens --clips 600
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from datatools import paths
from training.radar_data import RadarClipDataset
from training.radar_encoder import (RadarEncoder, encoder_kwargs,
                                     load_encoder_state)

TARGETS = ("lrr1_n_points", "lrr1_n_moving", "lrr1_max_rcs")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ridge_r2(features, target, train_frac=0.6, alpha=1.0):
    """R^2 on held-out rows, against predicting the training mean.

    Solved in the dual, because the honest form of this probe has far more
    features than rows: the 256 tokens the language model receives are 98,304
    numbers and there are only a few hundred clips. Kernel ridge on the Gram
    matrix costs O(rows^2) instead and fits the same linear model.

    Standardised first: the blocks have very different scales -- a summed point
    cloud against a mean-pooled token -- and one penalty across all of them would
    otherwise punish whichever happens to be larger.
    """
    n = len(target)
    split = max(int(n * train_frac), 2)
    x_train, x_test = features[:split], features[split:]
    y_train, y_test = target[:split], target[split:]
    if len(x_test) < 2:
        return float("nan")

    mu, sigma = x_train.mean(0), x_train.std(0) + 1e-6
    x_train = ((x_train - mu) / sigma).astype(np.float64)
    x_test = ((x_test - mu) / sigma).astype(np.float64)

    offset = y_train.mean()
    gram = x_train @ x_train.T
    dual = np.linalg.solve(gram + alpha * np.eye(len(gram)), y_train - offset)
    prediction = (x_test @ x_train.T) @ dual + offset

    residual = ((y_test - prediction) ** 2).sum()
    total = ((y_test - offset) ** 2).sum()
    return 1.0 - residual / max(total, 1e-9)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clips", type=int, default=600)
    ap.add_argument("--split", default="val")
    ap.add_argument("--radar", default="lrr1")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--checkpoint",
                    default="/NHNHOME/workspace/checkpoints/radar_encoder_all/encoder.pt")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)

    clips = pd.read_parquet(os.path.join(paths.COMMON_DIR, "nvidia_clips.parquet"))
    usable = clips[clips[f"has_{args.radar}"].fillna(False)
                   & clips["has_radar_extrinsics"].fillna(False)
                   & clips["has_egomotion"].fillna(False)
                   & (clips["split"] == args.split)]
    clip_ids = list(usable.index[: args.clips])
    log(f"{len(clip_ids)} clips from {args.split}")

    features = pd.read_parquet(
        os.path.join(paths.COMMON_DIR, "scene_features_all_clips.parquet"),
        columns=["clip_id", "frame"] + list(TARGETS))
    features = features[features["clip_id"].isin(set(clip_ids))]
    features = features.set_index(["clip_id", "frame"])

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu")
    saved = state.get("args", {})
    encoder = RadarEncoder(**{"dim": 384, "n_frames": 20,
                              **encoder_kwargs(saved)})
    load_encoder_state(encoder, state["model"])
    encoder = encoder.to(device).to(torch.bfloat16).eval()
    log(f"encoder restored from {args.checkpoint} "
        f"(readout {encoder.readout}, {encoder.n_tokens} tokens)")

    dataset = RadarClipDataset(clip_ids, radar=args.radar, n_frames=20,
                               with_boxes=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, num_workers=args.workers,
        collate_fn=lambda b: b)

    # Probed per frame, one model per frame index. Mean-pooling the 256 global
    # tokens instead would have thrown away the frame structure before asking
    # whether the frame structure is there -- and every radar question names a
    # frame. Here each frame gets its own probe over the whole token block, so a
    # failure means the tokens really do not resolve that frame.
    per_frame = {f: {"emitted": [], "frame": [], "sumpool": [], "count": [],
                     "targets": {name: [] for name in TARGETS}}
                 for f in range(20)}
    seen = 0
    for batch in loader:
        points = torch.stack([b["points"] for b in batch]).to(device, torch.bfloat16)
        mask = torch.stack([b["mask"] for b in batch]).to(device)
        with torch.no_grad():
            out = encoder(points, mask)
            per_point, _, _ = encoder.encode_points(points, mask)

        # The language model receives all 256 tokens, so the probe gets all 256.
        glob = out["tokens"].float().flatten(1).cpu().numpy().astype(np.float32)
        frame_tokens = out["frame_tokens"].float().flatten(2).cpu().numpy()
        weights = mask.unsqueeze(-1).to(per_point.dtype)
        summed = (per_point * weights).sum(dim=2).float().cpu().numpy()
        counts = mask.sum(dim=2).float().cpu().numpy()

        for i, item in enumerate(batch):
            clip_id = item["clip_id"]
            for f in range(20):
                key = (clip_id, f + 1)
                if key not in features.index:
                    continue
                row = features.loc[key]
                if pd.isna(row[TARGETS[0]]):
                    continue
                bucket = per_frame[f]
                bucket["emitted"].append(glob[i])
                bucket["frame"].append(frame_tokens[i, f])
                bucket["sumpool"].append(summed[i, f])
                bucket["count"].append([np.log1p(counts[i, f])])
                for name in TARGETS:
                    bucket["targets"][name].append(float(row[name]))
        seen += len(batch)
        if seen % 100 < args.batch:
            log(f"  {seen}/{len(clip_ids)} clips")

    usable_frames = [f for f in range(20)
                     if len(per_frame[f]["count"]) >= 120]
    log(f"{len(usable_frames)} frames with enough clips "
        f"({len(per_frame[usable_frames[0]]['count'])} each)"
        if usable_frames else "too few clips to probe")
    if not usable_frames:
        return 1

    blocks = ("emitted", "frame", "sumpool", "count")
    print()
    print(f"  {'target':18s}" + "".join(f"{k:>12s}" for k in blocks))
    for name in TARGETS:
        scores = {k: [] for k in blocks}
        for f in usable_frames:
            bucket = per_frame[f]
            n = len(bucket["count"])
            order = np.random.RandomState(f).permutation(n)
            y = np.asarray(bucket["targets"][name], dtype=np.float64)[order]
            for k in blocks:
                x = np.asarray(bucket[k], dtype=np.float32)[order]
                scores[k].append(ridge_r2(x, y))
        cells = "".join(f"{np.nanmean(scores[k]):12.3f}" for k in blocks)
        print(f"  {name:18s}{cells}")
    print()
    print("  Mean R^2 over frames, held-out rows; <=0 means no better than the")
    print("  mean. `emitted` is exactly what the language model receives;")
    print("  `sumpool` is a control that is not in the model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
