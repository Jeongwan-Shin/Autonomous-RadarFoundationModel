#!/usr/bin/env python3
"""Scan every obstacle.offline archive and summarise the tracks inside.

SUPERSEDED by `rescan_tracks.py`, which measures displacement in the world frame
instead of the rig frame. Kept only so the original `nvidia_tracks.parquet` can
be reproduced; do not build new filters on its rig-frame columns.

Tracking and agent-trajectory splits need to know how long each track is
observed before they can be filtered, and that is not recorded anywhere in the
metadata -- it only exists inside the per-clip parquet files. This walks all
1,838 chunk archives once and writes two tables to common/:

  nvidia_tracks.parquet        one row per (clip_id, track_id)
  nvidia_clip_obstacle.parquet one row per clip, aggregated

Shards are written per chunk so an interrupted run resumes cheaply.
"""

import argparse
import io
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import paths

NV = paths.NVIDIA_ROOT
OUT = paths.COMMON_DIR
SHARDS = os.path.join(OUT, "_track_shards")

OBSTACLE_DIR = "labels/obstacle.offline"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scan_chunk(chunk):
    """Summarise every track in one chunk. Returns (track_rows, clip_rows)."""
    shard = os.path.join(SHARDS, f"chunk_{chunk:04d}.parquet")
    clip_shard = os.path.join(SHARDS, f"clip_{chunk:04d}.parquet")
    if os.path.exists(shard) and os.path.exists(clip_shard):
        return chunk, True, 0

    zpath = os.path.join(NV, OBSTACLE_DIR,
                         f"obstacle.offline.chunk_{chunk:04d}.zip")
    if not os.path.exists(zpath):
        return chunk, False, 0

    tracks, clips = [], []
    with zipfile.ZipFile(zpath) as zf:
        for name in zf.namelist():
            if not name.endswith(".parquet"):
                continue
            clip_id = name.split(".obstacle")[0]
            df = pd.read_parquet(io.BytesIO(zf.read(name)))
            if df.empty:
                clips.append({"clip_id": clip_id, "chunk": chunk, "n_obs": 0,
                              "n_tracks": 0, "t_min_us": None, "t_max_us": None})
                continue

            df = df.sort_values("timestamp_us")
            for track_id, g in df.groupby("track_id", sort=False):
                pos = g[["center_x", "center_y", "center_z"]].to_numpy()
                step = np.linalg.norm(np.diff(pos, axis=0), axis=1) if len(pos) > 1 else np.zeros(0)
                t = g["timestamp_us"].to_numpy()
                tracks.append({
                    "clip_id": clip_id,
                    "chunk": chunk,
                    "track_id": str(track_id),
                    "label_class": g["label_class"].iloc[0],
                    "n_obs": len(g),
                    "t_start_us": int(t[0]),
                    "t_end_us": int(t[-1]),
                    "duration_s": float((t[-1] - t[0]) / 1e6),
                    # net displacement vs path length separates parked cars
                    # (both near zero) from ones that circle back.
                    "net_disp_m": float(np.linalg.norm(pos[-1] - pos[0])),
                    "path_len_m": float(step.sum()),
                    "mean_range_m": float(np.linalg.norm(pos[:, :2], axis=1).mean()),
                })

            clips.append({
                "clip_id": clip_id, "chunk": chunk,
                "n_obs": len(df), "n_tracks": df["track_id"].nunique(),
                "t_min_us": int(df["timestamp_us"].min()),
                "t_max_us": int(df["timestamp_us"].max()),
            })

    pd.DataFrame(tracks).to_parquet(shard, index=False)
    pd.DataFrame(clips).to_parquet(clip_shard, index=False)
    return chunk, True, len(tracks)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--combine-only", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(SHARDS, exist_ok=True)
    import json
    manifest = json.load(open(os.path.join(NV, "manifest.json")))
    chunks = sorted(manifest["chunks_complete"])

    if not args.combine_only:
        log(f"scanning {len(chunks)} chunks with {args.workers} workers")
        done = 0
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(scan_chunk, c) for c in chunks]
            for future in as_completed(futures):
                chunk, ok, n = future.result()
                done += 1
                if done % 100 == 0 or done == len(chunks):
                    rate = done / (time.monotonic() - started)
                    eta = (len(chunks) - done) / rate / 60 if rate else float("nan")
                    log(f"  {done}/{len(chunks)} chunks  {rate:.1f} chunk/s  ETA {eta:.1f} min")

    log("combining shards")
    track_files = sorted(f for f in os.listdir(SHARDS) if f.startswith("chunk_"))
    clip_files = sorted(f for f in os.listdir(SHARDS) if f.startswith("clip_"))

    tracks = pd.concat([pd.read_parquet(os.path.join(SHARDS, f)) for f in track_files],
                       ignore_index=True)
    clips = pd.concat([pd.read_parquet(os.path.join(SHARDS, f)) for f in clip_files],
                      ignore_index=True)

    tpath = os.path.join(OUT, "nvidia_tracks.parquet")
    cpath = os.path.join(OUT, "nvidia_clip_obstacle.parquet")
    tracks.to_parquet(tpath, index=False)
    clips.to_parquet(cpath, index=False)

    log(f"tracks: {len(tracks):,} rows -> {tpath} ({os.path.getsize(tpath)/1e6:.1f} MB)")
    log(f"clips : {len(clips):,} rows -> {cpath} ({os.path.getsize(cpath)/1e6:.1f} MB)")
    log(f"track length: median {tracks['n_obs'].median():.0f} obs, "
        f"{tracks['duration_s'].median():.2f} s")
    for thresh in (30, 60, 100):
        n = (tracks["n_obs"] >= thresh).sum()
        log(f"  tracks with >= {thresh} obs ({thresh/10:.0f}s): {n:,} "
            f"({n/len(tracks)*100:.1f}%)")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
