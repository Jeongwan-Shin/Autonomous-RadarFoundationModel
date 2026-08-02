#!/usr/bin/env python3
"""Recompute obstacle track statistics in the world frame.

The first pass measured track displacement in the rig frame, which is the wrong
frame for deciding whether an object moved: rig is ego-centric, so a parked car
sweeps past as the ego drives while a car matching the ego's speed looks frozen.
Checked on 5,000 tracks, the two measures correlate at 0.194, and 63.9% of
tracks are genuinely stationary against 1.1% by the rig measure. Any filter
built on the rig number keeps the wrong tracks.

This rescans every obstacle.offline archive, lifts each box into the clip-local
world frame with egomotion, and records both measures side by side plus the
fraction of each track that sits inside the front sensor sector.

    python -m datatools.rescan_tracks --workers 14
    python -m datatools.rescan_tracks --combine-only
"""

import argparse
import io
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import paths
from .geometry import interpolate_pose, rig_to_world

OBSTACLE_DIR = "labels/obstacle.offline"
EGOMOTION_DIR = "labels/egomotion.offline"
SHARD_DIRNAME = "_track_world_shards"

# A track counts as stationary if it moves less than this in the world frame
# over its whole observation. 2 m is about half a car length, comfortably above
# autolabel jitter without catching slow creeping traffic.
STATIONARY_M = 2.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_member(zip_path, member):
    with zipfile.ZipFile(zip_path) as zf:
        return pd.read_parquet(io.BytesIO(zf.read(member)))


def clip_tracks(obstacle, egomotion, chunk, clip_id):
    """Per-track summary for one clip, in both rig and world frames."""
    if obstacle.empty or egomotion.empty:
        return []

    ego_t = egomotion["timestamp"].to_numpy()
    ego_xyz = egomotion[["x", "y", "z"]].to_numpy()
    ego_quat = egomotion[["qx", "qy", "qz", "qw"]].to_numpy()

    obstacle = obstacle.sort_values("timestamp_us")
    t_us = obstacle["timestamp_us"].to_numpy()
    rig = obstacle[["center_x", "center_y", "center_z"]].to_numpy()

    pose_xyz, pose_rot = interpolate_pose(t_us, ego_t, ego_xyz, ego_quat)
    world = rig_to_world(rig, pose_xyz, pose_rot)

    azimuth = np.degrees(np.arctan2(rig[:, 1], rig[:, 0]))
    ground_range = np.hypot(rig[:, 0], rig[:, 1])
    in_fov = (np.abs(azimuth) <= paths.FRONT_FOV_DEG) & \
             (ground_range <= paths.FRONT_MAX_RANGE_M)

    frame = pd.DataFrame({
        "track_id": obstacle["track_id"].astype(str).to_numpy(),
        "label_class": obstacle["label_class"].to_numpy(),
        "t_us": t_us,
        "rx": rig[:, 0], "ry": rig[:, 1],
        "wx": world[:, 0], "wy": world[:, 1],
        "range_m": ground_range,
        "in_fov": in_fov,
    })

    out = []
    for track_id, g in frame.groupby("track_id", sort=False):
        t = g["t_us"].to_numpy()
        duration = float((t[-1] - t[0]) / 1e6)
        rig_xy = g[["rx", "ry"]].to_numpy()
        world_xy = g[["wx", "wy"]].to_numpy()

        def net(p):
            return float(np.linalg.norm(p[-1] - p[0]))

        def path(p):
            return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) \
                if len(p) > 1 else 0.0

        world_net = net(world_xy)
        out.append({
            "clip_id": clip_id,
            "chunk": chunk,
            "track_id": track_id,
            "label_class": g["label_class"].iloc[0],
            "n_obs": int(len(g)),
            "t_start_us": int(t[0]),
            "t_end_us": int(t[-1]),
            "duration_s": duration,
            # rig frame, kept only so the old filter can be audited
            "rig_net_disp_m": net(rig_xy),
            "rig_path_len_m": path(rig_xy),
            # world frame: the frame that actually answers "did it move?"
            "world_net_disp_m": world_net,
            "world_path_len_m": path(world_xy),
            "world_mean_speed_ms": float(path(world_xy) / duration) if duration > 0 else 0.0,
            "is_stationary": bool(world_net < STATIONARY_M),
            "mean_range_m": float(g["range_m"].mean()),
            "frac_in_front_fov": float(g["in_fov"].mean()),
        })
    return out


def scan_chunk(chunk, nvidia_root, shard_dir):
    shard = os.path.join(shard_dir, f"chunk_{chunk:04d}.parquet")
    if os.path.exists(shard):
        return chunk, 0

    obstacle_zip = os.path.join(nvidia_root, OBSTACLE_DIR,
                                f"obstacle.offline.chunk_{chunk:04d}.zip")
    ego_zip = os.path.join(nvidia_root, EGOMOTION_DIR,
                           f"egomotion.offline.chunk_{chunk:04d}.zip")
    if not (os.path.exists(obstacle_zip) and os.path.exists(ego_zip)):
        return chunk, 0

    rows = []
    with zipfile.ZipFile(obstacle_zip) as oz, zipfile.ZipFile(ego_zip) as ez:
        ego_members = {n.split(".egomotion")[0]: n for n in ez.namelist()
                       if n.endswith(".parquet")}
        for name in oz.namelist():
            if not name.endswith(".parquet"):
                continue
            clip_id = name.split(".obstacle")[0]
            ego_name = ego_members.get(clip_id)
            if ego_name is None:
                continue  # no egomotion -> cannot resolve the world frame
            obstacle = pd.read_parquet(io.BytesIO(oz.read(name)))
            egomotion = pd.read_parquet(io.BytesIO(ez.read(ego_name)))
            rows.extend(clip_tracks(obstacle, egomotion, chunk, clip_id))

    pd.DataFrame(rows).to_parquet(shard, index=False)
    return chunk, len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nvidia-root", default=paths.NVIDIA_ROOT)
    ap.add_argument("--out-dir", default=paths.COMMON_DIR)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--combine-only", action="store_true")
    args = ap.parse_args(argv)

    shard_dir = os.path.join(args.out_dir, SHARD_DIRNAME)
    os.makedirs(shard_dir, exist_ok=True)

    manifest = json.load(open(os.path.join(args.nvidia_root, "manifest.json")))
    chunks = sorted(manifest["chunks_complete"])

    if not args.combine_only:
        log(f"rescanning {len(chunks)} chunks with {args.workers} workers")
        started = time.monotonic()
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(scan_chunk, c, args.nvidia_root, shard_dir)
                       for c in chunks]
            for future in as_completed(futures):
                future.result()
                done += 1
                if done % 100 == 0 or done == len(chunks):
                    rate = done / (time.monotonic() - started)
                    eta = (len(chunks) - done) / rate / 60 if rate else float("nan")
                    log(f"  {done}/{len(chunks)}  {rate:.2f} chunk/s  ETA {eta:.1f} min")

    log("combining shards")
    files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".parquet"))
    tracks = pd.concat([pd.read_parquet(os.path.join(shard_dir, f)) for f in files],
                       ignore_index=True)

    out_path = os.path.join(args.out_dir, "nvidia_tracks_world.parquet")
    tracks.to_parquet(out_path, index=False)
    log(f"wrote {out_path}  rows={len(tracks):,}  "
        f"({os.path.getsize(out_path)/1e6:.1f} MB)")

    long = tracks[tracks["n_obs"] >= 30]
    log(f"tracks >= 3 s: {len(long):,}")
    log(f"  stationary in world (< {STATIONARY_M} m): "
        f"{long['is_stationary'].mean()*100:.1f}%")
    log(f"  would have been called stationary in rig: "
        f"{(long['rig_path_len_m'] < STATIONARY_M).mean()*100:.1f}%")
    corr = long["world_net_disp_m"].corr(long["rig_net_disp_m"])
    log(f"  world vs rig displacement correlation: {corr:.3f}")
    log(f"  fully inside front FOV: "
        f"{(long['frac_in_front_fov'] >= 0.9).mean()*100:.1f}%")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
