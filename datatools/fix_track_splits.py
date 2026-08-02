#!/usr/bin/env python3
"""Rebuild the tracking and agent-trajectory splits on world-frame motion.

Replaces the filters in `02_tracking/` and `03_2_agent_trajectory/`, which were
built from rig-frame displacement. Rig is ego-centric, so that measure asked
"did this object move relative to the car?" instead of "did it move?" - keeping
parked cars and discarding lead vehicles, the exact opposite of the intent.

Run after `datatools.rescan_tracks`, which produces
`common/nvidia_tracks_world.parquet`.

    python -m datatools.fix_track_splits
"""

import argparse
import json
import os
import sys
import time

import pandas as pd

from . import paths

# 3 s of observation is the minimum that exercises re-identification; the median
# track only lasts 2.5 s, so this is a real constraint rather than a formality.
TRACKING_MIN_OBS = 30
TRACKING_MIN_LONG_TRACKS = 3

# 6 s supports the standard 3 s observation + 3 s prediction split.
AGENT_MIN_OBS = 60

# Only 20% of tracks stay inside the forward sector for most of their life. A
# front-only model cannot predict what it never sees, so the trajectory set is
# restricted rather than left to score against unobservable agents.
AGENT_MIN_FOV_FRACTION = 0.5

NV_SPLITS = ("train", "val", "test")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write(frame, directory, name):
    path = os.path.join(directory, f"{name}.parquet")
    frame.to_parquet(path, index=False)
    log(f"    {name:38s} rows={len(frame):>9,}  {os.path.getsize(path)/1e6:>7.1f} MB")


def nvidia_columns():
    base = ["clip_id", "chunk", "split", "country", "month", "hour_of_day",
            "platform_class", "radar_config", "n_front_radars",
            "camera_zip", "camera_video", "camera_timestamps",
            "camera_intrinsics_parquet", "sensor_extrinsics_parquet",
            "radar_extrinsics_parquet", "has_radar_extrinsics",
            "radar_extrinsics_sensors",
            "obstacle_zip", "obstacle_member",
            "egomotion_zip", "egomotion_member"]
    for short in ("srr0", "mrr2", "lrr1"):
        base += [f"has_{short}", f"radar_{short}_zip", f"radar_{short}_member"]
    return base


def rebuild_tracking(clips, tracks, split_root):
    directory = os.path.join(split_root, "02_tracking")
    log("02_tracking")
    cols = nvidia_columns()

    long = tracks[tracks["n_obs"] >= TRACKING_MIN_OBS]
    per_clip = (long.groupby("clip_id")
                .agg(n_long_tracks=("track_id", "size"),
                     n_moving_tracks=("is_stationary", lambda s: int((~s).sum())),
                     n_long_tracks_in_fov=("frac_in_front_fov",
                                           lambda s: int((s >= 0.5).sum())),
                     max_track_s=("duration_s", "max"),
                     median_track_s=("duration_s", "median"))
                .reset_index())

    sel = clips[clips["has_obstacle"]].merge(per_clip, on="clip_id", how="inner")
    sel = sel[sel["n_long_tracks"] >= TRACKING_MIN_LONG_TRACKS]
    extra = ["n_long_tracks", "n_moving_tracks", "n_long_tracks_in_fov",
             "max_track_s", "median_track_s"]
    for split in NV_SPLITS:
        write(sel[sel["split"] == split][cols + extra], directory, f"nvidia_{split}")
    write(long, directory, "nvidia_long_tracks")

    with open(os.path.join(directory, "config.json")) as fh:
        cfg = json.load(fh)
    cfg["nvidia_filter"] = {
        "min_obs_per_track": TRACKING_MIN_OBS,
        "min_long_tracks_per_clip": TRACKING_MIN_LONG_TRACKS,
        "motion_frame": "world (clip-local)",
        "note": "n_moving_tracks and n_long_tracks_in_fov are provided per clip so "
                "a stricter subset can be taken without rescanning.",
    }
    cfg["superseded_filter"] = ("rig-frame path length, which measured motion "
                               "relative to the ego vehicle and so inverted the "
                               "stationary/moving distinction")
    with open(os.path.join(directory, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)


def rebuild_agent(clips, tracks, split_root):
    directory = os.path.join(split_root, "03_2_agent_trajectory")
    log("03_2_agent_trajectory")

    agents = tracks[(tracks["n_obs"] >= AGENT_MIN_OBS)
                    & (~tracks["is_stationary"])
                    & (tracks["frac_in_front_fov"] >= AGENT_MIN_FOV_FRACTION)].copy()

    keys = clips[["clip_id", "split", "chunk", "country", "hour_of_day",
                  "radar_config", "n_front_radars",
                  "obstacle_zip", "obstacle_member",
                  "egomotion_zip", "egomotion_member",
                  "camera_zip", "camera_video",
                  "sensor_extrinsics_parquet",
                  "radar_extrinsics_parquet", "has_radar_extrinsics"]]
    agents = agents.drop(columns=["chunk"]).merge(keys, on="clip_id", how="inner")
    for split in NV_SPLITS:
        write(agents[agents["split"] == split], directory, f"nvidia_{split}")

    with open(os.path.join(directory, "config.json")) as fh:
        cfg = json.load(fh)
    cfg["nvidia_filter"] = {
        "min_obs_per_track": AGENT_MIN_OBS,
        "exclude_stationary": True,
        "stationary_definition": "world-frame net displacement < 2 m",
        "min_fraction_inside_front_fov": AGENT_MIN_FOV_FRACTION,
        "motion_frame": "world (clip-local), via egomotion",
    }
    cfg["superseded_filter"] = ("rig-frame path length >= 2 m, which kept parked "
                               "vehicles and dropped vehicles travelling at the "
                               "ego speed")
    with open(os.path.join(directory, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split-root", default=paths.SPLIT_ROOT)
    args = ap.parse_args(argv)

    common = os.path.join(args.split_root, "common")
    tracks_path = os.path.join(common, "nvidia_tracks_world.parquet")
    if not os.path.exists(tracks_path):
        raise SystemExit(f"missing {tracks_path} - run datatools.rescan_tracks first")

    clips = pd.read_parquet(os.path.join(common, "nvidia_clips.parquet")).reset_index()
    tracks = pd.read_parquet(tracks_path)
    log(f"clips {len(clips):,} | tracks {len(tracks):,}")

    long = tracks[tracks["n_obs"] >= TRACKING_MIN_OBS]
    log(f"tracks >= 3 s: {len(long):,}  "
        f"stationary {long['is_stationary'].mean()*100:.1f}%  "
        f"mostly-in-FOV {(long['frac_in_front_fov'] >= 0.5).mean()*100:.1f}%")

    rebuild_tracking(clips, tracks, args.split_root)
    rebuild_agent(clips, tracks, args.split_root)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
