#!/usr/bin/env python3
"""Per-frame scene material: ego dynamics, labelled objects, radar, and the
overlap between the last two.

One row per (clip_id, frame). Frames are the same 1 Hz index the QA rationales
use -- frame N is t = (N - 1) seconds -- so this table joins directly against
task 10 and against anything built for task 11.

The columns are the raw material for six kinds of scene description, each of
which trains a different pathway:

  ego         speed / accel / yaw rate / lateral accel. Derived from
              `egomotion.offline`, which is a sensor measurement rather than an
              autolabel, so these are the most trustworthy numbers available.
  objects     counts by class, nearest range, moving vs stationary. From
              `obstacle.offline`, restricted to the forward sector the downloaded
              sensors can actually see.
  radar       return count, how many survive ego-motion compensation, RCS
              statistics. Without descriptions grounded in these, a
              radar+vision+text model can ignore its radar input entirely and
              still score well.
  overlap     which labelled boxes have a radar return inside them and which do
              not. Nothing else in the release expresses sensor complementarity,
              and it is strongly class-dependent: heavy trucks are seen by radar
              85.7% of the time, pedestrians 20.2%, protruding objects never.

Motion is judged in the world frame. `obstacle.offline` is stored per-timestamp
in the rig frame, where a parked car sweeps past as the ego drives and a car
matching the ego's speed looks frozen; the two measures correlate at only 0.202.

Discretised bins for the ego channels are included so the same table can feed a
token interface without a second pass. Edges come from the measured 1st and 99th
percentiles over 300 clips, not from guesses.

    python -m datatools.scene_features --clips qa      # the 1,999 QA clips
    python -m datatools.scene_features --clips all     # every downloaded clip
"""

import argparse
import glob
import io
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import paths
from .geometry import (doppler_residual, extrinsics_to_matrix, interpolate_pose,
                       points_in_box, rig_to_world, sensor_to_rig,
                       spherical_to_sensor_xyz)

FRAMES = list(range(1, 22))          # 1 Hz over a 20 s clip, matching the QA index
DOPPLER_THRESHOLD_MS = 1.0           # keeps 47 of 954 LRR returns per scan
STATIONARY_M = 2.0                   # world-frame displacement over the clip
BOX_MARGIN = 1.3
RADARS = ("mrr2", "lrr1", "srr0")

CLASSES = ("automobile", "person", "heavy_truck", "bus", "trailer", "rider",
           "protruding_object", "animal", "stroller", "other_vehicle")

# Bin edges for the ego channels, from measured p1..p99 over 300 clips:
# speed 0.01..30.39 m/s, accel -2.55..2.26, yaw rate -19.88..17.01 deg/s,
# lateral accel -2.35..2.07. Outliers land in the end bins rather than being
# clipped away.
BIN_EDGES = {
    "ego_speed_ms": np.linspace(0.0, 32.0, 33),
    "ego_accel_ms2": np.linspace(-4.0, 4.0, 33),
    "ego_yaw_rate_degs": np.linspace(-25.0, 25.0, 33),
    "ego_lat_accel_ms2": np.linspace(-4.0, 4.0, 33),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_member(root, zip_rel, member):
    with zipfile.ZipFile(os.path.join(root, zip_rel)) as zf:
        return pd.read_parquet(io.BytesIO(zf.read(member)))


def ego_series(ego):
    """Derived ego dynamics at the egomotion sample rate."""
    t = ego["timestamp"].to_numpy() / 1e6
    xy = ego[["x", "y"]].to_numpy()
    quat = ego[["qx", "qy", "qz", "qw"]].to_numpy()

    velocity = np.gradient(xy, t, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    accel = np.gradient(speed, t)
    # Yaw from the quaternion's z component, unwrapped so the rate is continuous
    # across the +/-pi boundary.
    qx, qy, qz, qw = quat.T
    yaw = np.unwrap(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2)))
    yaw_rate = np.gradient(yaw, t)
    travelled = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    return {"t": t, "velocity": velocity, "speed": speed, "accel": accel,
            "yaw_rate": yaw_rate, "travelled": travelled}


def at_time(series, t_s):
    return int(np.argmin(np.abs(series["t"] - t_s)))


def obstacle_world(obstacle, ego):
    """Boxes lifted into the world frame, plus per-track world displacement."""
    t_us = obstacle["timestamp_us"].to_numpy()
    rig = obstacle[["center_x", "center_y", "center_z"]].to_numpy()
    pose_xyz, pose_rot = interpolate_pose(
        t_us, ego["timestamp"].to_numpy(), ego[["x", "y", "z"]].to_numpy(),
        ego[["qx", "qy", "qz", "qw"]].to_numpy())
    world = rig_to_world(rig, pose_xyz, pose_rot)

    frame = obstacle.copy()
    frame["t_s"] = t_us / 1e6
    frame["wx"], frame["wy"] = world[:, 0], world[:, 1]
    frame["range_m"] = np.hypot(rig[:, 0], rig[:, 1])
    frame["azimuth_deg"] = np.degrees(np.arctan2(rig[:, 1], rig[:, 0]))

    span = frame.groupby("track_id")[["wx", "wy"]].agg(["first", "last"])
    displacement = np.hypot(span[("wx", "last")] - span[("wx", "first")],
                            span[("wy", "last")] - span[("wy", "first")])
    frame["track_moved"] = frame["track_id"].map(displacement >= STATIONARY_M)
    return frame


def radar_scan(radar, extrinsics, sensor_name, t_s, ego, ego_derived):
    """One radar scan in the rig frame, with the ego-compensated Doppler residual."""
    scans = np.sort(radar["timestamp"].unique())
    if len(scans) == 0:
        return None
    pick = scans[int(np.argmin(np.abs(scans / 1e6 - t_s)))]
    scan = radar[radar["timestamp"] == pick]
    if scan.empty:
        return None

    rotation, translation = extrinsics_to_matrix(extrinsics.loc[sensor_name])
    sensor_xyz = spherical_to_sensor_xyz(scan["azimuth"].to_numpy(),
                                         scan["elevation"].to_numpy(),
                                         scan["distance"].to_numpy())
    rig = sensor_to_rig(sensor_xyz, rotation, translation)

    # Ego velocity rotated from world into the rig frame. Using the scalar speed
    # along +x instead would misstate the expected Doppler whenever the vehicle
    # is turning.
    i = at_time(ego_derived, t_s)
    v_world = np.array([ego_derived["velocity"][i, 0],
                        ego_derived["velocity"][i, 1], 0.0])
    quat = ego[["qx", "qy", "qz", "qw"]].to_numpy()
    j = min(i, len(quat) - 1)
    from scipy.spatial.transform import Rotation
    v_rig = Rotation.from_quat(quat[j]).as_matrix().T @ v_world

    # doppler_ambiguity must be passed: the MRR's unambiguous interval is ~22 m/s,
    # so without unfolding every return at highway speed reads as a mover.
    residual = doppler_residual(rig, scan["radial_velocity"].to_numpy(), v_rig,
                                scan["doppler_ambiguity"].to_numpy())
    return {"rig": rig, "residual": residual, "t_s": pick / 1e6,
            "rcs": scan["rcs"].to_numpy(), "distance": scan["distance"].to_numpy()}


def frame_rows(clip_id, row, nvidia_root):
    ego = read_member(nvidia_root, row["egomotion_zip"], row["egomotion_member"])
    obstacle = read_member(nvidia_root, row["obstacle_zip"], row["obstacle_member"])
    derived = ego_series(ego)
    boxes = obstacle_world(obstacle, ego) if not obstacle.empty else None

    extrinsics = None
    radars = {}
    for short in RADARS:
        if not row.get(f"has_{short}", False):
            continue
        radars[short] = read_member(nvidia_root, row[f"radar_{short}_zip"],
                                    row[f"radar_{short}_member"])
    if radars:
        full = pd.read_parquet(os.path.join(nvidia_root,
                                            row["radar_extrinsics_parquet"]))
        if clip_id in full.index.get_level_values(0):
            extrinsics = full.loc[clip_id]

    sensor_name = {"srr0": "radar_front_center_srr_0",
                   "mrr2": "radar_front_center_mrr_2",
                   "lrr1": "radar_front_center_imaging_lrr_1"}

    out = []
    for frame in FRAMES:
        t_s = float(frame - 1)
        if t_s > derived["t"].max():
            break
        i = at_time(derived, t_s)
        record = {
            "clip_id": clip_id, "frame": frame, "t_s": t_s,
            "ego_speed_ms": float(derived["speed"][i]),
            "ego_accel_ms2": float(derived["accel"][i]),
            "ego_yaw_rate_degs": float(np.degrees(derived["yaw_rate"][i])),
            "ego_lat_accel_ms2": float(derived["speed"][i] * derived["yaw_rate"][i]),
            "ego_curvature_invm": float(derived["yaw_rate"][i]
                                        / max(derived["speed"][i], 0.5)),
            "ego_travelled_m": float(derived["travelled"][i]),
        }

        # Objects inside the forward sector, near this frame's time.
        in_fov_boxes = []
        if boxes is not None:
            near = boxes[(boxes["t_s"] - t_s).abs() <= 0.6]
            visible = near[(near["azimuth_deg"].abs() <= paths.FRONT_FOV_DEG)
                           & (near["range_m"] <= paths.FRONT_MAX_RANGE_M)]
            record["n_tracks_in_fov"] = int(visible["track_id"].nunique())
            record["n_moving_in_fov"] = int(
                visible.loc[visible["track_moved"].fillna(False), "track_id"].nunique())
            record["nearest_range_m"] = (float(visible["range_m"].min())
                                         if not visible.empty else None)
            record["nearest_class"] = (
                visible.loc[visible["range_m"].idxmin(), "label_class"]
                if not visible.empty else None)
            counts = visible.groupby("label_class")["track_id"].nunique()
            for name in CLASSES:
                record[f"n_{name}"] = int(counts.get(name, 0))
            in_fov_boxes = list(visible.drop_duplicates("track_id").itertuples())
            # For the radar-hit test the box has to be near the scan in time as
            # well: a 0.6 s stale box is up to 6 m out of place at 10 m/s, which
            # is enough to miss the returns that actually illuminate it.
            tight = near[(near["t_s"] - t_s).abs() <= 0.12]
            tight = tight[(tight["azimuth_deg"].abs() <= paths.FRONT_FOV_DEG)
                          & (tight["range_m"] <= paths.FRONT_MAX_RANGE_M)]
            hit_boxes = list(tight.drop_duplicates("track_id").itertuples())
        else:
            record["n_tracks_in_fov"] = 0
            record["n_moving_in_fov"] = 0
            record["nearest_range_m"] = None
            record["nearest_class"] = None
            hit_boxes = []
            for name in CLASSES:
                record[f"n_{name}"] = 0

        # Radar, per sensor, plus how many boxes it actually illuminates.
        boxes_hit = 0
        preferred_radar = next((r for r in ("lrr1", "mrr2", "srr0") if r in radars),
                               None)
        for short, radar in radars.items():
            scan = (radar_scan(radar, extrinsics, sensor_name[short], t_s,
                               ego, derived)
                    if extrinsics is not None else None)
            if scan is None:
                continue
            moving = scan["residual"] > DOPPLER_THRESHOLD_MS
            record[f"{short}_n_points"] = int(len(scan["rig"]))
            record[f"{short}_n_moving"] = int(moving.sum())
            record[f"{short}_max_rcs"] = float(np.max(scan["rcs"]))
            record[f"{short}_mean_rcs"] = float(np.mean(scan["rcs"]))
            record[f"{short}_p90_range_m"] = float(np.percentile(scan["distance"], 90))
            # Test with whichever front radar this clip actually carries, in
            # order of resolution. Restricting this to lrr1/mrr2 meant every
            # radar_config='low' clip -- 49% of frames, 29.5 million boxes --
            # reported 0% radar coverage and 100% camera-only, because its only
            # front radar is the SRR and the SRR was never checked.
            if short == preferred_radar:
                for box in hit_boxes:
                    inside = points_in_box(
                        scan["rig"],
                        np.array([box.center_x, box.center_y, box.center_z]),
                        [box.size_x, box.size_y, box.size_z],
                        [box.orientation_x, box.orientation_y,
                         box.orientation_z, box.orientation_w],
                        margin=BOX_MARGIN)
                    boxes_hit += int(inside.any())
        record["n_boxes_tested_for_radar"] = len(hit_boxes)
        record["n_boxes_with_radar"] = boxes_hit
        record["n_boxes_camera_only"] = max(0, len(hit_boxes) - boxes_hit)
        out.append(record)
    return out


def process(args_tuple):
    clip_id, row, nvidia_root = args_tuple
    try:
        return frame_rows(clip_id, row, nvidia_root)
    except Exception as exc:
        return [{"clip_id": clip_id, "frame": None,
                 "error": f"{type(exc).__name__}: {str(exc)[:80]}"}]


def add_bins(frame):
    for column, edges in BIN_EDGES.items():
        if column in frame.columns:
            frame[f"{column}_bin"] = np.clip(
                np.digitize(frame[column].to_numpy(dtype=float), edges) - 1,
                0, len(edges) - 2)
    return frame


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clips", default="qa", choices=("qa", "all"))
    ap.add_argument("--nvidia-root", default=paths.NVIDIA_ROOT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    clips = pd.read_parquet(os.path.join(paths.COMMON_DIR, "nvidia_clips.parquet"))
    if args.clips == "qa":
        ids = [os.path.basename(f)[:-5] for f in sorted(glob.glob(
            os.path.join(paths.SPLIT_ROOT, "10_radar_vision_qa/qa/*.json")))]
        default_out = "scene_features_qa_clips.parquet"
    else:
        ids = list(clips.index)
        default_out = "scene_features_all_clips.parquet"
    ids = [i for i in ids if i in clips.index]
    if args.limit:
        ids = ids[: args.limit]
    out_path = args.out or os.path.join(paths.COMMON_DIR, default_out)

    usable = clips.loc[ids]
    usable = usable[usable["has_egomotion"] & usable["has_obstacle"]]
    log(f"{len(usable):,} clips x up to {len(FRAMES)} frames")

    tasks = [(cid, usable.loc[cid].to_dict(), args.nvidia_root)
             for cid in usable.index]
    rows, errors = [], 0
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, t) for t in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result and "error" in result[0]:
                errors += 1
            else:
                rows.extend(result)
            if i % 200 == 0 or i == len(tasks):
                rate = i / (time.monotonic() - started)
                eta = (len(tasks) - i) / rate / 60 if rate else float("nan")
                log(f"  {i}/{len(tasks)}  {rate:.1f} clip/s  ETA {eta:.1f} min")

    frame = add_bins(pd.DataFrame(rows))
    frame.to_parquet(out_path, index=False)
    log(f"wrote {out_path}  rows={len(frame):,}  "
        f"({os.path.getsize(out_path)/1e6:.1f} MB)")
    if errors:
        log(f"clips that failed: {errors}")
    summarise(frame)
    return 0


def summarise(frame):
    log("")
    log(f"clips {frame['clip_id'].nunique():,}  frames/clip "
        f"{frame.groupby('clip_id').size().median():.0f}")
    for column in ("ego_speed_ms", "ego_accel_ms2", "ego_yaw_rate_degs",
                   "n_tracks_in_fov", "nearest_range_m"):
        if column in frame.columns:
            values = frame[column].dropna()
            log(f"  {column:22s} p1 {values.quantile(0.01):8.2f}  "
                f"p50 {values.quantile(0.5):8.2f}  p99 {values.quantile(0.99):8.2f}")
    for short in RADARS:
        column = f"{short}_n_points"
        if column in frame.columns and frame[column].notna().any():
            sub = frame[frame[column].notna()]
            log(f"  {short}: {len(sub):,} frames, {sub[column].median():.0f} points, "
                f"{sub[f'{short}_n_moving'].median():.0f} moving "
                f"({sub[f'{short}_n_moving'].sum()/sub[column].sum()*100:.1f}%)")
    if "n_boxes_with_radar" in frame.columns:
        hit = frame["n_boxes_with_radar"].sum()
        only = frame["n_boxes_camera_only"].sum()
        log(f"  boxes in FOV: {hit + only:,}  with radar {hit:,} "
            f"({hit/max(hit+only,1)*100:.1f}%)  camera-only {only:,}")


if __name__ == "__main__":
    sys.exit(main())
