#!/usr/bin/env python3
"""Per-object text targets for tasks 01-06, in one pass over the raw archives.

`scene_features.py` reduces each frame to counts. That is enough for a
description, but it turns detection into "how many cars are there", which is not
the task. This module keeps the objects themselves -- class, range, azimuth,
world-frame motion, and whether the radar illuminates them -- and writes the
instruction/target text directly, so the training loader stays a lookup.

Serialising here rather than in the loader is deliberate. The geometry needs the
obstacle archive, the egomotion archive and a radar scan per frame; doing that
inside a dataloader worker would re-open three zips per sample and put the
autolabel-to-world transform on the critical path of every step.

One row per (clip_id, task, frame). Tasks emitted:

  det_objects    01  every road user ahead: class, range, azimuth, motion
  track_identity 02  the same objects keyed by track id, with how long each has
                     been visible -- identity persistence, not re-detection
  plan_ego       03-1 future ego waypoints as (x, y) offsets in the current ego
                     frame, forward-positive
  agent_traj     03-2 one agent's future range/azimuth over the next 3 s
  world_model    04  the scene 3 s ahead, conditioned on the ego action taken
  depth_range    05  nearest object, and nearest object the radar confirms --
                     the pairing is what makes it radar-verifiable rather than
                     monocular guesswork
  motion_seg     06  moving vs stationary, judged in the world frame

Anchor frames are fixed at 6/11/16 (1-indexed, so t = 5/10/15 s). Every task
that predicts 3 s ahead therefore stays inside the 21-frame clip, and the three
anchors give an early, middle and late view of each clip without emitting 21
near-duplicate items per task.

    python -m datatools.frame_objects --clips all --workers 14
"""

import argparse
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

ANCHOR_FRAMES = (6, 11, 16)      # 1-indexed; t = frame - 1 seconds
HORIZON_S = (1.0, 2.0, 3.0)      # prediction offsets for 03-1, 03-2, 04
BOX_TIME_WINDOW_S = 0.15         # a 10 Hz label stream, so this catches 1-2 obs
DOPPLER_THRESHOLD_MS = 1.0
STATIONARY_M = 2.0
BOX_MARGIN = 1.3
MAX_LISTED = 8                   # objects per answer, nearest first

# Answers that cite radar evidence, and so may only be emitted for a clip that
# actually carries a front radar.
RADAR_EVIDENCE_TASKS = ("depth_range", "motion_seg", "motion_seg_cot",
                        "agent_traj_cot")

SENSOR_NAME = {"srr0": "radar_front_center_srr_0",
               "mrr2": "radar_front_center_mrr_2",
               "lrr1": "radar_front_center_imaging_lrr_1"}
RADAR_PREFERENCE = ("lrr1", "mrr2", "srr0")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_member(root, zip_rel, member):
    with zipfile.ZipFile(os.path.join(root, zip_rel)) as zf:
        return pd.read_parquet(io.BytesIO(zf.read(member)))


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def ego_frame(ego):
    """Ego pose and dynamics on the egomotion clock."""
    t = ego["timestamp"].to_numpy() / 1e6
    xy = ego[["x", "y"]].to_numpy()
    quat = ego[["qx", "qy", "qz", "qw"]].to_numpy()
    qx, qy, qz, qw = quat.T
    yaw = np.unwrap(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2)))
    velocity = np.gradient(xy, t, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    return {"t": t, "xy": xy, "yaw": yaw, "velocity": velocity, "speed": speed,
            "yaw_rate": np.gradient(yaw, t), "quat": quat}


def at_time(series, t_s):
    return int(np.argmin(np.abs(series["t"] - t_s)))


def boxes_world(obstacle, ego):
    """Boxes with rig-frame range/azimuth and a world-frame moved flag.

    Motion has to be judged in the world frame: `obstacle.offline` is stored
    relative to the rig, where a parked car sweeps past at the ego's own speed.
    """
    t_us = obstacle["timestamp_us"].to_numpy()
    rig = obstacle[["center_x", "center_y", "center_z"]].to_numpy()
    pose_xyz, pose_rot = interpolate_pose(
        t_us, ego["timestamp"].to_numpy(), ego[["x", "y", "z"]].to_numpy(),
        ego[["qx", "qy", "qz", "qw"]].to_numpy())
    world = rig_to_world(rig, pose_xyz, pose_rot)

    frame = obstacle.copy()
    frame["t_s"] = t_us / 1e6
    frame["range_m"] = np.hypot(rig[:, 0], rig[:, 1])
    frame["azimuth_deg"] = np.degrees(np.arctan2(rig[:, 1], rig[:, 0]))
    frame["wx"], frame["wy"] = world[:, 0], world[:, 1]

    span = frame.groupby("track_id")[["wx", "wy"]].agg(["first", "last"])
    moved = np.hypot(span[("wx", "last")] - span[("wx", "first")],
                     span[("wy", "last")] - span[("wy", "first")]) >= STATIONARY_M
    frame["moved"] = frame["track_id"].map(moved).fillna(False)

    first_seen = frame.groupby("track_id")["t_s"].min()
    frame["first_seen_s"] = frame["track_id"].map(first_seen)
    return frame


def visible_at(boxes, t_s):
    """One row per track near `t_s`, inside the forward sector, nearest first."""
    if boxes is None:
        return None
    near = boxes[(boxes["t_s"] - t_s).abs() <= BOX_TIME_WINDOW_S]
    near = near[(near["azimuth_deg"].abs() <= paths.FRONT_FOV_DEG)
                & (near["range_m"] <= paths.FRONT_MAX_RANGE_M)]
    if near.empty:
        return near
    # Closest observation in time wins, so range/azimuth are not up to a frame
    # stale before anything is written down.
    near = near.assign(_dt=(near["t_s"] - t_s).abs())
    near = near.sort_values("_dt").drop_duplicates("track_id")
    return near.sort_values("range_m")


def radar_scan(radar, extrinsics, sensor_name, t_s, ego, derived):
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

    i = at_time(derived, t_s)
    v_world = np.array([derived["velocity"][i, 0], derived["velocity"][i, 1], 0.0])
    from scipy.spatial.transform import Rotation
    j = min(i, len(derived["quat"]) - 1)
    v_rig = Rotation.from_quat(derived["quat"][j]).as_matrix().T @ v_world
    residual = doppler_residual(rig, scan["radial_velocity"].to_numpy(), v_rig,
                                scan["doppler_ambiguity"].to_numpy())
    return {"rig": rig, "residual": residual,
            "radial": scan["radial_velocity"].to_numpy(),
            "moving": residual > DOPPLER_THRESHOLD_MS}


def radar_hits(scan, rows):
    """Radar evidence per box: {track_id: (n_points, n_moving, radial_ms)}.

    `radial_ms` is the mean measured radial velocity of the returns inside the
    box -- negative closing, positive receding. It is here because it is the one
    radar quantity that stands *upstream* of an answer rather than being the
    answer: an object at 34 m closing at 12 m/s is at 22 m a second later, so a
    rationale that states the velocity determines the predicted position. Counts
    alone cannot support that chain.
    """
    out = {}
    if scan is None or rows is None or len(rows) == 0:
        return out
    for box in rows.itertuples():
        inside = points_in_box(
            scan["rig"],
            np.array([box.center_x, box.center_y, box.center_z]),
            [box.size_x, box.size_y, box.size_z],
            [box.orientation_x, box.orientation_y,
             box.orientation_z, box.orientation_w], BOX_MARGIN)
        if inside.any():
            # Averaged over the moving returns only. Including the static ones
            # drags the mean toward zero -- measured, it understated the actual
            # range change by 1.85x -- because a box overlapping road surface or
            # barrier collects clutter whose radial velocity is the ego's own.
            movers = inside & scan["moving"]
            radial = scan["radial"][movers] if movers.any() else scan["radial"][inside]
            out[box.track_id] = (int(inside.sum()),
                                 int(scan["moving"][inside].sum()),
                                 float(np.median(radial)))
    return out


def ego_waypoints(derived, t_s):
    """Future ego offsets in the ego frame at `t_s`, forward-positive."""
    i = at_time(derived, t_s)
    origin, yaw = derived["xy"][i], derived["yaw"][i]
    cos, sin = np.cos(yaw), np.sin(yaw)
    out = []
    for horizon in HORIZON_S:
        j = at_time(derived, t_s + horizon)
        if derived["t"][j] < t_s + horizon - 0.35:      # clip ends early
            return None
        delta = derived["xy"][j] - origin
        out.append((cos * delta[0] + sin * delta[1],
                    -sin * delta[0] + cos * delta[1]))
    return out


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------

def describe_object(row, hits=None, with_motion=True):
    """One object as text. `with_motion` is off where the grouping already says
    it -- task 06 lists under a moving/stationary heading, so repeating the word
    per object cost ~40% of that answer's length and taught nothing."""
    text = f"{row.label_class} {row.range_m:.0f} m az {row.azimuth_deg:+.0f} deg"
    if with_motion:
        text += " moving" if row.moved else " stationary"
    if hits is not None:
        n = hits.get(row.track_id, (0, 0, 0.0))[0]
        text += (f" ({n} radar return{'s' if n > 1 else ''})" if n
                 else " (no radar return)")
    return text


def ego_action(derived, t_s):
    """The manoeuvre the ego actually performs over the prediction horizon."""
    i, j = at_time(derived, t_s), at_time(derived, t_s + HORIZON_S[-1])
    speed_change = derived["speed"][j] - derived["speed"][i]
    yaw_change = np.degrees(derived["yaw"][j] - derived["yaw"][i])
    if speed_change > 1.5:
        pace = "accelerate"
    elif speed_change < -1.5:
        pace = "brake"
    else:
        pace = "hold speed"
    if yaw_change > 8:
        steer = "turn left"
    elif yaw_change < -8:
        steer = "turn right"
    else:
        steer = "go straight"
    return f"{pace} and {steer}"


def scene_summary(rows):
    if rows is None or len(rows) == 0:
        return "no road users ahead"
    counts = rows["label_class"].value_counts()
    parts = [f"{n} {name}" + ("s" if n > 1 else "") for name, n in counts.items()]
    moving = int(rows["moved"].sum())
    nearest = rows.iloc[0]
    return (f"{', '.join(parts)}; {moving} moving; nearest "
            f"{nearest.label_class} at {nearest.range_m:.0f} m")


def clip_items(clip_id, row, nvidia_root):
    ego = read_member(nvidia_root, row["egomotion_zip"], row["egomotion_member"])
    obstacle = read_member(nvidia_root, row["obstacle_zip"], row["obstacle_member"])
    derived = ego_frame(ego)
    boxes = boxes_world(obstacle, ego) if not obstacle.empty else None

    short = next((r for r in RADAR_PREFERENCE if row.get(f"has_{r}", False)), None)
    radar = extrinsics = None
    if short is not None and row.get("has_radar_extrinsics", False):
        radar = read_member(nvidia_root, row[f"radar_{short}_zip"],
                            row[f"radar_{short}_member"])
        full = pd.read_parquet(os.path.join(nvidia_root,
                                            row["radar_extrinsics_parquet"]))
        if clip_id in full.index.get_level_values(0):
            extrinsics = full.loc[clip_id]
        else:
            radar = None

    # 'none'-profile clips carry no front radar at all. Emitting a radar-evidence
    # answer for them would assert "no radar return" where the truth is "no
    # radar", which is a different claim and one the model would learn to make
    # from the camera alone.
    has_radar = radar is not None and extrinsics is not None

    items = []

    def emit(task, frame, prompt, target):
        items.append({"clip_id": clip_id, "task": task, "frame": int(frame),
                      "prompt": prompt, "target": target,
                      "needs_radar": task in RADAR_EVIDENCE_TASKS})

    for frame in ANCHOR_FRAMES:
        t_s = float(frame - 1)
        if t_s > derived["t"].max():
            break
        here = visible_at(boxes, t_s)
        scan = (radar_scan(radar, extrinsics, SENSOR_NAME[short], t_s, ego, derived)
                if radar is not None and extrinsics is not None else None)
        hits = radar_hits(scan, here if here is None or here.empty else here.head(24))

        listed = None if here is None or here.empty else here.head(MAX_LISTED)

        # 01 -- detection as an enumeration with geometry attached.
        if listed is None:
            answer = "No road users in the forward sector."
        else:
            answer = "; ".join(describe_object(r) for r in listed.itertuples())
        emit("det_objects", frame,
             "List every road user in the forward sector with its class, range "
             "and azimuth.", answer)

        # 02 -- identity: the same objects, keyed and aged.
        if listed is None:
            answer = "No tracks in the forward sector."
        else:
            answer = "; ".join(
                f"#{int(r.track_id)} {r.label_class} {r.range_m:.0f} m "
                f"visible {max(0.0, t_s - r.first_seen_s):.1f} s"
                for r in listed.itertuples())
        emit("track_identity", frame,
             "Give the track id, class, range and age of every object you are "
             "tracking ahead.", answer)

        # 05 -- depth, paired with what the radar can confirm.
        if has_radar:
            if listed is None:
                answer = "Nothing ahead to range."
            else:
                nearest = listed.iloc[0]
                confirmed = [r for r in listed.itertuples() if hits.get(r.track_id)]
                if confirmed:
                    first = confirmed[0]
                    radar_part = (f"nearest radar-confirmed {first.label_class} "
                                  f"at {first.range_m:.0f} m")
                else:
                    radar_part = "no object ahead carries a radar return"
                answer = (f"nearest {nearest.label_class} at "
                          f"{nearest.range_m:.0f} m; {radar_part}")
            emit("depth_range", frame,
                 "How far is the nearest object ahead, and the nearest one the "
                 "radar confirms?", answer)

        # 06 -- moving vs stationary, with the radar evidence spelled out.
        if has_radar:
            if listed is None:
                answer = "Nothing ahead to classify."
            else:
                moving = [describe_object(r, hits, with_motion=False)
                          for r in listed.itertuples() if r.moved]
                static = [describe_object(r, hits, with_motion=False)
                          for r in listed.itertuples() if not r.moved]
                answer = (f"moving: {', '.join(moving) if moving else 'none'}. "
                          f"stationary: {', '.join(static) if static else 'none'}.")
            question = ("Which objects ahead are moving and which are "
                        "stationary? Use the radar Doppler.")
            emit("motion_seg", frame, question, answer)

            # Doppler first, verdict second. Unlike agent_traj the verdict is a
            # threshold on the evidence rather than a physical consequence of it,
            # so this is the weaker of the two chains -- but it is still a chain,
            # and the stated velocities are checkable against the scan.
            if listed is not None:
                evidence = [
                    f"{r.label_class} at {r.range_m:.0f} m: "
                    f"{hits[r.track_id][0]} returns, mean radial "
                    f"{hits[r.track_id][2]:+.1f} m/s"
                    for r in listed.itertuples() if hits.get(r.track_id)]
                if evidence:
                    emit("motion_seg_cot", frame, question, json.dumps({
                        "rationale": "; ".join(evidence) + ". A residual above "
                                     "1 m/s after removing the ego's own motion "
                                     "means the object is moving.",
                        "answer": answer}))

        # 03-1 -- ego waypoints.
        waypoints = ego_waypoints(derived, t_s)
        if waypoints is not None:
            answer = "; ".join(f"+{h:.0f}s ({x:+.1f}, {y:+.1f})"
                               for h, (x, y) in zip(HORIZON_S, waypoints))
            emit("plan_ego", frame,
                 "Predict the ego vehicle's path over the next 3 seconds as "
                 "(x, y) offsets in metres.", answer)

        # 04 -- the scene 3 s on, given the action the ego takes.
        future = visible_at(boxes, t_s + HORIZON_S[-1])
        if waypoints is not None and future is not None:
            emit("world_model", frame,
                 f"The ego vehicle will {ego_action(derived, t_s)} over the next "
                 f"3 seconds. What will the forward scene look like then?",
                 scene_summary(future.head(MAX_LISTED)))

        # 03-2 -- one agent, forward in time. The nearest mover is the agent
        # worth predicting; a parked car's answer is its current position.
        if listed is not None and waypoints is not None:
            movers = [r for r in listed.itertuples() if r.moved]
            if movers:
                agent = movers[0]
                path = []
                for horizon in HORIZON_S:
                    later = visible_at(boxes, t_s + horizon)
                    if later is None or later.empty:
                        break
                    match = later[later["track_id"] == agent.track_id]
                    if match.empty:
                        break
                    hit = match.iloc[0]
                    path.append(f"+{horizon:.0f}s {hit.range_m:.0f} m az "
                                f"{hit.azimuth_deg:+.0f} deg")
                if path:
                    question = (f"Track #{int(agent.track_id)} is a "
                                f"{agent.label_class} at {agent.range_m:.0f} m, "
                                f"azimuth {agent.azimuth_deg:+.0f} deg. Where "
                                f"will it be over the next 3 seconds?")
                    emit("agent_traj", frame, question, "; ".join(path))

                    # The same question with the radar reading required first.
                    # This is the only task where a radar quantity sits strictly
                    # upstream of the answer: radial velocity times time is the
                    # change in range, so a model that cannot read the Doppler
                    # gets the rationale wrong before it gets the answer wrong,
                    # and the two can be scored separately. Emitted only when the
                    # agent actually carries returns -- inventing a velocity for
                    # an object the radar never saw is the failure this is meant
                    # to detect, not one to train in.
                    hit = hits.get(agent.track_id)
                    if hit and hit[0] >= 2:
                        n_points, _, radial = hit
                        closing = ("closing" if radial < -0.5 else
                                   "receding" if radial > 0.5 else
                                   "holding its range")
                        emit("agent_traj_cot", frame, question, json.dumps({
                            "rationale": (
                                f"The radar puts {n_points} returns on track "
                                f"#{int(agent.track_id)} at {agent.range_m:.0f} m, "
                                f"median radial velocity {radial:+.1f} m/s, so it "
                                f"is {closing}."),
                            "answer": "; ".join(path)}))
    return items


def process(args_tuple):
    clip_id, row, nvidia_root = args_tuple
    try:
        return clip_items(clip_id, row, nvidia_root)
    except Exception as exc:
        return [{"clip_id": clip_id, "task": None, "frame": -1, "prompt": "",
                 "target": f"{type(exc).__name__}: {str(exc)[:80]}"}]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clips", default="all", choices=("qa", "all"))
    ap.add_argument("--nvidia-root", default=paths.NVIDIA_ROOT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    clips = pd.read_parquet(os.path.join(paths.COMMON_DIR, "nvidia_clips.parquet"))
    if args.clips == "qa":
        import glob
        ids = [os.path.basename(f)[:-5] for f in sorted(glob.glob(
            os.path.join(paths.SPLIT_ROOT, "10_radar_vision_qa/qa/*.json")))]
        ids = [i for i in ids if i in clips.index]
    else:
        ids = list(clips.index)
    if args.limit:
        ids = ids[: args.limit]
    out_path = args.out or os.path.join(paths.COMMON_DIR,
                                        "instruct_items_tasks01_06.parquet")

    usable = clips.loc[ids]
    usable = usable[usable["has_egomotion"].fillna(False)
                    & usable["has_obstacle"].fillna(False)]
    log(f"{len(usable):,} clips x {len(ANCHOR_FRAMES)} anchor frames")

    tasks = [(cid, usable.loc[cid].to_dict(), args.nvidia_root)
             for cid in usable.index]
    rows, errors = [], 0
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, t) for t in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result and result[0].get("task") is None:
                errors += 1
            else:
                rows.extend(result)
            if i % 500 == 0 or i == len(tasks):
                rate = i / (time.monotonic() - started)
                eta = (len(tasks) - i) / rate / 60 if rate else float("nan")
                log(f"  {i}/{len(tasks)}  {rate:.1f} clip/s  ETA {eta:.1f} min")

    frame = pd.DataFrame(rows)
    frame["split"] = frame["clip_id"].map(clips["split"])
    frame.to_parquet(out_path, index=False)
    log(f"wrote {out_path}  rows={len(frame):,}  "
        f"({os.path.getsize(out_path)/1e6:.1f} MB)")
    if errors:
        log(f"clips that failed: {errors}")
    log("")
    for task, group in frame.groupby("task"):
        chars = group["target"].str.len()
        log(f"  {task:16s} {len(group):>9,}  target chars p50 {chars.median():.0f} "
            f"p99 {chars.quantile(0.99):.0f}")
        log(f"      e.g. {group['target'].iloc[0][:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
