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

  det_objects    01  every road user ahead: class and position. Motion is not
                     asked here: the label is a track-level fact -- did this
                     object move over the clip -- and the input is one instant.
                     Measured on the objects the radar can see at all, the
                     instantaneous Doppler agrees with that label 85.8% of the
                     time, so one answer in seven was unreachable from the
                     input. Motion belongs to task 06, which is given the
                     evidence for it
  track_step     02  tracking as it is normally posed: the detections at
                     t-4..t-1 go in, the objects at t come out, and an object
                     must keep the id it already has
  plan_ego       03-1 future ego waypoints as (x, y) offsets in the current ego
                     frame, forward-positive
  agent_traj     03-2 one agent's future position over the next 3 s, in
                     whichever form the instruction asks for: (x, y) metres
                     in the ego frame at the question, range/azimuth
                     relative to the ego at each horizon, or an image box
  world_model    04  the scene 3 s ahead, conditioned on the ego action taken
  (04 world_model and 05 depth_range are held back from this build)
  motion_seg     06  moving vs stationary, from two perpendicular residuals.
                     06.1 names each object by range and azimuth, 06.2 by its
                     image box; the input, the label and the reasoning are the
                     same and only the way an object is pointed at changes

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
from scipy.spatial.transform import Rotation

from . import paths
from .geometry import (camera_azimuth, doppler_residual, extrinsics_to_matrix,
                       image_bbox, interpolate_pose, points_in_box,
                       rig_to_world, sensor_to_rig, spherical_to_sensor_xyz)

ANCHOR_FRAMES = (6, 11, 16)      # 1-indexed; t = frame - 1 seconds
N_CLIP_FRAMES = 20               # the 1 Hz grid the model is shown

# `det_objects` alone is not a prediction: it asks what is ahead at one instant,
# so it is not bound by the 3 s horizon that fixes the shared anchors, and it
# takes a single video frame plus the radar scans just before that instant
# rather than the whole clip. Six evenly spaced instants therefore cost the same
# context as one of the old three did and double the items per clip.
DET_ANCHOR_S = (3, 6, 9, 12, 15, 18)

# Task 02 is about identity across time, so it keeps the whole clip as input and
# emits one answer per clip rather than one per instant.
CAMERA = "camera_front_wide_120fov"
TRACK_MIN_FRAMES = 2      # a single 1 Hz sighting is label noise, not a track
MAX_TRACKS = 10           # median clip has 38 tracks lasting >= 2 frames
BBOX_SCALE = 1000         # normalised, so the answer does not encode a resolution

# Task 02.2: tracking as it is normally posed -- see t-n..t-1, then name the
# objects at t. The window is five seconds; the answer is the last instant.
STEP_WINDOW_S = 5
# Every second, not every third. In a rollout the history is the model's
# own previous outputs, so the gap between anchors *is* the gap between
# history entries; anchors 3 s apart with a 1 s history would train one
# prompt shape and evaluate another.
STEP_ANCHOR_S = tuple(range(STEP_WINDOW_S, 20))

# Task 03.1: the ego's own path. Every two seconds from 3 to 17 -- 17 is the
# last instant whose 3 s horizon still lands inside a 20 s clip, and two seconds
# is where adjacent answers stop being the same problem: measured, they differ
# by a median 1.42 m at that spacing against 0.72 m at one second, and the
# reward's half-credit point is 1 m.
PLAN_ANCHOR_S = (3, 5, 7, 9, 11, 13, 15, 17)

# Task 03.2 rides the same anchors and the same two-second window: predicting
# where another object goes needs its velocity, which the Doppler measures
# instantaneously, not twenty seconds of context.
LEAVES = "leaves the forward sector"

# Classes from rarest to most common, measured over 266,850 labelled objects:
# automobile 84.5%, person 10.3%, heavy_truck 1.6%, trailer 0.9%,
# protruding_object 0.8%, rider 0.8%, bus 0.5%, other_vehicle 0.4%,
# stroller 0.2%, animal 0.1%. Used to pick which agent task 03.2 asks about --
# see `pick_agent`.
CLASS_RARITY = ("animal", "stroller", "other_vehicle", "bus", "rider",
                "protruding_object", "trailer", "heavy_truck", "person",
                "automobile")


def pick_agent(movers, turn):
    """Which moving object task 03.2 asks about, alternating by anchor.

    Always taking the nearest one made 73-78% of the questions about an
    automobile, because that is what is nearest in 73-78% of scenes. The pool
    is better than that -- among movers, 69.9% automobile, 20.9% person, 3.6%
    rider, 2.4% heavy_truck -- and asking about the rarest class present
    instead brings the questions to 54.2% automobile, 21.6% person, 7.3%
    rider, 6.7% heavy_truck, 5.2% trailer.

    Neither rule alone is right. Always-rarest never asks about the car
    directly ahead in a busy scene, which is the case a planner cares about
    most; always-nearest never asks about the cyclist. So the anchors
    alternate, and both are covered in the same clip.
    """
    if turn % 2 == 0:
        return movers[0]
    order = {c: i for i, c in enumerate(CLASS_RARITY)}
    return min(movers, key=lambda r: (order.get(r.label_class, 0),
                                      r.range_m))

# Task 06. Every two seconds: measured, only 3.7% of objects change between
# moving and stationary from one second to the next, so a 1 Hz grid would be
# 96% the same question asked again.
MOTION_ANCHOR_S = (2, 4, 6, 8, 10, 12, 14, 16, 18)
MOVING_MS = 1.0           # world-frame speed above which an object is moving
CAMERA_RESIDUAL_DEG = 1.0 # bearing drift a stationary object cannot explain
STEP_HISTORY = 4          # how many past 1 Hz frames of detections are shown
HORIZON_S = (1.0, 2.0, 3.0)      # prediction offsets for 03-1, 03-2, 04
BOX_TIME_WINDOW_S = 0.15         # a 10 Hz label stream, so this catches 1-2 obs
DOPPLER_THRESHOLD_MS = 1.0
STATIONARY_M = 2.0
BOX_MARGIN = 1.3
MAX_LISTED = 8                   # objects per answer, nearest first

# Answers that cite radar evidence, and so may only be emitted for a clip that
# actually carries a front radar.
RADAR_EVIDENCE_TASKS = ("depth_range", "motion_seg", "motion_seg_cot",
                        "agent_traj_cot", "det_objects_azdeg_cot",
                        "det_objects_3dbbox_cot", "depth_range_cot")

# What tells a CoT item apart from its plain twin. Every `_cot` task used to
# carry its plain twin's prompt verbatim -- measured over 3,000 pairs per task,
# 100% identical -- so the same input had two different right answers and no
# way to tell which was wanted. The model resolved it by picking one: evaluated
# at 4,400 steps it answered `plan_ego` and `agent_traj` in CoT format 100% of
# the time and `det_objects` 93%, scoring near zero on the plain tasks for a
# reason that has nothing to do with what it can see.
#
# `qa` was the exception and the control. Its two forms did differ -- "Answer
# with the letter" against "Reason step by step, then answer with the letter" --
# and it is the one task that split cleanly, 0% against 100%.
#
# Appended rather than prepended so the plain instruction keeps its own opening
# word, and defined once here because five separate name lists drifting apart
# is exactly what produced the defect above.
COT_INSTRUCTION = " Reason step by step first, then give the answer."


def cot_prompt(question):
    return question.rstrip() + COT_INSTRUCTION


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


def object_evidence(scan, rows):
    """Per-box radar evidence, geometry included: how many returns, how many
    moving, their radial velocity, and where the sensor puts them."""
    out = {}
    if scan is None or rows is None or len(rows) == 0:
        return out
    rig = scan["rig"]
    rng = np.hypot(rig[:, 0], rig[:, 1])
    az = np.degrees(np.arctan2(rig[:, 1], rig[:, 0]))
    for box in rows.itertuples():
        inside = points_in_box(
            rig, np.array([box.center_x, box.center_y, box.center_z]),
            [box.size_x, box.size_y, box.size_z],
            [box.orientation_x, box.orientation_y,
             box.orientation_z, box.orientation_w], BOX_MARGIN)
        if not inside.any():
            continue
        movers = inside & scan["moving"]
        radial = scan["radial"][movers] if movers.any() else scan["radial"][inside]
        out[box.track_id] = {
            "hit_points": float(inside.sum()),
            "hit_moving": float(scan["moving"][inside].sum()),
            "hit_radial": float(np.median(radial)),
            "hit_range_m": float(np.median(rng[inside])),
            "hit_azimuth_deg": float(np.median(az[inside])),
        }
    return out


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


def ego_controls(derived, t_s):
    """The same trajectory as a control sequence: future speed and yaw rate.

    The waypoint form asks where the car will be; this asks what it will be
    doing. Two scalars per horizon rather than a rotated coordinate pair, which
    is what a controller consumes and what does not depend on getting a frame
    transform right.
    """
    out = []
    for horizon in HORIZON_S:
        j = at_time(derived, t_s + horizon)
        if derived["t"][j] < t_s + horizon - 0.35:
            return None
        out.append((float(derived["speed"][j]),
                    float(np.degrees(derived["yaw_rate"][j]))))
    return out


def pose_at(derived, t_s):
    """Ego position and heading at `t_s`, interpolated between samples.

    `at_time` returns the nearest sample instead, and egomotion arrives at
    10 Hz: up to 50 ms of error, which at the median 8.5 m/s is 0.43 m. It is
    measurable -- transforming an object's own world coordinates back into the
    ego frame at the moment it was observed should return the coordinates the
    label already carries, and with nearest-sample poses it missed by a median
    0.32 m. That is a third of the displacement error `plan_ego_xy` is scored
    on, contributed by the reference rather than by the model, and the (x, y)
    trajectory forms are scored in metres, so it goes straight into ADE.

    Yaw is interpolated on the wrapped difference so a heading crossing +/-pi
    does not swing the long way round.
    """
    t = derived["t"]
    j = int(np.clip(np.searchsorted(t, t_s), 1, len(t) - 1))
    t0, t1 = t[j - 1], t[j]
    w = 0.0 if t1 <= t0 else float(np.clip((t_s - t0) / (t1 - t0), 0.0, 1.0))
    xy = derived["xy"][j - 1] + w * (derived["xy"][j] - derived["xy"][j - 1])
    y0, y1 = derived["yaw"][j - 1], derived["yaw"][j]
    return xy, y0 + w * ((y1 - y0 + np.pi) % (2 * np.pi) - np.pi)


NAV_DEG = 15.0        # heading change over the horizon that makes it a turn


def nav_command(derived, t_s):
    """LEFT, RIGHT or STRAIGHT over the next 3 s -- the route, not the path.

    A real navigation command comes from a map and a destination. This release
    carries neither: `labels/` holds egomotion and obstacles and nothing else,
    so the only place to get one is the ego's own future heading -- the thing
    `plan_ego` is asked to predict. That makes it a leak, and the size of the
    leak was measured rather than assumed. Against a ridge regression that
    already sees the speed and yaw rate the prompt carries, adding the command
    improves ADE by 3.3% overall: 1.7% on the 86.6% of anchors that are
    STRAIGHT, 8.1% on LEFT and 12.0% on RIGHT.

    It is worth that 3.3% because without it the task is ill-posed. At an
    intersection "where will the car go" has three right answers and the scene
    does not say which, so an unconditioned model is trained to predict the
    average of them -- a path it would never drive. Conditioning is also how a
    planner is actually driven. What must not be forgotten is that an error
    reported with the command given is "error once the intent is known", not
    "error predicting where the car goes", and the two are different claims.

    The finer `motion_intent` labels (STOP, CHANGE_LANE_*, TURN_*) are
    deliberately not used as input. Measured the same way, knowing STOP halves
    the error on those anchors -- 1.546 m to 0.777 m -- which is not a hint
    about the route, it is most of the answer.
    """
    i, j = at_time(derived, t_s), at_time(derived, t_s + HORIZON_S[-1])
    delta = np.degrees((derived["yaw"][j] - derived["yaw"][i] + np.pi)
                       % (2 * np.pi) - np.pi)
    return "LEFT" if delta > NAV_DEG else "RIGHT" if delta < -NAV_DEG else "STRAIGHT"


def agent_offset(derived, t_s, hit):
    """An agent's future position in the ego frame at `t_s`, forward-positive.

    Not the frame the label is stored in. `obstacle.offline` gives a position
    relative to the rig at the moment of observation, so `center_x` at t+3s
    answers "where will it be relative to where I will then be" -- which cannot
    be used without the ego's own future path, and is not what a trajectory
    benchmark means by (x, y). Fixing the frame at `t_s` makes the answer
    directly comparable to `plan_ego_xy`, whose offsets are in the same frame,
    and it is the same transform `ego_waypoints` applies -- `wx`/`wy` are
    already world coordinates, so only the ego pose at `t_s` is needed.
    """
    origin, yaw = pose_at(derived, t_s)
    cos, sin = np.cos(yaw), np.sin(yaw)
    dx, dy = hit.wx - origin[0], hit.wy - origin[1]
    return cos * dx + sin * dy, -sin * dx + cos * dy


def ego_waypoints(derived, t_s):
    """Future ego offsets in the ego frame at `t_s`, forward-positive."""
    origin, yaw = pose_at(derived, t_s)
    cos, sin = np.cos(yaw), np.sin(yaw)
    out = []
    for horizon in HORIZON_S:
        j = at_time(derived, t_s + horizon)
        if derived["t"][j] < t_s + horizon - 0.35:      # clip ends early
            return None
        delta = pose_at(derived, t_s + horizon)[0] - origin
        out.append((cos * delta[0] + sin * delta[1],
                    -sin * delta[0] + cos * delta[1]))
    return out


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------

def box_yaw(row):
    """Heading in the rig frame, degrees, positive to the left of the ego.

    The label carries a quaternion, which is four numbers of which three sit
    near zero for a vehicle on level ground, and which has a sign ambiguity --
    q and -q are the same rotation, so a model would be asked to reproduce a
    choice nobody makes consistently. One yaw angle carries what the box needs
    and is scored by an angular difference that knows 179 and -179 are close.
    """
    quat = [row.orientation_x, row.orientation_y,
            row.orientation_z, row.orientation_w]
    heading = Rotation.from_quat(quat).apply([1.0, 0.0, 0.0])
    return float(np.degrees(np.arctan2(heading[1], heading[0])))


def describe_object_xyz(row, hits=None, with_motion=True):
    """A 3D box in rig coordinates: centre, extent and heading.

    x forward, y left, z up. The polar form loses precision with distance --
    one degree of azimuth is 0.6 m at 34 m and 1.7 m at 100 m -- and the
    matcher converts back to Cartesian to score anyway, so this format skips a
    conversion rather than adding one. `z` is included because height separates
    a vehicle on an overpass from one on the road beneath it, which range and
    bearing alone cannot.

    Extent and heading are here because without them this was a point, not a
    box, whatever the task was called. A centre says where something is; a
    driving stack also needs how much room it takes and which way it faces,
    and both are in the labels already -- `size_*` and `orientation_*` -- so
    emitting only the centre was throwing away supervision that cost nothing.
    """
    # `size` and `deg` are named rather than left to the instruction. Three
    # bare numbers after a coordinate triple do not say which is length and
    # which is height, and the rest of these answers label every quantity --
    # "34 m az +13 deg". Three extra tokens per object buys an answer that
    # reads correctly on its own.
    text = (f"{row.label_class} ({row.center_x:.1f}, {row.center_y:.1f}, "
            f"{row.center_z:.1f}) size {row.size_x:.1f}x{row.size_y:.1f}x"
            f"{row.size_z:.1f} m yaw {box_yaw(row):+.0f} deg")
    if with_motion:
        text += " moving" if row.moved else " stationary"
    if hits is not None:
        n = hits.get(row.track_id, (0, 0, 0.0))[0]
        text += (f" ({n} radar return{'s' if n > 1 else ''})" if n
                 else " (no radar return)")
    return text


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

    # 'none'-profile clips carry no front radar at all -- 17,130 of 177,891, and
    # 9.6% of what this function used to emit. Nothing here is dropped for want
    # of a target: every answer is built from the obstacle labels and egomotion,
    # so a blind clip produced a perfectly valid one. What it could not produce
    # was an honest rationale. The evidence clauses came out as "no radar
    # return" for every object, which asserts the sensor looked and saw nothing
    # where the truth is that there is no sensor -- a different claim, and the
    # one a model would learn to make from the camera alone.
    #
    # This gate used to be computed and never read, so those items shipped.
    if radar is None or extrinsics is None:
        return []

    items = []

    def emit(task, frame, prompt, target, strat=""):
        """`strat` names a subgroup the sampler may want to balance.

        Some tasks have an answer distribution so lopsided that a model scores
        well by ignoring the part of the prompt that distinguishes the cases --
        88.2% of `plan_ego` anchors are STRAIGHT, so a model that never reads
        the navigation command loses almost nothing. Writing the subgroup here
        lets the loader balance exposure instead of deleting items: cutting to
        1:1:1 would throw away 83% of the task, and the ratio could not be
        changed without rebuilding this file.
        """
        items.append({"clip_id": clip_id, "task": task, "frame": int(frame),
                      "prompt": prompt, "target": target, "strat": strat,
                      "needs_radar": task in RADAR_EVIDENCE_TASKS})

    # 01 runs on its own instants. It asks what is ahead now, not what happens
    # next, so the 3 s horizon that pins the shared anchors does not apply, and
    # its single-frame input means six instants cost less context than one of
    # the shared anchors did.
    for seconds in DET_ANCHOR_S:
        t_s = float(seconds)
        if t_s > derived["t"].max():
            break
        here = visible_at(boxes, t_s)
        listed = None if here is None or here.empty else here.head(MAX_LISTED)
        scan = (radar_scan(radar, extrinsics, SENSOR_NAME[short], t_s, ego, derived)
                if radar is not None and extrinsics is not None else None)
        hits = radar_hits(scan, here if here is None or here.empty else here.head(24))
        evidence = (object_evidence(scan, listed)
                    if scan is not None and listed is not None else {})
        evidence_xy = {k: (v["hit_range_m"], v["hit_azimuth_deg"])
                       for k, v in evidence.items()}
        camera = camera_model(clip_id, row, nvidia_root)
        frame = seconds + 1                      # 1-indexed, t = frame - 1
        question = ("List every road user in the forward sector with its class, "
                    "range and azimuth.")
        if listed is None:
            answer = "No road users in the forward sector."
        else:
            answer = "; ".join(describe_object(r, with_motion=False)
                               for r in listed.itertuples())
        emit("det_objects_azdeg", frame, question, answer)

        question_xyz = ("List every road user in the forward sector as a 3D "
                        "box: class, centre (x, y, z) in metres with x "
                        "forward, y left and z up, then size as "
                        "length x width x height in metres, then yaw in "
                        "degrees, positive to the left of the ego heading.")
        if listed is None:
            answer_xyz = "No road users in the forward sector."
        else:
            answer_xyz = "; ".join(describe_object_xyz(r, with_motion=False)
                                   for r in listed.itertuples())
        emit("det_objects_3dbbox", frame, question_xyz, answer_xyz)

        # What each sensor actually contributes, per object, before the answer
        # fuses them. Measured over 2,700-4,200 objects: the camera recovers a
        # bearing from the box centre to a median 0.94 deg and gives no range at
        # all; the radar puts range within a median 0.70 m but only illuminates
        # 55% of the objects listed -- 85.7% of heavy trucks and 20.2% of
        # people. So the complementarity is coverage against range, not the
        # textbook angle-against-range, and the rationale says so object by
        # object rather than asserting it in general.
        if listed is not None and camera is not None:
            rot, translation, model = camera
            clauses = []
            for r in listed.itertuples():
                box = image_bbox([r.center_x, r.center_y, r.center_z],
                                 [r.size_x, r.size_y, r.size_z],
                                 [r.orientation_x, r.orientation_y,
                                  r.orientation_z, r.orientation_w],
                                 rot, translation, model)
                if box is None:
                    continue
                seen = camera_azimuth(((box[0] + box[2]) / 2,
                                       (box[1] + box[3]) / 2), model, rot)
                hit = hits.get(r.track_id)
                if hit and evidence_xy.get(r.track_id):
                    ex, ey = evidence_xy[r.track_id]
                    clauses.append(
                        f"{r.label_class}: camera az {seen:+.0f} deg, radar "
                        f"{hit[0]} return{'s' if hit[0] > 1 else ''} at "
                        f"{ex:.0f} m az {ey:+.0f} deg")
                else:
                    clauses.append(f"{r.label_class}: camera az {seen:+.0f} deg, "
                                   f"no radar return")
            if clauses:
                rationale = ("; ".join(clauses) + ". Range comes from the radar "
                             "where it has a return and from the camera alone "
                             "where it does not.")
                emit("det_objects_azdeg_cot", frame, cot_prompt(question),
                     json.dumps({"rationale": rationale, "answer": answer}))
                emit("det_objects_3dbbox_cot", frame, cot_prompt(question_xyz),
                     json.dumps({"rationale": rationale, "answer": answer_xyz}))

    for frame in ANCHOR_FRAMES:
        t_s = float(frame - 1)
        if t_s > derived["t"].max():
            break
        here = visible_at(boxes, t_s)
        scan = (radar_scan(radar, extrinsics, SENSOR_NAME[short], t_s, ego, derived)
                if radar is not None and extrinsics is not None else None)
        hits = radar_hits(scan, here if here is None or here.empty else here.head(24))

        listed = None if here is None or here.empty else here.head(MAX_LISTED)

        waypoints = ego_waypoints(derived, t_s)

    # 06 -- moving or stationary, decided from two perpendicular residuals.
    # 06.1 azdeg names objects in the world, 06.2 bbox in the image.
    speeds = world_speeds(boxes)
    camera_06 = camera_model(clip_id, row, nvidia_root)
    for seconds in MOTION_ANCHOR_S:
        t_s = float(seconds)
        if t_s > derived["t"].max():
            break
        here = visible_at(boxes, t_s)
        if here is None or here.empty:
            continue
        listed = here.head(MAX_LISTED)
        scan = (radar_scan(radar, extrinsics, SENSOR_NAME[short], t_s, ego, derived)
                if radar is not None and extrinsics is not None else None)
        evidence = object_evidence(scan, listed) if scan is not None else {}
        earlier = visible_at(boxes, t_s - 1.0)
        was = ({r.track_id: r for r in earlier.itertuples()}
               if earlier is not None and not earlier.empty else {})

        i = at_time(derived, t_s)
        v_world = np.array([derived["velocity"][i, 0],
                            derived["velocity"][i, 1], 0.0])
        j = min(i, len(derived["quat"]) - 1)
        v_rig = Rotation.from_quat(derived["quat"][j]).as_matrix().T @ v_world
        ego_speed = float(np.hypot(v_world[0], v_world[1]))
        ego_yaw = float(np.degrees(derived["yaw_rate"][i]))

        moving, still, clauses = [], [], []
        for r in listed.itertuples():
            if r.track_id not in speeds:
                continue
            times, series = speeds[r.track_id]
            speed = float(series[int(np.argmin(np.abs(times - t_s)))])
            (moving if speed > MOVING_MS else still).append(r)

            hit = evidence.get(r.track_id)
            ev = motion_evidence(r, was.get(r.track_id), derived, t_s, v_rig, hit)
            where = f"{r.label_class} at {r.range_m:.0f} m az {r.azimuth_deg:+.0f} deg"
            said = []
            if ev["measured_radial"] is not None:
                gap = abs(ev["measured_radial"] - ev["expected_radial"])
                said.append(f"radar expects {ev['expected_radial']:+.1f} m/s if "
                            f"stationary, measures {ev['measured_radial']:+.1f} "
                            f"(residual {gap:.1f})")
            else:
                said.append("no radar return")
            if ev["expected_az"] is not None:
                drift = abs(ev["measured_az"] - ev["expected_az"])
                said.append(f"camera expects az {ev['expected_az']:+.0f} deg if "
                            f"stationary, sees {ev['measured_az']:+.0f} "
                            f"(residual {drift:.1f} deg)")
            clauses.append(f"{where}: " + ", ".join(said))

        if not moving and not still:
            continue
        rationale = (f"The ego is travelling at {ego_speed:.1f} m/s, yaw "
                     f"{ego_yaw:+.1f} deg/s. " + "; ".join(clauses) +
                     f". A radial residual above {MOVING_MS:.0f} m/s or a "
                     f"bearing residual above {CAMERA_RESIDUAL_DEG:.0f} deg "
                     f"means the object is moving; the Doppler is blind to an "
                     f"object crossing the ego's path and the bearing is blind "
                     f"to one approaching head-on, so either is enough.")

        for form in ("azdeg", "bbox"):
            def render(group):
                out = []
                for r in group:
                    if form == "bbox":
                        if camera_06 is None:
                            return None
                        rot, translation, model = camera_06
                        box = image_bbox([r.center_x, r.center_y, r.center_z],
                                         [r.size_x, r.size_y, r.size_z],
                                         [r.orientation_x, r.orientation_y,
                                          r.orientation_z, r.orientation_w],
                                         rot, translation, model)
                        if box is None:
                            continue
                        width, height = model["resolution"]
                        out.append(f"{r.label_class} "
                                   f"[{round(box[0]/width*BBOX_SCALE)}, "
                                   f"{round(box[1]/height*BBOX_SCALE)}, "
                                   f"{round(box[2]/width*BBOX_SCALE)}, "
                                   f"{round(box[3]/height*BBOX_SCALE)}]")
                    else:
                        out.append(f"{r.label_class} {r.range_m:.0f} m az "
                                   f"{r.azimuth_deg:+.0f} deg")
                return ", ".join(out) if out else "none"
            a, b_ = render(moving), render(still)
            if a is None or b_ is None:
                continue
            how = ("as range in metres and azimuth in degrees" if form == "azdeg"
                   else "as an image bounding box [x1, y1, x2, y2] normalised "
                        "to 0-1000")
            question = (f"Which objects ahead are moving and which are "
                        f"stationary? Use the radar Doppler and the camera. "
                        f"Identify each {how}.")
            answer = f"moving: {a}. stationary: {b_}."
            emit(f"motion_seg_{form}", seconds + 1, question, answer)
            emit(f"motion_seg_{form}_cot", seconds + 1, cot_prompt(question),
                 json.dumps({"rationale": rationale, "answer": answer}))

    # 03.1 -- the ego's own path, on its own anchors, in two representations
    # the instruction picks between: where the car will be, and what it will be
    # doing. Measured, constant-velocity extrapolation misses by a median
    # 1.09 m and clears the reward's 1 m half-credit point only 47.9% of the
    # time, so the scene genuinely matters here and this is not a task that
    # reading the speedometer solves.
    for seconds in PLAN_ANCHOR_S:
        t_s = float(seconds)
        if t_s > derived["t"].max():
            break
        way = ego_waypoints(derived, t_s)
        controls = ego_controls(derived, t_s)
        if way is None or controls is None:
            continue
        i = at_time(derived, t_s)
        speed = float(np.hypot(derived["velocity"][i, 0],
                               derived["velocity"][i, 1]))
        rationale = (f"The ego vehicle is travelling at {speed:.1f} m/s and "
                     f"will {ego_action(derived, t_s)}. At that speed it covers "
                     f"about {speed:.0f} m per second.")
        # The route the car is following. Stated first because it is a
        # condition on the question, not a fact about the scene.
        command = nav_command(derived, t_s)
        asked_of = f"Navigation command: {command}. "
        forms = {
            "xy": (asked_of + "Predict the ego vehicle's path over the next 3 "
                   "seconds as (x, y) offsets in metres.",
                   "; ".join(f"+{h:.0f}s ({x:+.1f}, {y:+.1f})"
                             for h, (x, y) in zip(HORIZON_S, way))),
            "control": (asked_of + "Predict the ego vehicle's speed and yaw "
                        "rate over the next 3 seconds.",
                        "; ".join(f"+{h:.0f}s {v:.1f} m/s, yaw {w:+.1f} deg/s"
                                  for h, (v, w) in zip(HORIZON_S, controls))),
        }
        for form, (question, answer) in forms.items():
            emit(f"plan_ego_{form}", seconds + 1, question, answer, command)
            emit(f"plan_ego_{form}_cot", seconds + 1, cot_prompt(question),
                 json.dumps({"rationale": rationale, "answer": answer}), command)

    # 03.2 -- one other object, forward in time. Two representations again:
    # where it will be in the world, and where it will appear in the image.
    #
    # A track that leaves the forward sector says so, rather than the answer
    # simply stopping. Measured, the agent survives all three horizons only
    # 51.4% of the time, so "it will be gone" is the more common outcome after
    # two seconds and is a useful prediction in its own right; truncating the
    # list instead taught the model that a short answer is always safe.
    for turn, seconds in enumerate(PLAN_ANCHOR_S):
        t_s = float(seconds)
        if t_s > derived["t"].max():
            break
        here = visible_at(boxes, t_s)
        if here is None or here.empty:
            continue
        listed = here.head(MAX_LISTED)
        movers = [r for r in listed.itertuples() if r.moved]
        if not movers:
            continue
        agent = pick_agent(movers, turn)
        scan = (radar_scan(radar, extrinsics, SENSOR_NAME[short], t_s, ego, derived)
                if radar is not None and extrinsics is not None else None)
        evidence = object_evidence(scan, listed) if scan is not None else {}
        camera = camera_model(clip_id, row, nvidia_root)

        future, gone = [], False
        for horizon in HORIZON_S:
            later = visible_at(boxes, t_s + horizon)
            match = (None if later is None or later.empty
                     else later[later["track_id"] == agent.track_id])
            if match is None or match.empty:
                gone = True
                future.append((horizon, None))
                break
            future.append((horizon, match.iloc[0]))
        if not future:
            continue

        def render(form):
            parts = []
            for horizon, hit in future:
                if hit is None:
                    parts.append(f"+{horizon:.0f}s {LEAVES}")
                    continue
                if form == "bbox":
                    if camera is None:
                        return None
                    rot, translation, model = camera
                    box = image_bbox([hit.center_x, hit.center_y, hit.center_z],
                                     [hit.size_x, hit.size_y, hit.size_z],
                                     [hit.orientation_x, hit.orientation_y,
                                      hit.orientation_z, hit.orientation_w],
                                     rot, translation, model)
                    if box is None:
                        parts.append(f"+{horizon:.0f}s {LEAVES}")
                        continue
                    width, height = model["resolution"]
                    parts.append(
                        f"+{horizon:.0f}s [{round(box[0]/width*BBOX_SCALE)}, "
                        f"{round(box[1]/height*BBOX_SCALE)}, "
                        f"{round(box[2]/width*BBOX_SCALE)}, "
                        f"{round(box[3]/height*BBOX_SCALE)}]")
                elif form == "xy":
                    x, y = agent_offset(derived, t_s, hit)
                    parts.append(f"+{horizon:.0f}s ({x:+.1f}, {y:+.1f})")
                else:
                    parts.append(f"+{horizon:.0f}s {hit.range_m:.0f} m az "
                                 f"{hit.azimuth_deg:+.0f} deg")
            return "; ".join(parts)

        # Three ways to say where an object goes, and they do not share a
        # frame -- so each instruction states its own, or the same question has
        # two right answers and no way to tell which was wanted.
        #
        # `xy` is the form a trajectory benchmark means: metres in the ego
        # frame at the moment of the question, the same frame `plan_ego_xy`
        # uses, so an ego path and an agent path can be drawn on one picture
        # and the error is a displacement in metres. `azdeg` and `bbox` are
        # both relative to where the ego will be at each horizon -- a bearing
        # and an image box have no other meaning -- which makes them the
        # "where do I look" forms rather than the "where will it be" one.
        #
        # Polar is kept because it is what the radar measures, but it is a poor
        # error metric on its own: five degrees of azimuth is 0.44 m at 5 m and
        # 6.97 m at 80 m, so one `az_mae` covers answers that are a car's width
        # apart and answers that are two lanes apart.
        asked = {"azdeg": ("as range in metres and azimuth in degrees relative "
                           "to the ego at that time"),
                 "bbox": ("as an image bounding box [x1, y1, x2, y2] normalised "
                          "to 0-1000"),
                 "xy": ("as (x, y) offsets in metres from the ego's position "
                        "now, x forward and y to the left")}
        seen_az, now_box = None, None
        if camera is not None:
            rot, translation, model = camera
            box = image_bbox([agent.center_x, agent.center_y, agent.center_z],
                             [agent.size_x, agent.size_y, agent.size_z],
                             [agent.orientation_x, agent.orientation_y,
                              agent.orientation_z, agent.orientation_w],
                             rot, translation, model)
            if box:
                seen_az = camera_azimuth(((box[0] + box[2]) / 2,
                                          (box[1] + box[3]) / 2), model, rot)
                width, height = model["resolution"]
                now_box = (f"[{round(box[0]/width*BBOX_SCALE)}, "
                           f"{round(box[1]/height*BBOX_SCALE)}, "
                           f"{round(box[2]/width*BBOX_SCALE)}, "
                           f"{round(box[3]/height*BBOX_SCALE)}]")
        hit = evidence.get(agent.track_id)
        clauses = [f"#{int(agent.track_id)} {agent.label_class} is at "
                   f"{agent.range_m:.0f} m az {agent.azimuth_deg:+.0f} deg now"]
        if seen_az is not None:
            clauses.append(f"camera az {seen_az:+.0f} deg")
        if hit:
            clauses.append(f"radar {int(hit['hit_points'])} returns at "
                           f"{hit['hit_range_m']:.0f} m, radial "
                           f"{hit['hit_radial']:+.1f} m/s")
            move = ("closing" if hit["hit_radial"] < -0.5 else
                    "receding" if hit["hit_radial"] > 0.5 else
                    "holding its range")
            clauses.append(f"so it is {move}: radial velocity times time is the "
                           f"change in range")
        else:
            clauses.append("no radar return, so its speed is not measured")
        if gone:
            clauses.append("and it leaves the forward sector inside the horizon")
        rationale = ", ".join(clauses) + "."

        # The question points at the object in the same terms the answer is
        # asked for. It used to give a range and a bearing whatever the form,
        # so an item whose answer is an image box opened by naming the object
        # in metres -- the model had to convert polar coordinates into the
        # image before it could even tell which object was meant, which is
        # arithmetic the task is not about. Answering in (x, y) from a polar
        # question had the same shape. Pointing at a box with a box is also
        # what a perception stack actually hands a predictor.
        now = {"azdeg": (f"at {agent.range_m:.0f} m, azimuth "
                         f"{agent.azimuth_deg:+.0f} deg"),
               "xy": "at ({:+.1f}, {:+.1f}) m".format(
                   *agent_offset(derived, t_s, agent)),
               "bbox": None if now_box is None else f"at {now_box}"}

        for form, how in asked.items():
            answer = render(form)
            if not answer or now[form] is None:
                continue
            question = (f"Track #{int(agent.track_id)} is a {agent.label_class} "
                        f"{now[form]}. Where will it be over "
                        f"the next 3 seconds? Answer {how}, or say "
                        f"'{LEAVES}' for a horizon it does not reach.")
            emit(f"agent_traj_{form}", seconds + 1, question, answer)
            emit(f"agent_traj_{form}_cot", seconds + 1, cot_prompt(question),
                 json.dumps({"rationale": rationale, "answer": answer}))

    # 02 -- tracking, posed as tracking: a history goes in, one instant comes
    # out. The ids the answer must use are fixed by the history block in the
    # prompt, not by the label, so the same check works when the history is the
    # model's own output at evaluation time as when it is the truth in training.
    ids = local_ids(boxes)
    camera = camera_model(clip_id, row, nvidia_root)
    ASKED = {"azdeg": "with its class, range and azimuth",
             "bbox": "with its class and image bounding box [x1, y1, x2, y2] "
                     "normalised to 0-1000"}
    for seconds in STEP_ANCHOR_S:
        if float(seconds) > derived["t"].max():
            break
        past = [(back, step_frame(boxes, float(seconds - back), ids, camera))
                for back in range(STEP_HISTORY, 0, -1) if seconds - back >= 0]
        step_scan = (radar_scan(radar, extrinsics, SENSOR_NAME[short],
                                float(seconds), ego, derived)
                     if radar is not None and extrinsics is not None else None)
        now = step_frame(boxes, float(seconds), ids, camera, scan=step_scan)
        if not past:
            continue
        for form, asked in ASKED.items():
            if form == "bbox" and not any(o["bbox"] for o in now):
                continue
            history = "\n".join(f"t-{back}s: {step_text(objects, form)}"
                                for back, objects in past)
            prompt = ("Previous detections:\n" + history + "\n"
                      f"Continue tracking: list the road users at this instant "
                      f"{asked}, reusing the id each object already has and "
                      f"giving a new id to anything not seen before.")
            emit(f"track_step_{form}", seconds + 1, prompt, step_text(now, form))
            if camera is not None:
                emit(f"track_step_{form}_cot", seconds + 1, cot_prompt(prompt),
                     json.dumps({"rationale": step_reason(now, past[-1][1], form),
                                 "answer": step_text(now, form)}))

    return items


def camera_model(clip_id, row, nvidia_root):
    """(rotation, translation, intrinsics) for the front wide camera, or None.

    The lens is f-theta, not pinhole, so the intrinsics are a polynomial rather
    than a matrix; `geometry.ftheta_project` consumes this dict directly.
    """
    if not isinstance(row.get("camera_intrinsics_parquet"), str):
        return None
    if not isinstance(row.get("radar_extrinsics_parquet"), str):
        return None
    try:
        intr = pd.read_parquet(os.path.join(nvidia_root,
                                            row["camera_intrinsics_parquet"]))
        ext = pd.read_parquet(os.path.join(nvidia_root,
                                           row["radar_extrinsics_parquet"]))
        if (clip_id, CAMERA) not in intr.index or (clip_id, CAMERA) not in ext.index:
            return None
        model = json.loads(intr.loc[(clip_id, CAMERA), "model_parameters"])
        rot, translation = extrinsics_to_matrix(ext.loc[(clip_id, CAMERA)])
    except Exception:
        return None
    return rot, translation, model


def track_records(boxes, radar, extrinsics, short, ego, derived, camera):
    """One record per track: where it is, whether the radar sees it, how long.

    Tracks are kept whatever the radar does, with `radar_confirmed` saying which
    the sensor supports. Dropping the unconfirmed ones would have removed 58% of
    forward tracks and 80% of pedestrians -- measured -- turning "track what is
    ahead" into "track vehicles", and teaching the model that people are not
    there to be tracked. The flag keeps both: it cannot be answered from the
    camera, because which object carries a return is a fact about the radar.
    """
    seen, confirmed = {}, set()
    for frame in range(1, N_CLIP_FRAMES + 1):
        t_s = float(frame - 1)
        here = visible_at(boxes, t_s)
        if here is None or here.empty:
            continue
        scan = (radar_scan(radar, extrinsics, SENSOR_NAME[short], t_s, ego, derived)
                if radar is not None and extrinsics is not None else None)
        hits = radar_hits(scan, here.head(24)) if scan is not None else {}
        # `radar_hits` keys off the raw track_id, which the obstacle archive
        # stores as a string; the records key off int(track_id). Comparing the
        # two silently never matched, so every track read radar_confirmed=false.
        confirmed |= {int(k) for k in hits}
        for r in here.itertuples():
            record = seen.setdefault(int(r.track_id),
                                     {"first": frame, "count": 0})
            record.update(last=frame, row=r)
            record["count"] += 1

    out = []
    for tid, rec in seen.items():
        if rec["count"] < TRACK_MIN_FRAMES:
            continue
        r = rec["row"]
        bbox = None
        if camera is not None:
            rot, translation, model = camera
            box = image_bbox([r.center_x, r.center_y, r.center_z],
                             [r.size_x, r.size_y, r.size_z],
                             [r.orientation_x, r.orientation_y,
                              r.orientation_z, r.orientation_w],
                             rot, translation, model)
            if box:
                width, height = model["resolution"]
                bbox = [round(box[0] / width * BBOX_SCALE),
                        round(box[1] / height * BBOX_SCALE),
                        round(box[2] / width * BBOX_SCALE),
                        round(box[3] / height * BBOX_SCALE)]
        out.append({"track_id": tid, "class": r.label_class, "bbox": bbox,
                    "range_m": round(float(r.range_m)),
                    "azimuth_deg": round(float(r.azimuth_deg)),
                    "motion": "moving" if r.moved else "stationary",
                    "radar_confirmed": tid in confirmed,
                    "first_observed_frame": rec["first"],
                    "last_observed_frame": rec["last"]})
    # Nearest first at the last sighting, matching every other task's ordering.
    return sorted(out, key=lambda e: e["range_m"])[:MAX_TRACKS]


def world_speeds(boxes):
    """{track_id: (times, speed)} in the world frame, by central difference.

    The motion label used to be a track-level fact -- did this object move at
    any point in the clip -- which a two second window cannot establish. This is
    the instantaneous quantity the question actually asks about, and it agrees
    with the radar's Doppler verdict 88.6% of the time against 85.8% for the
    track-level version.
    """
    out = {}
    for tid, g in boxes.groupby("track_id"):
        g = g.sort_values("t_s")
        times = g["t_s"].to_numpy()
        if len(times) < 3:
            continue
        out[tid] = (times, np.hypot(np.gradient(g["wx"].to_numpy(), times),
                                    np.gradient(g["wy"].to_numpy(), times)))
    return out


def motion_evidence(row, previous, derived, t_s, v_rig, hit):
    """What each sensor says about this object's motion, and what it expects.

    Both sensors are blind along one axis and they are perpendicular axes. The
    Doppler measures range rate, so an object crossing the ego's path shows
    almost none however fast it moves; the bearing measures the crossing, so an
    object approaching head-on barely shifts. Measured over 3,964 objects: the
    ego-compensated bearing drift is a median 0.24 deg for stationary objects
    and 2.31 deg for moving ones, and it catches 61.4% of the movers the Doppler
    misses.

    Both expectations come from the ego's own state, which is why the rationale
    starts there: speed projects onto the ray to give the radial a stationary
    object would show, and speed with yaw rate gives where it would have drifted.
    """
    azimuth = np.radians(row.azimuth_deg)
    ray = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    expected_radial = -float(ray @ v_rig)

    expected_az = None
    if previous is not None:
        i = at_time(derived, t_s)
        origin, yaw = derived["xy"][i], derived["yaw"][i]
        cos, sin = np.cos(yaw), np.sin(yaw)
        dx, dy = previous.wx - origin[0], previous.wy - origin[1]
        expected_az = float(np.degrees(np.arctan2(-sin * dx + cos * dy,
                                                  cos * dx + sin * dy)))
    return {
        "expected_radial": expected_radial,
        "measured_radial": None if hit is None else float(hit["hit_radial"]),
        "expected_az": expected_az,
        "measured_az": float(row.azimuth_deg),
    }


def local_ids(boxes):
    """Track ids renumbered 1, 2, 3... in order of first appearance in the clip.

    The archive's ids are arbitrary integers in the hundreds. What the task is
    about is whether an object keeps the *same* number over time, not what that
    number is, so small ordinals make the answer shorter and the question
    honest: a model that calls the first car #1 and the label calls it #68 is
    not wrong, and renumbering removes the temptation to score it as if it were.
    """
    order = boxes.sort_values("t_s").drop_duplicates("track_id")["track_id"]
    return {tid: i + 1 for i, tid in enumerate(order)}


def step_frame(boxes, t_s, ids, camera=None, limit=MAX_LISTED, scan=None):
    """The objects at one instant, with both representations attached.

    Two output formats exist for the same instant -- range/azimuth and an image
    box -- so the geometry is computed once and each format reads what it needs.
    """
    here = visible_at(boxes, t_s)
    if here is None or here.empty:
        return []
    listed = here.head(limit)
    evidence = object_evidence(scan, listed) if scan is not None else {}
    out, pixel_boxes = [], []
    for r in listed.itertuples():
        if r.track_id not in ids:
            continue
        bbox = camera_az = None
        raw = None
        if camera is not None:
            rot, translation, model = camera
            raw = image_bbox([r.center_x, r.center_y, r.center_z],
                             [r.size_x, r.size_y, r.size_z],
                             [r.orientation_x, r.orientation_y,
                              r.orientation_z, r.orientation_w],
                             rot, translation, model)
            if raw:
                width, height = model["resolution"]
                bbox = [round(raw[0] / width * BBOX_SCALE),
                        round(raw[1] / height * BBOX_SCALE),
                        round(raw[2] / width * BBOX_SCALE),
                        round(raw[3] / height * BBOX_SCALE)]
                camera_az = camera_azimuth(((raw[0] + raw[2]) / 2,
                                            (raw[1] + raw[3]) / 2), model, rot)
        hit = evidence.get(r.track_id)
        # Objects are nearest-first, so everything already in the list is in
        # front of this one and is what can be hiding it.
        out.append({"id": ids[r.track_id], "cls": r.label_class,
                    "rng": float(r.range_m), "az": float(r.azimuth_deg),
                    "moved": bool(r.moved), "bbox": bbox, "camera_az": camera_az,
                    "occluded": occluded_fraction(raw, pixel_boxes),
                    "radar": ((hit["hit_range_m"], hit["hit_azimuth_deg"],
                               int(hit["hit_points"])) if hit else None)})
        pixel_boxes.append(raw)
    return out


def occluded_fraction(box, nearer):
    """How much of an image box the boxes in front of it cover.

    Measured on 2,257 listed objects, a quarter are more than half covered, and
    the radar still has a return on 48% of those -- so what each sensor can
    contribute changes object by object, and a tracking rationale that does not
    say which is missing the case fusion exists for. The sum over nearer boxes
    double-counts where they overlap each other, which can only overstate the
    occlusion; it is capped at 1 and used as an ordering, not a measurement.
    """
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    area = max(1e-6, (x2 - x1) * (y2 - y1))
    total = 0.0
    for other in nearer:
        if other is None:
            continue
        ix = max(0.0, min(x2, other[2]) - max(x1, other[0]))
        iy = max(0.0, min(y2, other[3]) - max(y1, other[1]))
        total += ix * iy
    return min(1.0, total / area)


def step_reason(objects, previous, form="azdeg"):
    """Why each object is where the answer says, sensor by sensor.

    Three checkable facts per object: where it was a second ago, what the camera
    can still see of it, and what the radar measures. The last two are the point
    -- the camera loses an object to occlusion far more often than the radar
    loses it to anything, and the radar has no range for the 40% it never
    illuminates, so neither alone accounts for the answer.
    """
    was = {o["id"]: o for o in previous}
    clauses = []
    for i, o in enumerate(objects):
        parts = []
        before = was.get(o["id"])
        if before is not None:
            parts.append(f"was at {before['rng']:.0f} m az {before['az']:+.0f} deg")
        else:
            parts.append("newly seen")
        if o.get("camera_az") is not None:
            hidden = o.get("occluded", 0.0)
            parts.append(f"camera az {o['camera_az']:+.0f} deg"
                         + (f", {hidden * 100:.0f}% occluded" if hidden > 0.5 else ""))
        if o.get("radar"):
            rng, az, n = o["radar"]
            parts.append(f"radar {n} return{'s' if n > 1 else ''} at {rng:.0f} m "
                         f"az {az:+.0f} deg")
        else:
            parts.append("no radar return")
        clauses.append(f"#{o['id']} {o['cls']}: " + ", ".join(parts))
    return ("; ".join(clauses) + ". Range comes from the radar where it has a "
            "return and from the camera alone where it does not.")


def step_text(objects, form="azdeg"):
    """One instant as text, in whichever representation the instruction asked
    for. The history block uses the same form as the answer, so the ids the
    model has to carry forward are written the way it must write them back.

    No motion here either, for the reason task 01 has none: `moved` is a
    track-level fact measured over the whole clip, and a five second window
    cannot establish it. Carrying it in the history would be worse than useless
    -- the model could score by copying its own first guess forward, and in a
    rollout that first guess is made from an empty history.
    """
    if form == "bbox":
        objects = [o for o in objects if o["bbox"]]
    if not objects:
        return "none"
    if form == "bbox":
        return "; ".join(
            f"#{o['id']} {o['cls']} [{o['bbox'][0]}, {o['bbox'][1]}, "
            f"{o['bbox'][2]}, {o['bbox'][3]}]" for o in objects)
    return "; ".join(f"#{o['id']} {o['cls']} {o['rng']:.0f} m "
                     f"az {o['az']:+.0f} deg" for o in objects)


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
