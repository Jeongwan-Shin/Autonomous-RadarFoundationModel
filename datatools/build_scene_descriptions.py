#!/usr/bin/env python3
"""Generate scene descriptions for task 11 from the measured scene features.

Six kinds, because each one forces a different pathway to carry signal. A model
trained only on camera-derived text can ignore its radar input and still score
well, which is exactly the failure the radar and complementarity kinds exist to
prevent.

  objects           what is in the forward sector, from obstacle.offline
  ego_maneuver      what the vehicle is doing, from egomotion.offline
  radar             what the radar sees, including how much of it is moving
  complementarity   which labelled objects the radar illuminates and which it
                    misses -- the only expression of sensor complementarity in
                    the release
  context           where and when, from the collection metadata
  qa_summary        aggregated from the verified QA rationales

Every number is read from `common/scene_features_qa_clips.parquet`, which was
computed from the labels; nothing here is invented. Phrasing is varied across a
handful of templates per kind, chosen by hashing (clip_id, frame, kind) so the
output is reproducible.

Conventions that the text depends on, both checked against the data rather than
assumed:

  frame       1 Hz, frame N is t = (N - 1) s -- the same index the QA uses, so a
              description and a QA item about the same frame line up.
  yaw sign    positive yaw rate is a left turn. Verified on five high-yaw clips:
              the unwrapped yaw and the heading atan2(vy, vx) move together to
              within 1 degree, so the frame is x-forward / y-left.

    python -m datatools.build_scene_descriptions
    python -m datatools.build_scene_descriptions --limit 50 --print 8
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from . import paths

CLASS_WORD = {"automobile": "car", "person": "pedestrian",
              "heavy_truck": "heavy truck", "bus": "bus", "trailer": "trailer",
              "rider": "rider", "protruding_object": "protruding object",
              "animal": "animal", "stroller": "stroller",
              "other_vehicle": "vehicle"}
COUNT_CLASSES = list(CLASS_WORD)

STOPPED_MS = 0.5
BRAKE_MS2 = -1.0
ACCEL_MS2 = 1.0
TURN_DEGS = 6.0
SHARP_TURN_DEGS = 15.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick(options, *key_parts):
    """Deterministic choice, so a rerun produces identical text."""
    digest = hashlib.md5("|".join(str(p) for p in key_parts).encode()).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def plural(count, word):
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def object_phrase(row):
    parts = []
    for name in COUNT_CLASSES:
        count = int(row.get(f"n_{name}", 0) or 0)
        if count:
            parts.append(plural(count, CLASS_WORD[name]))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def describe_objects(row, key):
    listing = object_phrase(row)
    if listing is None:
        return ("The forward sector is clear of labelled road users."
                if int(row["n_tracks_in_fov"] or 0) == 0 else None)
    nearest = row.get("nearest_range_m")
    nearest_word = CLASS_WORD.get(row.get("nearest_class"), "object")
    moving = int(row.get("n_moving_in_fov", 0) or 0)
    total = int(row["n_tracks_in_fov"] or 0)
    stationary = max(0, total - moving)

    templates = [
        f"Ahead: {listing}. The closest is a {nearest_word} at {nearest:.0f} m.",
        f"{listing.capitalize()} in the forward sector, nearest a {nearest_word} "
        f"{nearest:.0f} m away.",
        f"The scene holds {listing}; a {nearest_word} at {nearest:.0f} m is closest.",
    ]
    text = pick(templates, *key)
    if moving and stationary:
        text += f" {moving} of the {total} are moving, {stationary} are stationary."
    elif moving:
        text += f" All {total} are moving."
    elif total:
        text += f" All {total} are stationary."
    return text


def describe_ego_maneuver(row, key):
    speed = float(row["ego_speed_ms"])
    accel = float(row["ego_accel_ms2"])
    yaw = float(row["ego_yaw_rate_degs"])

    if speed < STOPPED_MS:
        motion = "The ego vehicle is stopped"
    else:
        motion = pick([f"The ego vehicle is travelling at {speed:.1f} m/s "
                       f"({speed*3.6:.0f} km/h)",
                       f"Ego speed is {speed:.1f} m/s ({speed*3.6:.0f} km/h)"],
                      *key)
    clauses = []
    if accel <= BRAKE_MS2:
        clauses.append(f"decelerating at {abs(accel):.1f} m/s²")
    elif accel >= ACCEL_MS2:
        clauses.append(f"accelerating at {accel:.1f} m/s²")
    # Positive yaw rate is a left turn; verified against the heading angle.
    if abs(yaw) >= TURN_DEGS:
        side = "left" if yaw > 0 else "right"
        sharp = "sharply " if abs(yaw) >= SHARP_TURN_DEGS else ""
        clauses.append(f"turning {sharp}{side} at {abs(yaw):.0f} deg/s")
    if not clauses:
        clauses.append("holding a steady course")
    return motion + ", " + " while ".join(clauses) + "."


def describe_radar(row, key):
    available = [s for s in ("lrr1", "mrr2", "srr0")
                 if pd.notna(row.get(f"{s}_n_points"))]
    if not available:
        return None
    label = {"lrr1": "long-range imaging radar", "mrr2": "mid-range radar",
             "srr0": "short-range radar"}
    parts = []
    for short in available:
        total = int(row[f"{short}_n_points"])
        moving = int(row[f"{short}_n_moving"])
        parts.append(f"the {label[short]} returns {total} detections, "
                     f"{moving} of which are not explained by the ego's own motion")
    text = pick([f"At this instant {parts[0]}.",
                 f"Radar: {parts[0]}."], *key)
    if len(parts) > 1:
        text = text.rstrip(".") + "; " + "; ".join(parts[1:]) + "."
    best = row.get("lrr1_max_rcs", row.get("mrr2_max_rcs"))
    if pd.notna(best):
        strength = "strong" if best > 10 else "modest"
        text += (f" The brightest reflector measures {best:.0f} dBsm, "
                 f"a {strength} return.")
    return text


def describe_complementarity(row, key):
    tested = int(row.get("n_boxes_tested_for_radar", 0) or 0)
    if tested == 0:
        return None
    hit = int(row.get("n_boxes_with_radar", 0) or 0)
    only = int(row.get("n_boxes_camera_only", 0) or 0)
    if only == 0:
        return (f"All {tested} labelled objects in the forward sector carry a "
                f"radar return, so both sensors agree on what is present.")
    if hit == 0:
        return (f"None of the {tested} labelled objects in the forward sector "
                f"produces a radar return; at this instant they are visible to "
                f"the camera alone.")
    verb = "produces" if hit == 1 else "produce"
    tail = ("it is" if only == 1 else f"those {only} are")
    return pick([
        f"Of {tested} labelled objects ahead, {hit} {verb} a radar return and "
        f"{only} do not -- {tail} visible to the camera alone.",
        f"Radar illuminates {hit} of the {tested} objects ahead; the remaining "
        f"{only} appear only in the camera.",
    ], *key)


def describe_context(clip_row, key):
    hour = int(clip_row["hour_of_day"])
    month = int(clip_row["month"])
    country = clip_row["country"]
    if hour >= 20 or hour <= 5:
        light = "at night"
    elif hour <= 8:
        light = "in the early morning"
    elif hour >= 17:
        light = "in the late afternoon"
    else:
        light = "during the day"
    season = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring",
              5: "spring", 6: "summer", 7: "summer", 8: "summer",
              9: "autumn", 10: "autumn", 11: "autumn"}[month]
    return pick([
        f"Recorded in {country} {light}, in {season} (month {month}, hour {hour}).",
        f"{country}, {light}, {season}. Local hour {hour}.",
    ], *key)


def describe_clip_summary(group, clip_row, key):
    """One paragraph per clip, from the frame rows plus the clip metadata."""
    speed = group["ego_speed_ms"]
    distance = float(group["ego_travelled_m"].max())
    yaw = group["ego_yaw_rate_degs"]
    turned = (yaw.abs() >= TURN_DEGS).sum()
    braked = (group["ego_accel_ms2"] <= BRAKE_MS2).sum()
    stopped = (speed < STOPPED_MS).sum()
    peak_tracks = int(group["n_tracks_in_fov"].max())

    bits = [f"Over 20 s the ego vehicle covers {distance:.0f} m, "
            f"with speed between {speed.min():.1f} and {speed.max():.1f} m/s"]
    if stopped:
        bits.append(f"is stopped in {stopped} of the {len(group)} sampled frames")
    if braked:
        bits.append(f"brakes hard in {braked} of them")
    if turned:
        direction = "left" if yaw[yaw.abs() >= TURN_DEGS].mean() > 0 else "right"
        bits.append(f"turns {direction} across {turned} frames")
    text = ", ".join(bits) + "."
    text += (f" Up to {peak_tracks} labelled road users are in the forward sector "
             f"at once.")
    if pd.notna(group["lrr1_n_points"]).any():
        text += (f" Long-range radar averages "
                 f"{group['lrr1_n_points'].mean():.0f} detections per scan, "
                 f"{group['lrr1_n_moving'].mean():.0f} of them moving.")
    return text


def qa_summary(doc):
    """Facts drawn only from QA items whose numeric claims were verified."""
    verified = [q for q in doc.get("qa", [])
                if q.get("verification", {}).get("status") == "agrees"]
    if not verified:
        return None
    taxa = {}
    for item in verified:
        # Three items in the delivered set carry no taxonomy field.
        name = item.get("taxonomy") or "unlabelled"
        taxa[name] = taxa.get(name, 0) + 1
    listing = ", ".join(f"{k.lower()} {v}" for k, v in sorted(taxa.items()))
    return (f"{len(verified)} question-answer pairs about this clip have had every "
            f"numeric claim recomputed against the labels and agree "
            f"({listing}). Their rationales are camera-grounded: they cite object "
            f"positions, speeds and distances taken from obstacle.offline and "
            f"egomotion.offline, not from radar returns.")


KINDS = ("objects", "ego_maneuver", "radar", "complementarity")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", default=os.path.join(
        paths.COMMON_DIR, "scene_features_qa_clips.parquet"))
    ap.add_argument("--out-dir", default=os.path.join(
        paths.SPLIT_ROOT, "11_radar_vision_scene_description"))
    ap.add_argument("--qa-dir", default=os.path.join(
        paths.SPLIT_ROOT, "10_radar_vision_qa/qa"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--print", type=int, default=0, dest="show")
    args = ap.parse_args(argv)

    features = pd.read_parquet(args.features)
    clips = pd.read_parquet(os.path.join(paths.COMMON_DIR, "nvidia_clips.parquet"))
    if args.limit:
        keep = features["clip_id"].drop_duplicates().head(args.limit)
        features = features[features["clip_id"].isin(keep)]
    log(f"{features['clip_id'].nunique():,} clips, {len(features):,} frames")

    frame_rows, clip_rows = [], []
    for clip_id, group in features.groupby("clip_id", sort=True):
        if clip_id not in clips.index:
            continue
        clip_row = clips.loc[clip_id]
        group = group.sort_values("frame")

        for row in group.to_dict("records"):
            key = (clip_id, row["frame"])
            for kind, fn in (("objects", describe_objects),
                             ("ego_maneuver", describe_ego_maneuver),
                             ("radar", describe_radar),
                             ("complementarity", describe_complementarity)):
                text = fn(row, key + (kind,))
                if text:
                    frame_rows.append({
                        "clip_id": clip_id, "frame": int(row["frame"]),
                        "t_s": float(row["t_s"]), "kind": kind, "text": text,
                        "split": clip_row["split"]})

        for kind, text in (
                ("context", describe_context(clip_row, (clip_id, "context"))),
                ("clip_summary", describe_clip_summary(
                    group, clip_row, (clip_id, "clip_summary")))):
            clip_rows.append({"clip_id": clip_id, "kind": kind, "text": text,
                              "split": clip_row["split"]})

        qa_path = os.path.join(args.qa_dir, f"{clip_id}.json")
        if os.path.exists(qa_path):
            text = qa_summary(json.load(open(qa_path)))
            if text:
                clip_rows.append({"clip_id": clip_id, "kind": "qa_summary",
                                  "text": text, "split": clip_row["split"]})

    os.makedirs(args.out_dir, exist_ok=True)
    frame_df = pd.DataFrame(frame_rows)
    clip_df = pd.DataFrame(clip_rows)
    frame_path = os.path.join(args.out_dir, "descriptions_frame.parquet")
    clip_path = os.path.join(args.out_dir, "descriptions_clip.parquet")
    frame_df.to_parquet(frame_path, index=False)
    clip_df.to_parquet(clip_path, index=False)
    log(f"wrote {frame_path}  rows={len(frame_df):,} "
        f"({os.path.getsize(frame_path)/1e6:.1f} MB)")
    log(f"wrote {clip_path}   rows={len(clip_df):,} "
        f"({os.path.getsize(clip_path)/1e6:.1f} MB)")

    log("")
    log("frame-level descriptions by kind")
    for kind, count in frame_df["kind"].value_counts().items():
        lengths = frame_df.loc[frame_df["kind"] == kind, "text"].str.len()
        log(f"  {kind:17s} {count:7,}  median {lengths.median():3.0f} chars")
    log("clip-level descriptions by kind")
    for kind, count in clip_df["kind"].value_counts().items():
        lengths = clip_df.loc[clip_df["kind"] == kind, "text"].str.len()
        log(f"  {kind:17s} {count:7,}  median {lengths.median():3.0f} chars")

    if args.show:
        log("")
        for kind in frame_df["kind"].unique():
            sub = frame_df[frame_df["kind"] == kind].head(args.show // 4 + 1)
            for _, r in sub.iterrows():
                log(f"[{kind}] {r.clip_id[:8]} f{r.frame}: {r.text}")
        for kind in clip_df["kind"].unique():
            r = clip_df[clip_df["kind"] == kind].iloc[0]
            log(f"[{kind}] {r.clip_id[:8]}: {r.text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
