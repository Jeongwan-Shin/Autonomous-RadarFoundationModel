#!/usr/bin/env python3
"""Check the numeric claims in the radar-vision QA rationales against the labels.

The QA set states concrete quantities -- ego position and speed at a given
frame, other agents' positions, distances between them. Those are all derivable
from `egomotion.offline` and `obstacle.offline`, so every such claim can be
recomputed and compared instead of trusted.

Conventions established by fitting the data rather than assumed:

  frame index   1 Hz, frame N corresponds to t = (N - 1) seconds. Frame numbers
                span 1..22, so they index a ~1 fps summary of the 20 s clip, not
                the 605 camera frames.
  coordinates   clip-local world frame, the same frame `egomotion.offline` uses
                (origin at the ego pose at t=0). Matching stated ego positions
                against egomotion gave a median error of 0.279 m, which is what
                confirmed the frame choice.

`obstacle.offline` boxes are stored in the rig frame at their own timestamp, so
they are lifted into the world frame before being compared with a stated agent
position.

Claims come from `qa_claims.extract`, which works sentence by sentence. The
earlier whole-text approach paired every number with the first frame in the
rationale and let subject-verb gaps run across clauses; both mistakes turned
correct statements into apparent errors. See `qa_claims` for the evidence.

Agents are named in prose ("the black sedan turning right"), so a claim about one
cannot be tied to a specific track_id. Agent positions are therefore checked for
plausibility -- is any labelled object near the stated point at that time --
rather than for identity.

    python -m datatools.verify_qa
    python -m datatools.verify_qa --limit 200      # quick pass
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
from .geometry import interpolate_pose, rig_to_world
from .qa_claims import extract

FRAME_HZ = 1.0          # frame N -> t = (N - 1) / FRAME_HZ seconds

TOLERANCE = {
    "ego_pos": 3.0,       # m; rounding plus up to a frame of slack
    "ego_speed": 1.0,     # m/s
    "agent_pos": 5.0,     # m; "is any object here", so ~one vehicle length
    "agent_speed": 1.5,   # m/s; a presence test again, not an identity one
    "distance": 5.0,      # m
    "future_pos": 5.0,    # m
}
UNITS = {"ego_pos": "m", "ego_speed": "m/s", "agent_pos": "m",
         "agent_speed": "m/s",
         "distance": "m", "future_pos": "m"}


def frame_to_seconds(frame):
    return (frame - 1) / FRAME_HZ


def load_clip(nvidia_root, row):
    """egomotion and obstacle for one clip, obstacle lifted into world frame."""
    with zipfile.ZipFile(os.path.join(nvidia_root, row["egomotion_zip"])) as zf:
        ego = pd.read_parquet(io.BytesIO(zf.read(row["egomotion_member"])))
    with zipfile.ZipFile(os.path.join(nvidia_root, row["obstacle_zip"])) as zf:
        obs = pd.read_parquet(io.BytesIO(zf.read(row["obstacle_member"])))

    ego_t = ego["timestamp"].to_numpy()
    ego_xyz = ego[["x", "y", "z"]].to_numpy()
    ego_quat = ego[["qx", "qy", "qz", "qw"]].to_numpy()
    ego_speed = np.linalg.norm(
        np.gradient(ego_xyz, ego_t / 1e6, axis=0)[:, :2], axis=1)

    if obs.empty:
        world, obs_t = np.zeros((0, 3)), np.zeros(0)
    else:
        obs_t = obs["timestamp_us"].to_numpy()
        rig = obs[["center_x", "center_y", "center_z"]].to_numpy()
        pose_xyz, pose_rot = interpolate_pose(obs_t, ego_t, ego_xyz, ego_quat)
        world = rig_to_world(rig, pose_xyz, pose_rot)

    # World-frame speed per track, so a stated agent speed can be checked the
    # same way a stated agent position is: does some labelled object move at
    # about that rate at that instant.
    obs_speed = np.zeros(len(obs_t))
    if len(obs_t):
        ids = obs["track_id"].to_numpy()
        for tid in np.unique(ids):
            m = ids == tid
            if m.sum() < 3:
                continue
            seconds = obs_t[m] / 1e6
            order = np.argsort(seconds)
            xy = world[m][:, :2][order]
            v = np.gradient(xy, seconds[order], axis=0)
            speeds = np.linalg.norm(v, axis=1)
            slot = np.where(m)[0][order]
            obs_speed[slot] = speeds

    return {"ego_t": ego_t, "ego_xy": ego_xyz[:, :2], "ego_speed": ego_speed,
            "obs_t": obs_t, "obs_xy": world[:, :2], "obs_speed": obs_speed}


def ego_at(clip, t_s):
    t_us = np.clip(t_s * 1e6, clip["ego_t"][0], clip["ego_t"][-1])
    i = int(np.argmin(np.abs(clip["ego_t"] - t_us)))
    return clip["ego_xy"][i], clip["ego_speed"][i]


def objects_near(clip, t_s, window_s=0.6):
    if len(clip["obs_t"]) == 0:
        return np.zeros((0, 2))
    lo, hi = (t_s - window_s) * 1e6, (t_s + window_s) * 1e6
    mask = (clip["obs_t"] >= lo) & (clip["obs_t"] <= hi)
    return clip["obs_xy"][mask]


def evaluate(clip, claim):
    """Recompute one claim. Returns (error, computed) or (None, None) if the
    claim cannot be judged -- no frame stated, or nothing labelled to compare."""
    if claim["frame"] is None:
        return None, None
    t_s = frame_to_seconds(claim["frame"])
    ego_xy, ego_speed = ego_at(clip, t_s)
    kind = claim["kind"]

    if kind == "ego_pos":
        stated = np.array([claim["x"], claim["y"]])
        return float(np.linalg.norm(stated - ego_xy)), [round(v, 3) for v in ego_xy]

    if kind == "ego_speed":
        return abs(claim["value"] - float(ego_speed)), round(float(ego_speed), 3)

    if kind == "future_pos":
        target, _ = ego_at(clip, t_s + claim["horizon_s"])
        stated = np.array([claim["x"], claim["y"]])
        return float(np.linalg.norm(stated - target)), [round(v, 3) for v in target]

    if kind == "agent_pos":
        near = objects_near(clip, t_s)
        if len(near) == 0:
            return None, None
        stated = np.array([claim["x"], claim["y"]])
        distances = np.linalg.norm(near - stated, axis=1)
        i = int(np.argmin(distances))
        return float(distances[i]), [round(v, 3) for v in near[i]]

    if kind == "agent_speed":
        if len(clip["obs_t"]) == 0:
            return None, None
        lo, hi = (t_s - 0.6) * 1e6, (t_s + 0.6) * 1e6
        mask = (clip["obs_t"] >= lo) & (clip["obs_t"] <= hi)
        speeds = clip["obs_speed"][mask]
        if len(speeds) == 0:
            return None, None
        i = int(np.argmin(np.abs(speeds - claim["value"])))
        return float(abs(speeds[i] - claim["value"])), round(float(speeds[i]), 3)

    if kind == "distance":
        # The text gives the range to one named agent, not to the closest one, and
        # prose cannot be tied to a track. So ask whether *some* labelled object
        # sits at the stated range -- the same presence test used for agent_pos.
        # Comparing against the nearest object instead dropped agreement to 35%
        # purely because the referenced agent usually was not the nearest.
        near = objects_near(clip, t_s)
        if len(near) == 0:
            return None, None
        ranges = np.linalg.norm(near - ego_xy, axis=1)
        i = int(np.argmin(np.abs(ranges - claim["value"])))
        return float(abs(ranges[i] - claim["value"])), round(float(ranges[i]), 3)

    return None, None


def check_item(clip, rationale):
    results = []
    for index, claim in enumerate(extract(rationale)):
        error, computed = evaluate(clip, claim)
        if error is None:
            continue
        kind = claim["kind"]
        results.append({
            "claim_index": index,
            "kind": kind,
            "stated": claim.get("value", [claim.get("x"), claim.get("y")]),
            "computed": computed,
            "error": round(error, 4),
            "tolerance": TOLERANCE[kind],
            "unit": UNITS[kind],
            "ok": bool(error <= TOLERANCE[kind]),
            "frame": claim["frame"],
        })
    return results


def process_clip(args_tuple):
    nvidia_root, row, items = args_tuple
    try:
        clip = load_clip(nvidia_root, row)
    except Exception as exc:
        return [{"clip_id": row["clip_id"], "qa_index": i, "n_checks": 0,
                 "n_pass": 0, "load_error": type(exc).__name__}
                for i, _ in enumerate(items)]

    out = []
    for index, (taxonomy, rationale) in enumerate(items):
        results = check_item(clip, rationale)
        record = {"clip_id": row["clip_id"], "qa_index": index,
                  "taxonomy": taxonomy,
                  "n_checks": len(results),
                  "n_pass": sum(r["ok"] for r in results),
                  "claims": json.dumps(results) if results else None}
        for kind in TOLERANCE:
            errs = [r["error"] for r in results if r["kind"] == kind]
            if errs:
                record[f"{kind}_err"] = max(errs)
                record[f"{kind}_ok"] = all(r["ok"] for r in results
                                           if r["kind"] == kind)
        out.append(record)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--qa-dir",
                    default=os.path.join(paths.SPLIT_ROOT, "10_radar_vision_qa/qa"))
    ap.add_argument("--out",
                    default=os.path.join(paths.SPLIT_ROOT,
                                         "10_radar_vision_qa/verification.parquet"))
    ap.add_argument("--nvidia-root", default=paths.NVIDIA_ROOT)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    clips = pd.read_parquet(os.path.join(paths.COMMON_DIR, "nvidia_clips.parquet"))
    files = sorted(glob.glob(os.path.join(args.qa_dir, "*.json")))
    if args.limit:
        files = files[: args.limit]
    print(f"[{time.strftime('%H:%M:%S')}] {len(files)} QA files", flush=True)

    tasks, skipped = [], 0
    for path in files:
        doc = json.load(open(path))
        clip_id = doc["clip_id"]
        if clip_id not in clips.index:
            skipped += 1
            continue
        row = clips.loc[clip_id]
        if not (row["has_egomotion"] and row["has_obstacle"]):
            skipped += 1
            continue
        tasks.append((args.nvidia_root,
                      {"clip_id": clip_id,
                       "egomotion_zip": row["egomotion_zip"],
                       "egomotion_member": row["egomotion_member"],
                       "obstacle_zip": row["obstacle_zip"],
                       "obstacle_member": row["obstacle_member"]},
                      [(q.get("taxonomy"), q.get("rationale", ""))
                       for q in doc.get("qa", [])]))
    if skipped:
        print(f"  {skipped} clips skipped (not downloaded or missing labels)",
              flush=True)

    rows = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_clip, t) for t in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            if i % 400 == 0 or i == len(tasks):
                rate = i / (time.monotonic() - started)
                print(f"  {i}/{len(tasks)} clips  {rate:.1f}/s", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}  rows={len(frame):,}")
    report(frame)
    return 0


def report(frame):
    total = len(frame)
    print("\n" + "=" * 72)
    print(f"QA items: {total:,}")
    checkable = frame["n_checks"] > 0
    print(f"  contain a checkable claim: {checkable.sum():,} "
          f"({checkable.mean()*100:.1f}%)")
    print(f"  no numeric claim to check: {(~checkable).sum():,} "
          f"({(~checkable).mean()*100:.1f}%)")

    print("\nper-quantity agreement")
    for kind, tol in TOLERANCE.items():
        flag, err = f"{kind}_ok", f"{kind}_err"
        if flag not in frame.columns:
            continue
        sub = frame[frame[flag].notna()]
        if sub.empty:
            continue
        # object dtype because booleans sit beside NaN; mean() is unreliable there
        rate = sub[flag].astype(bool).mean()
        errs = sub[err].replace([np.inf], np.nan).dropna()
        print(f"  {kind:11s} n={len(sub):6,}  within {tol} {UNITS[kind]}: "
              f"{rate*100:5.1f}%  median {errs.median():6.2f}  "
              f"p90 {errs.quantile(0.9):7.2f}")

    if checkable.any():
        sub = frame[checkable]
        agree = (sub["n_pass"] == sub["n_checks"])
        print(f"\n  all claims in the item agree: {agree.mean()*100:.1f}%")
        print(f"  claim-level agreement       : "
              f"{sub['n_pass'].sum()/sub['n_checks'].sum()*100:.1f}% "
              f"({int(sub['n_pass'].sum()):,}/{int(sub['n_checks'].sum()):,})")

    if "taxonomy" in frame.columns:
        print("\nby taxonomy (claim-level agreement)")
        for tax, g in frame[checkable].groupby("taxonomy"):
            print(f"  {tax:14s} n={len(g):6,}  "
                  f"{g['n_pass'].sum()/g['n_checks'].sum()*100:5.1f}%")


if __name__ == "__main__":
    sys.exit(main())
