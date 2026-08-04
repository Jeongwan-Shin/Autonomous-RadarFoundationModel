#!/usr/bin/env python3
"""Scan-level summaries or per-object evidence -- which one do the tasks need?

`probe_task_coupling.py` regressed each task's target on the five scan-level
structure quantities and found almost nothing: R^2 0.002 for `agent_traj`, 0.007
for `plan_ego`, 0.032 for `det_objects`. Fixing those probes would buy the
downstream tasks nothing. But that was measured against scalar stand-ins -- an
object count for a task whose answer is a list -- so a low number there is
suggestive rather than conclusive.

This settles it at the granularity the tasks actually use. For every labelled
object ahead, two competing feature sets predict its attributes:

  scan   the five scan-level quantities, identical for every object in a frame
  object the returns inside that object's own box -- count, moving count,
         median radial velocity -- which is what `radar_hits` already computes

If the object features win by a wide margin, the probes are asking at the wrong
granularity and the fix is not a better scan summary but a per-object question.
That also decides what a GRPO reward should score, since a reward can only shape
what it can measure per item.

    python -m datatools.object_coupling --workers 12
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import paths
from .frame_objects import (ANCHOR_FRAMES, RADAR_PREFERENCE, SENSOR_NAME,
                            boxes_world, ego_frame, log, object_evidence,
                            radar_scan, read_member, visible_at)
from .geometry import points_in_box
from .radar_structure import structure

SCAN_FEATURES = ("bearing_deg", "lateral_pct", "spread_deg", "closing_ms",
                 "near_far_pct")
# `radar_hits` returns counts and a velocity but throws the geometry away, so
# on its own it cannot say where an object is. The measured range and bearing of
# the returns inside the box are what the sensor actually reports, and leaving
# them out would rig the comparison against the object-level features.
OBJECT_FEATURES = ("hit_points", "hit_moving", "hit_radial",
                   "hit_range_m", "hit_azimuth_deg")

# What a task has to say about each object. These are the actual quantities the
# answer text is built from, not stand-ins.
OBJECT_TARGETS = {
    "range_m": ("det_objects / depth_range: 물체까지 거리", "range_m"),
    "azimuth_deg": ("det_objects: 물체의 방위각", "azimuth_deg"),
    "moved": ("motion_seg: 이동/정지 판정", "moved"),
}


def clip_rows(clip_id, row, nvidia_root):
    """One row per (frame, object): scan summary, its own radar evidence, truth."""
    short = next((r for r in RADAR_PREFERENCE if row.get(f"has_{r}", False)), None)
    if short is None or not row.get("has_radar_extrinsics", False):
        return []
    if not (row.get("has_egomotion", False) and row.get("has_obstacle", False)):
        return []
    ego = read_member(nvidia_root, row["egomotion_zip"], row["egomotion_member"])
    obstacle = read_member(nvidia_root, row["obstacle_zip"], row["obstacle_member"])
    if obstacle.empty:
        return []
    derived = ego_frame(ego)
    boxes = boxes_world(obstacle, ego)
    if boxes is None or boxes.empty:
        return []
    radar = read_member(nvidia_root, row[f"radar_{short}_zip"],
                        row[f"radar_{short}_member"])
    full = pd.read_parquet(os.path.join(nvidia_root,
                                        row["radar_extrinsics_parquet"]))
    if clip_id not in full.index.get_level_values(0):
        return []
    extrinsics = full.loc[clip_id]

    out = []
    for frame in ANCHOR_FRAMES:
        t_s = float(frame - 1)
        scan = radar_scan(radar, extrinsics, SENSOR_NAME[short], t_s, ego, derived)
        if scan is None:
            continue
        st = structure(scan)
        if st is None:
            continue
        rows = visible_at(boxes, t_s)
        if rows is None or rows.empty:
            continue
        hits = object_evidence(scan, rows)
        summary = {
            "bearing_deg": st["bearing_deg"],
            "lateral_pct": 100.0 * st["n_left"] / (st["n_left"] + st["n_right"]),
            "spread_deg": st["spread_deg"],
            "closing_ms": st["closing_ms"],
            "near_far_pct": 100.0 * st["n_near"] / st["n_total"],
        }
        for r in rows.itertuples():
            ev = hits.get(r.track_id)
            if ev is None:
                # No return inside the box. Real and common -- radar illuminates
                # 20% of pedestrians -- so these rows stay in, with the count at
                # zero and the geometry at NaN, which the fit then drops.
                ev = {"hit_points": 0.0, "hit_moving": 0.0, "hit_radial": 0.0,
                      "hit_range_m": np.nan, "hit_azimuth_deg": np.nan}
            out.append({
                "clip_id": clip_id, "frame": int(frame),
                **summary, **ev,
                "range_m": float(r.range_m),
                "azimuth_deg": float(r.azimuth_deg),
                "moved": float(getattr(r, "moved", np.nan)),
            })
    return out


def process(args_tuple):
    clip_id, row, nvidia_root = args_tuple
    try:
        return clip_rows(clip_id, row, nvidia_root)
    except Exception:
        return []


def ridge_r2(x_train, y_train, x_test, y_test, alpha):
    mu, sd = x_train.mean(0), x_train.std(0)
    sd[sd == 0] = 1.0
    a, b = (x_train - mu) / sd, (x_test - mu) / sd
    a = np.hstack([a, np.ones((len(a), 1))])
    b = np.hstack([b, np.ones((len(b), 1))])
    pen = alpha * np.eye(a.shape[1])
    pen[-1, -1] = 0.0
    w = np.linalg.solve(a.T @ a + pen, a.T @ y_train)
    resid = ((y_test - b @ w) ** 2).sum()
    total = ((y_test - y_test.mean()) ** 2).sum()
    return 1.0 - resid / total if total > 0 else 0.0


def compare(frame, target):
    """Both feature sets on the SAME rows, and the same train/test split.

    The scan summary is defined for every object; the object evidence only for
    the ones the radar actually illuminates, which is 39% of them. Letting each
    fit choose its own rows compared a hard subset against an easy one -- the
    scan features looked worse partly because they were being asked about
    objects the radar never saw.
    """
    columns = list(SCAN_FEATURES) + list(OBJECT_FEATURES) + [target]
    sub = frame[columns].dropna()
    if len(sub) < 1000:
        return None
    y = sub[target].to_numpy(float)
    rng = np.random.default_rng(0)
    cut = rng.permutation(len(sub))
    tr, te = cut[: int(0.7 * len(cut))], cut[int(0.7 * len(cut)):]
    out = {}
    for name, features in (("scan", SCAN_FEATURES), ("object", OBJECT_FEATURES)):
        x = sub[list(features)].to_numpy(float)
        out[name] = max(ridge_r2(x[tr], y[tr], x[te], y[te], a)
                        for a in (0.1, 1.0, 10.0, 100.0, 1000.0))
    out["n"] = len(sub)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nvidia-root", default=paths.NVIDIA_ROOT)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--clips", type=int, default=20000,
                    help="objects come thick, so a subset is already millions "
                         "of rows and the estimate is not sample-limited")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    clips = pd.read_parquet(os.path.join(paths.COMMON_DIR, "nvidia_clips.parquet"))
    usable = clips[clips["has_egomotion"].fillna(False)
                   & clips["has_obstacle"].fillna(False)
                   & clips["has_radar_extrinsics"].fillna(False)]
    # Sample across the file rather than take a prefix: the clip index is
    # ordered by rig, so the first N are all one radar profile and carry no LRR.
    if args.clips and len(usable) > args.clips:
        step = len(usable) // args.clips
        usable = usable.iloc[::step].iloc[: args.clips]
    log(f"{len(usable):,} clips -> per-object rows")

    tasks = [(cid, usable.loc[cid].to_dict(), args.nvidia_root)
             for cid in usable.index]
    rows, started = [], time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, t) for t in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            rows.extend(f.result())
            if i % 2000 == 0 or i == len(tasks):
                log(f"  {i}/{len(tasks)}  {i/(time.monotonic()-started):.0f} clip/s "
                    f"  {len(rows):,} objects")
    frame = pd.DataFrame(rows)
    log(f"{len(frame):,} object rows")

    log("")
    log("물체 속성을 무엇이 설명하는가 (홀드아웃 R2)")
    log(f"{'물체 속성':34s} {'스캔 요약':>10s} {'물체별 증거':>12s} {'n':>12s}")
    results = []
    for target, (label, column) in OBJECT_TARGETS.items():
        r = compare(frame, column)
        if r is None:
            continue
        results.append({"target": target, "label": label, "scan_r2": r["scan"],
                        "object_r2": r["object"], "n": r["n"]})
        log(f"{label:34s} {r['scan']:>10.3f} {r['object']:>12.3f} {r['n']:>12,}")

    log("")
    for r in results:
        ratio = r["object_r2"] / r["scan_r2"] if r["scan_r2"] > 0.001 else float("inf")
        verdict = ("물체별 증거가 압도" if ratio > 3
                   else "물체별 증거가 우세" if ratio > 1.3 else "차이 없음")
        log(f"  {r['target']:14s} 물체/스캔 = {ratio:6.1f}배   {verdict}")

    out = args.out or os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "runs", "09_structure", "object_coupling.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(results).to_json(out, orient="records", indent=1,
                                  force_ascii=False)
    log(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
