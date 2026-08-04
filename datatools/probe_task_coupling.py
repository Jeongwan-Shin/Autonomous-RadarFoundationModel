#!/usr/bin/env python3
"""Which radar structure quantity does each task actually depend on?

`radar_structure.py` established that the language model recovers scalar
summaries of a scan and not its spatial layout. That is one number per probe and
it treats the five probes as interchangeable, which they are not: `det_objects`
needs to know where returns are, `motion_seg` needs to know how fast they close,
`depth_range` needs to know how they distribute in range. Knowing which probe
reads zero does not say which task that zero is costing.

This measures the coupling directly. For every task, regress its target on the
five structure quantities and report each one's unique contribution -- the drop
in held-out R^2 when that quantity alone is removed from an otherwise complete
model. A probe with a large unique contribution is one the task cannot be solved
without; a probe near zero is one the task never needed, so fixing it would buy
that task nothing however badly the model reads it.

The point is to choose. A reward can only shape what it scores, so the structure
quantity worth putting in a GRPO reward is the one the downstream tasks are
demonstrably a function of -- not whichever probe happens to have the cleanest
contamination number.

    python -m datatools.probe_task_coupling --workers 12
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import paths
from .frame_objects import (ANCHOR_FRAMES, RADAR_PREFERENCE, SENSOR_NAME, log,
                            ego_frame, radar_scan, read_member)
from .radar_structure import structure

QUANTITIES = ("bearing_deg", "lateral_pct", "spread_deg", "closing_ms",
              "near_far_pct")

# What `radar_objects_probe.py` asks about instead: the nearest object the radar
# illuminates, not the scan as a whole. Compared against QUANTITIES on the same
# rows and the same split, this is the evidence for choosing what a GRPO reward
# should score.
# Prefixed on join: the probe table's `nearest_range_m` collides with the scene
# feature of the same name, which is also `depth_range`'s target, and pandas
# silently renames both to _x/_y so the lookup finds neither.
OBJECT_QUANTITIES = ("obj_nearest_range_m", "obj_nearest_azimuth_deg",
                     "obj_nearest_radial_ms", "obj_n_illuminated")

# Task targets taken from the frame-level feature table rather than parsed out
# of the answer text. The text is what the model is graded on, but its first
# number is a different quantity per task and sometimes an object count where
# the task is really about geometry; these are the underlying quantities the
# answers are built from, so the regression measures dependence on the task
# rather than on a formatting choice.
TASK_TARGETS = {
    "det_objects_azdeg": ("n_tracks_in_fov", "전방 시야 안 물체 수"),
    "motion_seg": ("n_moving_in_fov", "움직이는 물체 수"),
    "depth_range": ("nearest_range_m", "가장 가까운 물체까지 거리"),
    "plan_ego": ("ego_yaw_rate_degs", "자차 요레이트 (조향 계획)"),
    "agent_traj": ("nearest_class_speed", "가장 가까운 물체의 접근 속도"),
    "radar_probe": ("lrr1_n_points", "레이더 검출 수 (지도학습된 스칼라)"),
    "radar_transfer": ("mrr2_n_points", "MRR 검출 수 (교차 센서)"),
}


def clip_vector(clip_id, row, nvidia_root):
    """All five structure quantities at every anchor frame of one clip."""
    short = next((r for r in RADAR_PREFERENCE if row.get(f"has_{r}", False)), None)
    if short is None or not row.get("has_radar_extrinsics", False):
        return []
    ego = read_member(nvidia_root, row["egomotion_zip"], row["egomotion_member"])
    derived = ego_frame(ego)
    radar = read_member(nvidia_root, row[f"radar_{short}_zip"],
                        row[f"radar_{short}_member"])
    full = pd.read_parquet(os.path.join(nvidia_root,
                                        row["radar_extrinsics_parquet"]))
    if clip_id not in full.index.get_level_values(0):
        return []
    extrinsics = full.loc[clip_id]

    out = []
    for frame in ANCHOR_FRAMES:
        scan = radar_scan(radar, extrinsics, SENSOR_NAME[short],
                          float(frame - 1), ego, derived)
        if scan is None:
            continue
        st = structure(scan)
        if st is None:
            continue
        out.append({
            "clip_id": clip_id, "frame": int(frame),
            "bearing_deg": st["bearing_deg"],
            "lateral_pct": 100.0 * st["n_left"] / (st["n_left"] + st["n_right"]),
            "spread_deg": st["spread_deg"],
            "closing_ms": st["closing_ms"],
            "near_far_pct": 100.0 * st["n_near"] / st["n_total"],
        })
    return out


def process(args_tuple):
    clip_id, row, nvidia_root = args_tuple
    try:
        return clip_vector(clip_id, row, nvidia_root)
    except Exception:
        return []


def build_vectors(nvidia_root, workers, limit, out_path):
    clips = pd.read_parquet(os.path.join(paths.COMMON_DIR, "nvidia_clips.parquet"))
    usable = clips[clips["has_egomotion"].fillna(False)
                   & clips["has_radar_extrinsics"].fillna(False)]
    if limit:
        usable = usable.iloc[:limit]
    log(f"{len(usable):,} clips -> structure vectors")
    tasks = [(cid, usable.loc[cid].to_dict(), nvidia_root) for cid in usable.index]
    rows, started = [], time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process, t) for t in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            rows.extend(f.result())
            if i % 2000 == 0 or i == len(tasks):
                rate = i / (time.monotonic() - started)
                log(f"  {i}/{len(tasks)}  {rate:.0f} clip/s")
    frame = pd.DataFrame(rows)
    frame["split"] = frame["clip_id"].map(clips["split"])
    frame.to_parquet(out_path, index=False)
    log(f"wrote {out_path}  rows={len(frame):,}")
    return frame


def ridge_r2(x_train, y_train, x_test, y_test, alpha):
    """Held-out R^2 of a ridge fit. Columns are standardised on the training
    split so the penalty is comparable across quantities on different scales."""
    mu, sd = x_train.mean(0), x_train.std(0)
    sd[sd == 0] = 1.0
    a, b = (x_train - mu) / sd, (x_test - mu) / sd
    a = np.hstack([a, np.ones((len(a), 1))])
    b = np.hstack([b, np.ones((len(b), 1))])
    penalty = alpha * np.eye(a.shape[1])
    penalty[-1, -1] = 0.0                       # never penalise the intercept
    w = np.linalg.solve(a.T @ a + penalty, a.T @ y_train)
    pred = b @ w
    resid = ((y_test - pred) ** 2).sum()
    total = ((y_test - y_test.mean()) ** 2).sum()
    return 1.0 - resid / total if total > 0 else 0.0


def compare_families(frame, targets):
    """Scan summary vs object-level, on identical rows and an identical split."""
    rng = np.random.default_rng(0)
    out = []
    for task, (column, label) in targets.items():
        if column not in frame.columns:
            continue
        cols = list(QUANTITIES) + list(OBJECT_QUANTITIES) + [column]
        sub = frame[[c for c in cols if c in frame.columns]].dropna()
        if len(sub) < 1000 or not set(OBJECT_QUANTITIES) <= set(sub.columns):
            continue
        y = sub[column].to_numpy(float)
        cut = rng.permutation(len(sub))
        tr, te = cut[: int(0.7 * len(cut))], cut[int(0.7 * len(cut)):]
        row = {"task": task, "label": label, "n": len(sub)}
        for name, feats in (("scan", QUANTITIES), ("object", OBJECT_QUANTITIES)):
            x = sub[list(feats)].to_numpy(float)
            row[name] = max(ridge_r2(x[tr], y[tr], x[te], y[te], a)
                            for a in (0.1, 1.0, 10.0, 100.0, 1000.0))
        out.append(row)
    return pd.DataFrame(out)


def coupling(frame, targets):
    """Unique contribution of each quantity: full R^2 minus leave-one-out R^2.

    Leave-one-out rather than the single-quantity R^2, because the five are
    correlated -- dense scans are wide scans -- and a quantity that only repeats
    what the others already say should score zero, not score twice.
    """
    rng = np.random.default_rng(0)
    results = []
    for task, (column, label) in targets.items():
        if column not in frame.columns:
            continue
        sub = frame[list(QUANTITIES) + [column]].dropna()
        if len(sub) < 500:
            continue
        x = sub[list(QUANTITIES)].to_numpy(float)
        y = sub[column].to_numpy(float)
        cut = rng.permutation(len(sub))
        train, test = cut[: int(0.7 * len(cut))], cut[int(0.7 * len(cut)):]
        best = max((ridge_r2(x[train], y[train], x[test], y[test], a), a)
                   for a in (0.1, 1.0, 10.0, 100.0, 1000.0))
        full_r2, alpha = best
        row = {"task": task, "label": label, "n": len(sub), "full_r2": full_r2}
        for i, q in enumerate(QUANTITIES):
            keep = [j for j in range(len(QUANTITIES)) if j != i]
            without = ridge_r2(x[train][:, keep], y[train], x[test][:, keep],
                               y[test], alpha)
            row[q] = full_r2 - without
        results.append(row)
    return pd.DataFrame(results)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nvidia-root", default=paths.NVIDIA_ROOT)
    ap.add_argument("--vectors", default=os.path.join(paths.COMMON_DIR,
                                                      "radar_structure_vectors.parquet"))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.rebuild or not os.path.exists(args.vectors):
        vec = build_vectors(args.nvidia_root, args.workers, args.limit, args.vectors)
    else:
        vec = pd.read_parquet(args.vectors)
        log(f"{len(vec):,} structure vectors from {args.vectors}")

    feats = pd.read_parquet(os.path.join(paths.COMMON_DIR,
                                         "scene_features_all_clips.parquet"))
    # The agent-trajectory target is a closing speed, which the feature table
    # carries only as ego speed and a nearest range; the change in that range
    # over the anchor spacing is the closest honest stand-in.
    feats = feats.sort_values(["clip_id", "frame"])
    feats["nearest_class_speed"] = feats.groupby("clip_id")["nearest_range_m"].diff()
    merged = vec.merge(feats, on=["clip_id", "frame"], how="inner")
    log(f"{len(merged):,} frames joined with scene features")

    table = coupling(merged, TASK_TARGETS)
    if table.empty:
        log("no task had enough joined rows")
        return 1

    log("")
    log("각 구조량의 고유 기여 (전체 R2 - 그 항만 뺀 R2). 클수록 그 태스크가 "
        "그 양 없이는 풀리지 않습니다.")
    header = f"{'task':16s} {'n':>7s} {'전체 R2':>8s}  " + \
             "".join(f"{q.split('_')[0]:>11s}" for q in QUANTITIES)
    log(header)
    for _, r in table.sort_values("full_r2", ascending=False).iterrows():
        log(f"{r['task']:16s} {int(r['n']):>7,} {r['full_r2']:>8.3f}  " +
            "".join(f"{r[q]:>11.3f}" for q in QUANTITIES))

    log("")
    log("태스크별로 가장 중요한 구조량:")
    for _, r in table.iterrows():
        top = max(QUANTITIES, key=lambda q: r[q])
        log(f"  {r['task']:16s} -> {top:14s} ({r[top]:+.3f})   {r['label']}")

    probes = os.path.join(paths.COMMON_DIR, "radar_object_probes.parquet")
    if os.path.exists(probes):
        obj = pd.read_parquet(probes, columns=[
            "clip_id", "frame", "n_illuminated", "nearest_range_m",
            "nearest_azimuth_deg", "nearest_radial_ms"]).drop_duplicates(
                ["clip_id", "frame"])
        obj = obj.rename(columns={c: f"obj_{c}" for c in obj.columns
                                  if c not in ("clip_id", "frame")})
        both = merged.merge(obj, on=["clip_id", "frame"], how="inner")
        fam = compare_families(both, TASK_TARGETS)
        if not fam.empty:
            log("")
            log("스캔 요약 vs 물체 단위 (같은 행, 같은 분할, 홀드아웃 R2)")
            log(f"{'task':16s} {'n':>9s} {'스캔 5개':>10s} {'물체 4개':>10s} {'비율':>8s}")
            for _, r in fam.sort_values("object", ascending=False).iterrows():
                ratio = r["object"] / r["scan"] if r["scan"] > 0.001 else float("inf")
                log(f"{r['task']:16s} {int(r['n']):>9,} {r['scan']:>10.3f} "
                    f"{r['object']:>10.3f} {ratio:>7.1f}x")

    out = args.out or os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "runs", "09_structure", "coupling.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    table.to_json(out, orient="records", indent=1, force_ascii=False)
    log(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
