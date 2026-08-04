#!/usr/bin/env python3
"""Object-level radar questions -- the granularity the tasks actually use.

`radar_structure.py` asked five questions about the scan as a whole and the
answer came back near zero on the two uncontaminated ones. That was honest but
it was the wrong thing to ask. Measured on 346,565 objects with both feature
sets on the same rows and the same split, the scan summary explains almost none
of what a task has to say about an object, while the returns inside that
object's own box explain nearly all of it:

                       scan summary   per-object returns
    range to object          0.078          0.999
    azimuth of object        0.007          0.985
    moving or stationary     0.084          0.316

So the information is in the radar, essentially perfectly, and it survives only
at object granularity. Summarising a scan into one bearing and one spread throws
away exactly what `det_objects`, `depth_range` and `motion_seg` are functions of.

These probes ask at that granularity. Every one is about the object the radar
illuminates, not the object the camera sees -- radar illuminates 85.7% of heavy
trucks and 20.2% of people, so which object counts as radar-confirmed is itself
unavailable to the camera. That is what stops these from being monocular
guesswork, and the shuffled-radar control is what checks it.

None of the five supervised scalars in `head_stats` is per-object, so nothing
here is the circular measurement `radar_probe` turned out to be. Contamination
is measured rather than asserted -- see `--report`.

Evaluation only, like `radar_structure`. If a later run trains on some of these,
the ones it trains on stop being instruments and must leave the report.

    python -m datatools.radar_objects_probe --workers 12
"""

import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import paths
from .frame_objects import (ANCHOR_FRAMES, RADAR_PREFERENCE, SENSOR_NAME,
                            boxes_world, ego_frame, log, radar_scan, read_member,
                            visible_at)
from .object_coupling import object_evidence

MIN_RETURNS = 2          # one return is a position estimate, not a detection


def illuminated(scan, rows):
    """Objects the radar actually sees, nearest first, with what it measures.

    Sorted by the radar's own range rather than the label's, because the whole
    point is to ask what the sensor reports. The two agree closely -- that is
    the 1.31 m median calibration check -- but where they differ the sensor is
    what the question is about.
    """
    evidence = object_evidence(scan, rows)
    out = []
    for r in rows.itertuples():
        ev = evidence.get(r.track_id)
        if ev is None or ev["hit_points"] < MIN_RETURNS:
            continue
        out.append({**ev, "label_class": r.label_class,
                    "label_range_m": float(r.range_m)})
    return sorted(out, key=lambda o: o["hit_range_m"])


PROBES = (
    ("obj_range",
     "How far away is the nearest object the radar illuminates, in metres?",
     lambda o: len(o) >= 1,
     lambda o: f"{o[0]['hit_range_m']:.0f} m."),
    ("obj_azimuth",
     "At what azimuth is the nearest object the radar illuminates, in degrees, "
     "positive left?",
     lambda o: len(o) >= 1,
     lambda o: f"{o[0]['hit_azimuth_deg']:+.0f} deg."),
    ("obj_closing",
     "What is the radial velocity of the nearest object the radar illuminates, "
     "in m/s, negative closing?",
     lambda o: len(o) >= 1,
     lambda o: f"{o[0]['hit_radial']:+.1f} m/s."),
    # Two objects, not one: a model that has learned "the nearest thing is about
    # 20 m" scores well on obj_range without reading anything. A gap between the
    # first and second cannot be answered that way.
    ("obj_gap",
     "How much farther away is the second nearest radar-illuminated object than "
     "the nearest, in metres?",
     lambda o: len(o) >= 2,
     lambda o: f"{o[1]['hit_range_m'] - o[0]['hit_range_m']:.0f} m."),
    ("obj_moving",
     "Of the objects the radar illuminates, how many have moving returns?",
     lambda o: len(o) >= 1,
     lambda o: f"{sum(1 for x in o if x['hit_moving'] > 0)} of {len(o)}."),
)


def clip_items(clip_id, row, nvidia_root):
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

    digest = int(hashlib.sha1(clip_id.encode()).hexdigest()[:8], 16)
    items = []
    for frame in ANCHOR_FRAMES:
        t_s = float(frame - 1)
        scan = radar_scan(radar, extrinsics, SENSOR_NAME[short], t_s, ego, derived)
        if scan is None:
            continue
        rows = visible_at(boxes, t_s)
        if rows is None or rows.empty:
            continue
        objects = illuminated(scan, rows)
        if not objects:
            continue
        # Rotate on the clip digest, not the frame: the anchors are 6/11/16 and
        # all three are 1 mod 5, so rotating on the frame alone would only ever
        # emit the second form.
        order = [(digest + frame + i) % len(PROBES) for i in range(len(PROBES))]
        for idx in order:
            form, question, ready, answer = PROBES[idx]
            if not ready(objects):
                continue        # e.g. obj_gap with only one illuminated object
            items.append({
                "clip_id": clip_id, "task": "radar_objects", "form": form,
                "frame": int(frame),
                "prompt": f"At frame {frame}. {question}",
                "target": answer(objects), "sensor": short,
                "n_illuminated": len(objects),
                "nearest_range_m": objects[0]["hit_range_m"],
                "nearest_azimuth_deg": objects[0]["hit_azimuth_deg"],
                "nearest_radial_ms": objects[0]["hit_radial"],
            })
            break
    return items


def process(args_tuple):
    clip_id, row, nvidia_root = args_tuple
    try:
        return clip_items(clip_id, row, nvidia_root)
    except Exception:
        return []


def report(path):
    """Contamination per form: is any supervised scalar already the answer?"""
    import re
    number = re.compile(r"-?\d+(?:\.\d+)?")
    pr = pd.read_parquet(path)
    pr = pr[pr["split"] == "test"]
    sf = pd.read_parquet(os.path.join(paths.COMMON_DIR,
                                      "scene_features_all_clips.parquet"),
                         columns=["clip_id", "frame", "lrr1_n_points",
                                  "lrr1_n_moving", "lrr1_max_rcs"])
    m = pr.merge(sf, on=["clip_id", "frame"]).dropna(subset=["lrr1_n_points"])
    m = m.assign(first=m["target"].map(lambda s: float(number.findall(s)[0])))
    log("")
    log(f"{'form':14s} {'n':>9s} {'오염도':>9s}   판정")
    for form, g in m.groupby("form"):
        worst = max(abs(float(np.corrcoef(g["first"], g[c])[0, 1]))
                    for c in ("lrr1_n_points", "lrr1_n_moving", "lrr1_max_rcs"))
        log(f"{form:14s} {len(g):>9,} {worst:>9.3f}   "
            f"{'깨끗' if worst < 0.3 else '부분적' if worst < 0.7 else '오염'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nvidia-root", default=paths.NVIDIA_ROOT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", action="store_true",
                    help="skip the build, just re-measure contamination")
    args = ap.parse_args(argv)

    out_path = args.out or os.path.join(paths.COMMON_DIR,
                                        "radar_object_probes.parquet")
    if args.report:
        report(out_path)
        return 0

    clips = pd.read_parquet(os.path.join(paths.COMMON_DIR, "nvidia_clips.parquet"))
    usable = clips[clips["has_egomotion"].fillna(False)
                   & clips["has_obstacle"].fillna(False)
                   & clips["has_radar_extrinsics"].fillna(False)]
    if args.limit:
        step = max(1, len(usable) // args.limit)
        usable = usable.iloc[::step].iloc[: args.limit]
    log(f"{len(usable):,} clips x {len(ANCHOR_FRAMES)} anchors")

    tasks = [(cid, usable.loc[cid].to_dict(), args.nvidia_root)
             for cid in usable.index]
    rows, started = [], time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, t) for t in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            rows.extend(f.result())
            if i % 5000 == 0 or i == len(tasks):
                log(f"  {i}/{len(tasks)}  "
                    f"{i/(time.monotonic()-started):.0f} clip/s  {len(rows):,} items")

    frame = pd.DataFrame(rows)
    frame["split"] = frame["clip_id"].map(clips["split"])
    frame.to_parquet(out_path, index=False)
    log(f"wrote {out_path}  rows={len(frame):,}")
    for form, g in frame.groupby("form"):
        log(f"  {form:14s} {len(g):>8,}   e.g. {g['target'].iloc[0]}")
    report(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
