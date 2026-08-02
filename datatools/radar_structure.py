#!/usr/bin/env python3
"""Radar questions the encoder was never told the answer to.

`radar_probe` turned out to be a circular measurement. The encoder is trained
with `head_stats` to write five scalars into the tokens the LLM reads --
log_n_points, log_n_moving, max_rcs, mean_rcs, max_range -- and two of the three
`radar_probe` question forms ask for exactly those. Split by form, the two
supervised ones contribute +0.84 and +0.80 over a shuffled-radar control while
the one unsupervised form, `illuminated`, contributes 0.00 in every run measured.
So the headline "+0.539 radar contribution" measures a scalar readout, not an
understanding of the scan.

`illuminated` alone is one probe, and one probe reading zero could be a quirk of
that question. These are five more, chosen so that no supervised statistic
answers them:

  bearing     azimuth of the nearest return -- where, not how many. Invariant to
              every count and every RCS aggregate.
  lateral     share of returns left of centre, as a percentage. Asked as two
              raw counts it was not honest: n_left correlated 0.84 with the
              supervised total, so reading the count and halving it scored
              +0.999. A percentage is scale-free and the count buys nothing.
  closing     radial velocity of the fastest approaching return. `n_moving` is
              supervised and is a count over a Doppler threshold; the speed
              itself is not, and the sign distinguishes approach from recession.
  near_far    share of returns inside 20 m, as a percentage, for the same
              reason. `max_range` is supervised, so the extent is known; how the
              returns distribute inside it is not.
  spread      azimuth span between the leftmost and rightmost return, in
              degrees. Purely angular -- there is no angular statistic in
              `head_stats` at all.

How clean each one actually is, measured on the 92k test probes as the largest
absolute correlation between the answer's first number and any of the three
supervised scalars the encoder is handed (n_points, n_moving, max_rcs):

    bearing   0.035   clean -- purely angular, no supervised scalar touches it
    lateral   0.123   clean, once asked as a percentage rather than two counts
    spread    0.458   partly; wide scans tend to be strong scans
    closing   0.562   partly; more movers, faster closing
    near_far  0.690   partly; dense scans put proportionally more returns far

So `bearing` and `lateral` carry the weight of the conclusion and the other
three are read with that caveat, not dropped -- a probe that a scalar half
explains still fails informatively if the model scores zero on it.

Evaluation only. Training on these would defeat their purpose: the point is to
ask what the model was never optimised to answer, so a gain here is transfer
rather than memorisation of another skeleton.

    python -m datatools.radar_structure --workers 14
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
from .frame_objects import (ANCHOR_FRAMES, RADAR_PREFERENCE, SENSOR_NAME, log,
                            ego_frame, radar_scan, read_member)

NEAR_M = 20.0
FAR_M = 60.0
MIN_POINTS = 12          # below this the answers are noise, not structure


def structure(scan):
    """The five quantities, from rig-frame points. None if the scan is too thin.

    `rig` is x forward, y left, so azimuth is atan2(y, x) and a positive bearing
    is to the left. Range is the planar distance -- elevation is dropped because
    the questions are all about the ground-plane layout the camera also sees,
    which is what makes a wrong answer informative rather than ambiguous.
    """
    rig = scan["rig"]
    if rig.shape[0] < MIN_POINTS:
        return None
    x, y = rig[:, 0], rig[:, 1]
    rng = np.hypot(x, y)
    az = np.degrees(np.arctan2(y, x))
    ahead = x > 0
    if ahead.sum() < MIN_POINTS:
        return None
    az, rng, radial = az[ahead], rng[ahead], scan["radial"][ahead]
    return {
        "bearing_deg": float(az[int(np.argmin(rng))]),
        "n_left": int((az > 0).sum()),
        "n_right": int((az <= 0).sum()),
        # Negative radial velocity is closing. If nothing is approaching, the
        # least-receding return is the honest answer rather than a fabricated 0.
        "closing_ms": float(radial.min()),
        "n_near": int((rng < NEAR_M).sum()),
        "n_total": int(rng.size),
        "spread_deg": float(az.max() - az.min()),
    }


PROBES = (
    ("bearing",
     "At what azimuth is the nearest radar return, in degrees, positive left?",
     lambda s: f"{s['bearing_deg']:+.0f} deg."),
    ("lateral",
     "What percentage of the radar returns are left of centre?",
     lambda s: f"{100.0 * s['n_left'] / (s['n_left'] + s['n_right']):.0f}%."),
    ("closing",
     "What is the radial velocity of the fastest approaching radar return, in "
     "m/s, negative closing?",
     lambda s: f"{s['closing_ms']:+.1f} m/s."),
    ("near_far",
     "What percentage of the radar returns are within 20 m?",
     lambda s: f"{100.0 * s['n_near'] / s['n_total']:.0f}%."),
    ("spread",
     "How many degrees of azimuth separate the leftmost and rightmost radar "
     "return?",
     lambda s: f"{s['spread_deg']:.0f} deg."),
)


def clip_items(clip_id, row, nvidia_root):
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

    items = []
    for frame in ANCHOR_FRAMES:
        t_s = float(frame - 1)
        scan = radar_scan(radar, extrinsics, SENSOR_NAME[short], t_s, ego,
                          derived)
        if scan is None:
            continue
        stats = structure(scan)
        if stats is None:
            continue
        # One form per frame, as `radar_probe` does, so a clip cannot be
        # answered by copying its own earlier answer under a different label.
        # The anchors are 6/11/16 and all three are 1 mod 5, so rotating on the
        # frame alone would only ever emit the second form; the clip digest
        # breaks that alignment while staying deterministic across workers,
        # which Python's salted hash() would not.
        digest = int(hashlib.sha1(clip_id.encode()).hexdigest()[:8], 16)
        form, question, answer = PROBES[(digest + frame) % len(PROBES)]
        items.append({"clip_id": clip_id, "task": "radar_structure",
                      "form": form, "frame": int(frame),
                      "prompt": f"At frame {frame}. {question}",
                      "target": answer(stats), "sensor": short})
    return items



def process(args_tuple):
    clip_id, row, nvidia_root = args_tuple
    try:
        return clip_items(clip_id, row, nvidia_root)
    except Exception as exc:
        return [{"clip_id": clip_id, "task": None, "form": "",
                 "frame": -1, "prompt": "",
                 "target": f"{type(exc).__name__}: {str(exc)[:80]}", "sensor": ""}]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nvidia-root", default=paths.NVIDIA_ROOT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    clips = pd.read_parquet(os.path.join(paths.COMMON_DIR, "nvidia_clips.parquet"))
    usable = clips[clips["has_egomotion"].fillna(False)
                   & clips["has_radar_extrinsics"].fillna(False)]
    if args.limit:
        usable = usable.iloc[: args.limit]
    out_path = args.out or os.path.join(paths.COMMON_DIR,
                                        "radar_structure_probes.parquet")
    log(f"{len(usable):,} clips with a front radar x {len(ANCHOR_FRAMES)} anchors")

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
                log(f"  {i}/{len(tasks)}  {rate:.1f} clip/s  "
                    f"ETA {(len(tasks) - i) / rate / 60:.1f} min")

    frame = pd.DataFrame(rows)
    frame["split"] = frame["clip_id"].map(clips["split"])
    frame.to_parquet(out_path, index=False)
    log(f"wrote {out_path}  rows={len(frame):,}")
    if errors:
        log(f"clips that failed: {errors}")
    for form, group in frame.groupby("form"):
        log(f"  {form:10s} {len(group):>8,}   e.g. {group['target'].iloc[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
