#!/usr/bin/env python3
"""Record, per clip, whether radar extrinsics exist and where to find them.

Radar poses come from the non-offline `calibration/sensor_extrinsics/`, and they
are not universal: sampling 25 chunks, 93.1% of clips carry radar entries. A
clip without them cannot have its radar returns lifted into the rig frame, so
every radar task needs to know before it starts rather than discovering it in
the data loader.

Adds three columns to `common/nvidia_clips.parquet`:

  radar_extrinsics_parquet   chunk-level file, keyed by (clip_id, sensor_name)
  has_radar_extrinsics       any radar entry for this clip
  radar_extrinsics_sensors   which front radars are covered, comma separated

    python -m datatools.index_radar_extrinsics
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from . import paths

FEATURE = "calibration/sensor_extrinsics"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scan_chunk(chunk, nvidia_root):
    path = os.path.join(nvidia_root, FEATURE,
                        f"sensor_extrinsics.chunk_{chunk:04d}.parquet")
    if not os.path.exists(path):
        return {}
    frame = pd.read_parquet(path).reset_index()
    front = frame[frame["sensor_name"].isin(paths.RADARS)]
    return (front.groupby("clip_id")["sensor_name"]
            .apply(lambda s: ",".join(sorted(paths.RADAR_SHORT[x] for x in s)))
            .to_dict())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nvidia-root", default=paths.NVIDIA_ROOT)
    ap.add_argument("--common-dir", default=paths.COMMON_DIR)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv)

    clips_path = os.path.join(args.common_dir, "nvidia_clips.parquet")
    clips = pd.read_parquet(clips_path)
    chunks = sorted(clips["chunk"].unique())
    log(f"scanning {len(chunks)} extrinsics files for {len(clips):,} clips")

    covered = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(scan_chunk, int(c), args.nvidia_root) for c in chunks]
        for future in as_completed(futures):
            covered.update(future.result())

    tag = clips["chunk"].map(lambda c: f"chunk_{c:04d}")
    clips["radar_extrinsics_parquet"] = f"{FEATURE}/sensor_extrinsics." + tag + ".parquet"
    clips["radar_extrinsics_sensors"] = clips.index.map(covered).fillna("")
    clips["has_radar_extrinsics"] = clips["radar_extrinsics_sensors"] != ""

    clips.to_parquet(clips_path)
    log(f"wrote {clips_path}")
    log(f"  clips with radar extrinsics: {clips['has_radar_extrinsics'].mean()*100:.1f}%")

    # A clip that has radar data but no pose for it is unusable for any task
    # that needs radar in the rig frame; surface the count rather than let it
    # silently shrink a training set later.
    has_radar = clips["n_front_radars"] > 0
    gap = has_radar & ~clips["has_radar_extrinsics"]
    log(f"  clips with radar data:              {int(has_radar.sum()):,}")
    log(f"  of those, missing radar extrinsics: {int(gap.sum()):,} "
        f"({gap.sum()/max(has_radar.sum(),1)*100:.1f}%)")
    for sensors, count in clips["radar_extrinsics_sensors"].value_counts().items():
        log(f"    [{sensors or 'none'}] {count:,}")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
