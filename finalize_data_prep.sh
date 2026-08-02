#!/bin/bash
# Runs everything that still depends on the world-frame track rescan, then
# verifies the result and leaves a DONE marker.
#
# Order matters: build_task_splits regenerates all ten task folders, and
# fix_track_splits then overwrites 02 and 03_2 with the world-frame filters.
# Running them the other way round would silently reinstate the rig-frame
# filter that selected the wrong tracks.
#
#   bash finalize_data_prep.sh [rescan_log_path]
set -u
CODE=/NHNHOME/workspace/AutonomousRadarFoundationModel
SPLIT=/NHNHOME/workspace/dataset/raw_Auto_datasets/preprocessed_train_test_split
LEGACY=/home/dgist_sks0/workspace/RaViLa_AutoDriver_Foundation_Model
RESCAN_LOG="${1:-$LEGACY/rescan_tracks.log}"
MARKER="$SPLIT/DATA_PREP_DONE"

cd "$CODE"
rm -f "$MARKER"

echo "[$(date '+%H:%M:%S')] waiting for rescan: $RESCAN_LOG"
until grep -q '^\[.*\] done$' "$RESCAN_LOG" 2>/dev/null; do sleep 60; done
echo "[$(date '+%H:%M:%S')] rescan finished"

echo "=== 1/4 regenerating all task splits (picks up radar extrinsics columns) ==="
python3 -m datatools.build_task_splits || exit 1

echo "=== 2/4 rebuilding 02 and 03_2 on world-frame motion ==="
python3 -m datatools.fix_track_splits || exit 1

echo "=== 3/4 regenerating documentation ==="
python3 -m datatools.make_readme || exit 1

echo "=== 4/4 verifying ==="
python3 - <<'PY'
import glob, json, os
import pandas as pd
import pyarrow.parquet as pq

SPLIT = "/NHNHOME/workspace/dataset/raw_Auto_datasets/preprocessed_train_test_split"
problems = []

# Tasks 10 and 11 are filled from another machine, so they are reported rather
# than required to look any particular way. Only the ten generated folders must
# actually contain parquet.
expected = sorted(d for d in os.listdir(SPLIT)
                  if os.path.isdir(os.path.join(SPLIT, d)) and d[0].isdigit())
print(f"task folders: {len(expected)}")
for name in expected:
    files = glob.glob(os.path.join(SPLIT, name, "*.parquet"))
    external = name.startswith(("10_", "11_"))
    if not external and not files:
        problems.append(f"{name}: no parquet files")
    rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    if external:
        items = [n for n in os.listdir(os.path.join(SPLIT, name)) if n != "README.md"]
        print(f"  {name:38s} {len(items):2d} items  {rows:>10,} parquet rows"
              f"   (external{', empty' if not items else ''})")
    else:
        print(f"  {name:38s} {len(files):2d} files  {rows:>10,} rows")

# splits must not overlap on clip_id
for name in expected:
    if name.startswith(("10_", "11_")):
        continue  # externally produced; their split semantics are not ours
    seen = {}
    for split in ("train", "val", "test"):
        path = os.path.join(SPLIT, name, f"nvidia_{split}.parquet")
        if os.path.exists(path):
            seen[split] = set(pd.read_parquet(path, columns=["clip_id"])["clip_id"])
    keys = list(seen)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            overlap = seen[keys[i]] & seen[keys[j]]
            if overlap:
                problems.append(f"{name}: {keys[i]}/{keys[j]} share "
                                f"{len(overlap):,} clip_ids")

# the world-frame filter must actually be in force
cfg = json.load(open(os.path.join(SPLIT, "03_2_agent_trajectory/config.json")))
if cfg.get("nvidia_filter", {}).get("motion_frame", "").startswith("world"):
    print("03_2 filter frame: world  OK")
else:
    problems.append("03_2 is not using the world-frame filter")

tracks = os.path.join(SPLIT, "common/nvidia_tracks_world.parquet")
if os.path.exists(tracks):
    t = pd.read_parquet(tracks, columns=["n_obs", "is_stationary",
                                         "frac_in_front_fov"])
    long = t[t["n_obs"] >= 30]
    print(f"tracks: {len(t):,} total, {len(long):,} >= 3 s, "
          f"{long['is_stationary'].mean()*100:.1f}% stationary, "
          f"{(long['frac_in_front_fov'] >= 0.5).mean()*100:.1f}% mostly in FOV")
else:
    problems.append("common/nvidia_tracks_world.parquet missing")

if problems:
    print("\nPROBLEMS:")
    for p in problems:
        print(f"  - {p}")
    raise SystemExit(1)
print("\nall checks passed")
PY
status=$?

if [ $status -ne 0 ]; then
    echo "=== VERIFICATION FAILED ==="
    exit 1
fi

# The legacy checkout only still existed so the running job could keep writing
# its log there. Once the rescan is done it is safe to drop.
if [ -d "$LEGACY" ]; then
    cp -n "$LEGACY"/*.log "$CODE"/ 2>/dev/null
    rm -rf "$LEGACY"
    echo "removed legacy checkout $LEGACY (logs copied here)"
fi

date '+%Y-%m-%d %H:%M:%S' > "$MARKER"
echo "=== DATA PREP COMPLETE ==="
