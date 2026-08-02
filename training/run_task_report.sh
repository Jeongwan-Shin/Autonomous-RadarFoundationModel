#!/bin/bash
# Score every task on a finished checkpoint, spread over the five devices.
#
# Generation runs one item at a time -- batching a decoder-only model needs left
# padding, and left padding moves the video and radar placeholders that the
# injector locates by position. So the parallelism has to come from running
# different tasks at once rather than different items.
#
# Tasks are grouped by how long their answers are. Object lists generate up to
# 260 tokens and dominate the wall clock, so they are split across devices
# rather than queued behind each other.
set -u
CODE=/NHNHOME/workspace/AutonomousRadarFoundationModel
PY=/NHNHOME/workspace/venv/av/bin/python
cd "$CODE"

CKPT=${CKPT:-/NHNHOME/workspace/checkpoints/vlm_8B_long_base}
ITEMS=${ITEMS:-200}
SPLIT=${SPLIT:-val}
NAME=$(basename "$CKPT")
OUT=runs/06_task_report
mkdir -p "$OUT"

# One line per device. The long-answer tasks lead each group.
# Not `GROUPS`: bash keeps that name for the caller's group ids, assigning to it
# is silently ignored, and `${GROUPS[0]}` then returns a real group id. Every
# device was launched with `--tasks 1999`.
TASK_GROUPS=(
  "det_objects,radar_probe"
  "track_identity,radar_transfer"
  "motion_seg,retrieval"
  "desc_radar,plan_ego,agent_traj"
  "desc_complementarity,depth_range,world_model"
)

echo "[$(date '+%H:%M:%S')] waiting for training to finish"
while pgrep -f "training\.train_vlm" > /dev/null; do sleep 60; done
for _ in $(seq 1 120); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
           | awk '{s+=$1} END {print s+0}')
    [ "$used" -lt 2000 ] && break
    sleep 10
done

pids=""
for i in 0 1 2 3 4; do
    tasks=${TASK_GROUPS[$i]}
    echo "[$(date '+%H:%M:%S')] device $i: $tasks"
    CUDA_VISIBLE_DEVICES=$i $PY -m training.eval_all_tasks \
        --checkpoint "$CKPT" --tasks "$tasks" --split "$SPLIT" \
        --items "$ITEMS" --workers 3 --show 3 \
        --out "$OUT/${NAME}_${SPLIT}_g${i}.json" \
        > "$OUT/${NAME}_${SPLIT}_g${i}.log" 2>&1 < /dev/null &
    pids="$pids $!"
    sleep 15
done
for pid in $pids; do wait "$pid"; done

echo "[$(date '+%H:%M:%S')] === TASK REPORT DONE ==="
$PY - "$OUT" "$NAME" "$SPLIT" <<'EOF'
import glob, json, os, sys
out, name, split = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for path in sorted(glob.glob(os.path.join(out, f"{name}_{split}_g*.json"))):
    rows.extend(json.load(open(path)))

HEAD = {"detection": ("f1", "F1", 100, "%"),
        "waypoints": ("displacement_mae_m", "position err", 1, " m"),
        "trajectory": ("range_mae_m", "range err", 1, " m"),
        "quantity": ("corr", "correlation", 1, ""),
        "tags": ("f1", "F1", 100, "%"),
        "choice": ("accuracy", "accuracy", 100, "%")}
print(f"\n{'task':22s}{'metric':>14s}{'n':>6s}{'full':>12s}{'shuffled':>12s}{'radar':>10s}")
print("-" * 76)
for r in sorted(rows, key=lambda r: r["task"]):
    full = r.get("full") or {}
    spec = HEAD.get(full.get("metric"))
    if not spec:
        print(f"{r['task']:22s}{'loss only':>14s}{r['n']:>6d}")
        continue
    key, label, scale, unit = spec
    a, b = full.get(key), (r.get("shuffled") or {}).get(key)
    lower_better = key.endswith("_m")
    delta = ("--" if a is None or b is None else
             f"{((b - a) if lower_better else (a - b)) * scale:+.1f}")
    print(f"{r['task']:22s}{label:>14s}{r['n']:>6d}"
          f"{'--' if a is None else f'{a*scale:.1f}{unit}':>12s}"
          f"{'--' if b is None else f'{b*scale:.1f}{unit}':>12s}{delta:>10s}")
print("\n  radar = full - shuffled, signed so positive always means the radar helped")
EOF
