#!/bin/bash
# Train and evaluate every tier, in order, on the 5 available devices.
#
# Sequential rather than concurrent: 32B and 30B-A3B each need all five devices,
# and 8B at micro-batch 8 already peaks near 110 GiB. Sharing would force the
# batch down and break the comparison, which only holds because every tier runs
# at a global batch of 160.
#
# The system torchrun launches workers with /usr/bin/python and would not find
# transformers, so the venv interpreter drives torch.distributed.run directly.
set -u
CODE=/NHNHOME/workspace/AutonomousRadarFoundationModel
PY=/NHNHOME/workspace/venv/av/bin/python
CKPT=/NHNHOME/workspace/checkpoints
cd "$CODE"

SAMPLES=${SAMPLES:-60000}
EVAL_SAMPLES=${EVAL_SAMPLES:-1500}
# `desc_radar` and `desc_complementarity` are the only text that puts a loss on
# the radar pathway, so they carry double weight. ood_reasoning is left out of
# training entirely: 360 items, and the only human-verified text in the release,
# so it is worth more as an untouched evaluation set.
MIX="radar_probe:3,qa:2,desc_radar:2,desc_complementarity:2,desc_objects:1,desc_ego_maneuver:1"
TASKS="qa,description,radar_probe"

wait_for_gpus() {
    # A previous run's processes can outlive the python exit briefly; starting
    # the next tier on top of them would OOM.
    for _ in $(seq 1 60); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
               | awk '{s+=$1} END {print s+0}')
        [ "$used" -lt 2000 ] && return 0
        sleep 10
    done
    echo "WARNING: GPUs still busy, continuing anyway"
}

train() {
    local model=$1 stage=$2 extra=${3:-}
    local out="$CKPT/vlm_${model}_${stage}"
    echo "[$(date '+%H:%M:%S')] === train $model / $stage ==="
    wait_for_gpus
    $PY -m torch.distributed.run --nproc_per_node=5 -m training.train_vlm \
        --model "$model" --stage "$stage" --epochs 1 --workers 5 \
        --tasks "$TASKS" --samples "$SAMPLES" --mixture "$MIX" \
        --out "$out" $extra || { echo "FAILED: $model $stage"; return 1; }
}

assess() {
    local model=$1 stage=$2
    local out="$CKPT/vlm_${model}_${stage}"
    echo "[$(date '+%H:%M:%S')] === eval $model / $stage ==="
    wait_for_gpus
    $PY -m training.eval_vlm --model "$model" --checkpoint "$out" \
        --tasks "$TASKS,ood_reasoning" --samples "$EVAL_SAMPLES" \
        || echo "eval failed for $model $stage"
}

train 8B align && assess 8B align

train 8B joint "--lr 5e-5" && assess 8B joint
train 32B align && assess 32B align
train 30B-A3B align && assess 30B-A3B align

echo "[$(date '+%H:%M:%S')] === ALL TIERS DONE ==="
for d in "$CKPT"/vlm_*/eval.json; do
    [ -f "$d" ] && echo "--- $d" && $PY -c "
import json,sys
s=json.load(open('$d'))
for mode in s:
    for task,v in sorted(s[mode].items()):
        metric = f\"{v['accuracy']*100:.1f}% acc\" if 'accuracy' in v else f\"{v['loss']:.4f} nll\"
        print(f'  {mode:10s} {task:22s} {metric}')
"
done
