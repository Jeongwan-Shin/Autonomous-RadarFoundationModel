#!/bin/bash
# Re-runs the joint stages once the align chain finishes.
#
# joint OOMed at micro-batch 8: align keeps no activations for a backward pass
# through the frozen language model, joint does, and the logits tensor alone is
# batch x seq x 151,936 upcast to fp32. The per-stage plan drops joint to
# micro-batch 2 with 16 accumulation steps and gradient checkpointing, holding
# the global batch at 160 so the tiers stay comparable.
set -u
CODE=/NHNHOME/workspace/AutonomousRadarFoundationModel
PY=/NHNHOME/workspace/venv/av/bin/python
CKPT=/NHNHOME/workspace/checkpoints
cd "$CODE"
MIX="radar_probe:3,qa:2,desc_radar:2,desc_complementarity:2,desc_objects:1,desc_ego_maneuver:1"
TASKS="qa,description,radar_probe"

echo "[$(date '+%H:%M:%S')] waiting for the align chain"
while pgrep -f run_all_tiers.sh > /dev/null; do sleep 60; done

wait_for_gpus() {
    for _ in $(seq 1 90); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
               | awk '{s+=$1} END {print s+0}')
        [ "$used" -lt 2000 ] && return 0
        sleep 10
    done
}

for model in 8B 32B; do
    out="$CKPT/vlm_${model}_joint"
    echo "[$(date '+%H:%M:%S')] === train $model / joint ==="
    wait_for_gpus
    $PY -m torch.distributed.run --nproc_per_node=5 -m training.train_vlm \
        --model "$model" --stage joint --epochs 1 --workers 5 --lr 5e-5 \
        --tasks "$TASKS" --samples 60000 --mixture "$MIX" --out "$out" \
        || { echo "FAILED: $model joint"; continue; }
    echo "[$(date '+%H:%M:%S')] === eval $model / joint ==="
    wait_for_gpus
    $PY -m training.eval_vlm --model "$model" --checkpoint "$out" \
        --tasks "$TASKS,ood_reasoning" --samples 1500 || echo "eval failed: $model"
done
echo "[$(date '+%H:%M:%S')] === JOINT STAGES DONE ==="
