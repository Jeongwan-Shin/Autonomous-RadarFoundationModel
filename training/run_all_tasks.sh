#!/bin/bash
# All eleven tasks as one instruction-following model, per tier.
#
# What changed against run_all_tiers.sh, and why:
#
#   --tasks all       the earlier runs trained tasks 10 and 11 only. Tasks 01-06
#                     are now pre-serialised by datatools.frame_objects, and 07
#                     and 09 are derived in the loader, so the instruction is the
#                     only thing that distinguishes them.
#   --all-profiles    sensor profiles are a property of the rig: 43% of clips
#                     carry the imaging LRR, 51% only the SRR, 7% no front radar.
#                     Training across all three is task 08 as the release poses
#                     it. Note the radar encoder was pretrained on LRR alone, so
#                     the SRR clips are a genuine domain shift -- that is task 07.
#   --radar-dropout   the previous joint checkpoints ignored the radar: blanking
#                     it moved 32B's loss by 0.0002 nll. Blanking it during
#                     training, with radar-only targets replaced by a refusal,
#                     puts a cost on answering those from the language prior.
#
# The mixture lives in instruct_data.DEFAULT_MIXTURE now rather than being passed
# per run, because it has to name sixteen item types instead of six.
set -u
CODE=/NHNHOME/workspace/AutonomousRadarFoundationModel
PY=/NHNHOME/workspace/venv/av/bin/python
CKPT=/NHNHOME/workspace/checkpoints
cd "$CODE"

SAMPLES=${SAMPLES:-60000}
# 3,200 rather than the 1,500 the six-task runs used: sixteen item types share
# the budget, and 1,500 would leave under 100 items per task to read a per-task
# ablation from.
EVAL_SAMPLES=${EVAL_SAMPLES:-3200}
DROPOUT=${DROPOUT:-0.25}
MODELS=${MODELS:-"8B 32B"}

wait_for_gpus() {
    for _ in $(seq 1 90); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
               | awk '{s+=$1} END {print s+0}')
        [ "$used" -lt 2000 ] && return 0
        sleep 10
    done
    echo "WARNING: GPUs still busy, continuing anyway"
}

train() {
    local model=$1 stage=$2 lr=$3
    local out="$CKPT/vlm_${model}_${stage}_alltasks"
    echo "[$(date '+%H:%M:%S')] === train $model / $stage ==="
    wait_for_gpus
    $PY -m torch.distributed.run --nproc_per_node=5 -m training.train_vlm \
        --model "$model" --stage "$stage" --epochs 1 --workers 5 --lr "$lr" \
        --tasks all --all-profiles --radar-dropout "$DROPOUT" \
        --samples "$SAMPLES" --out "$out" \
        || { echo "FAILED: $model $stage"; return 1; }
}

assess() {
    local model=$1 stage=$2
    local out="$CKPT/vlm_${model}_${stage}_alltasks"
    echo "[$(date '+%H:%M:%S')] === eval $model / $stage ==="
    wait_for_gpus
    # ood_reasoning is added here and nowhere else: it is the only human-verified
    # text in the release and nothing is fitted to it.
    $PY -m training.eval_vlm --model "$model" --checkpoint "$out" \
        --tasks all,ood_reasoning --all-profiles --samples "$EVAL_SAMPLES" \
        || echo "eval failed for $model $stage"
}

for model in $MODELS; do
    train "$model" align 1e-4 && assess "$model" align
    train "$model" joint 5e-5 && assess "$model" joint
done
echo "[$(date '+%H:%M:%S')] === ALL TASKS DONE ==="

for f in "$CKPT"/vlm_*_alltasks/eval.json; do
    [ -f "$f" ] || continue
    echo "--- $f"
    $PY - "$f" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
tasks = sorted(d.get("full", {}))
print(f"  {'task':22s}{'full':>9s}{'zeroed':>9s}{'shuffled':>9s}{'z-f':>9s}")
for t in tasks:
    f = d["full"][t]["loss"]
    z = d.get("zeroed", {}).get(t, {}).get("loss")
    s = d.get("shuffled", {}).get(t, {}).get("loss")
    cells = "".join(f"{v:9.4f}" if v is not None else f"{'-':>9s}" for v in (f, z, s))
    print(f"  {t:22s}{cells}{(z - f) if z is not None else 0:9.4f}")
EOF
done
