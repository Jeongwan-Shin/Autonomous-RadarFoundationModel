#!/bin/bash
# Which training recipe keeps the model reading its radar?
#
# The finding this exists to act on: with the encoder fixed, `align` reaches a
# shuffled-minus-full gap of +0.0491 and a radar/camera selectivity of 3.89, and
# then every recipe that lets the language model adapt destroys it -- LoRA to
# +0.0017, full fine-tuning to +0.0002. The encoder is not the problem. The
# language model finds the camera and the ego stream easier than the radar, and
# takes them.
#
# Four recipes, each on a fifth of an epoch so the comparison costs an hour
# rather than eight. Short runs cannot settle final quality, but the collapse
# they are being screened for happened within 40 steps in every run so far.
#
#   full_base            the collapse, reproduced at this budget as the control
#   full_resume          starts from the align run's connector instead of noise
#   full_contrast        a hinge requiring another clip's radar to cost 0.15 nats
#   full_resume_contrast both
#
# The routed encoder is not in the search. Probed, it retains less than the dense
# one it was meant to improve on -- 0.648 / 0.596 / 0.716 against
# 0.706 / 0.626 / 0.875 -- so spending a slot on it would be testing a
# representation already known to be worse.
#
# Judged on selectivity, not on loss: every one of these will improve the loss.
set -u
CODE=/NHNHOME/workspace/AutonomousRadarFoundationModel
PY=/NHNHOME/workspace/venv/av/bin/python
CKPT=/NHNHOME/workspace/checkpoints
cd "$CODE"

SAMPLES=${SAMPLES:-20000}
EVAL_SAMPLES=${EVAL_SAMPLES:-2400}
MODEL=${MODEL:-8B}
ENC_DENSE=$CKPT/radar_encoder_multi/encoder.pt
ENC_MOE=$CKPT/radar_encoder_multi_moe/encoder.pt
ALIGN=$CKPT/vlm_8B_align_fixed

wait_for_gpus() {
    for _ in $(seq 1 120); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
               | awk '{s+=$1} END {print s+0}')
        [ "$used" -lt 2000 ] && return 0
        sleep 10
    done
    echo "WARNING: GPUs still busy, continuing anyway"
}

recipe() {
    local name=$1 encoder=$2; shift 2
    local out="$CKPT/vlm_${MODEL}_${name}"
    echo "[$(date '+%H:%M:%S')] === train $name : $* ==="
    wait_for_gpus
    $PY -m torch.distributed.run --nproc_per_node=5 -m training.train_vlm \
        --model "$MODEL" --stage full --epochs 1 --workers 5 \
        --tasks all --all-profiles --radar-dropout 0.25 \
        --radar-checkpoint "$encoder" --master-dtype bf16 \
        --samples "$SAMPLES" --out "$out" "$@" \
        || { echo "FAILED: $name"; return 1; }

    echo "[$(date '+%H:%M:%S')] === eval $name ==="
    wait_for_gpus
    $PY -m training.eval_vlm --model "$MODEL" --checkpoint "$out" \
        --tasks all,ood_reasoning --all-profiles --samples "$EVAL_SAMPLES" \
        || echo "eval failed: $name"
}

# A 2x2 over the two candidate causes. `--resume` matters because until today
# the connector was rebuilt from noise in every stage after `align`, so the
# language model met a random projection of the radar on step 1 and had every
# reason to route around it. The hinge attacks the same collapse from the loss
# side instead of the initialisation side.
recipe full_base            "$ENC_DENSE" --lr 2e-5
recipe full_resume          "$ENC_DENSE" --lr 2e-5 --resume "$ALIGN"
recipe full_contrast        "$ENC_DENSE" --lr 2e-5 \
    --radar-contrast 1.0 --radar-margin 0.15
recipe full_resume_contrast "$ENC_DENSE" --lr 2e-5 --resume "$ALIGN" \
    --radar-contrast 1.0 --radar-margin 0.15

echo "[$(date '+%H:%M:%S')] === RECIPE SEARCH DONE ==="
$PY -m training.compare_runs
