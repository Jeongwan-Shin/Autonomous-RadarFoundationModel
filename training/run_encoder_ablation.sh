#!/bin/bash
# Two encoder variants, probed rather than trained into a language model.
#
# Both address the same gap: pretraining has only ever seen `lrr1` clips, while
# the instruction model with --all-profiles is fed SRR on 51% of them. The
# encoder has therefore been meeting a sensor it was never trained on, and the
# routed experts had a constant sensor id to route on, which makes them
# decoration.
#
#   multi        lrr1 + srr0, dense MLP. Isolates the effect of the data.
#   multi_moe    the same, with 4 routed experts. Isolates the effect of the
#                experts, given the data.
#
# The probe is the whole point: an encoder that cannot retain a frame's radar
# statistics cannot help a language model answer questions about them, and
# finding that out costs six minutes here against two hours of fine-tuning.
set -u
CODE=/NHNHOME/workspace/AutonomousRadarFoundationModel
PY=/NHNHOME/workspace/venv/av/bin/python
CKPT=/NHNHOME/workspace/checkpoints
cd "$CODE"

echo "[$(date '+%H:%M:%S')] waiting for the fixed-encoder pipeline"
while pgrep -f run_fixed_encoder.sh > /dev/null; do sleep 60; done

wait_for_gpus() {
    for _ in $(seq 1 90); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
               | awk '{s+=$1} END {print s+0}')
        [ "$used" -lt 2000 ] && return 0
        sleep 10
    done
    echo "WARNING: GPUs still busy, continuing anyway"
}

run() {
    local name=$1 experts=$2
    local out="$CKPT/radar_encoder_$name"
    echo "[$(date '+%H:%M:%S')] === pretrain $name (experts=$experts) ==="
    wait_for_gpus
    $PY -m torch.distributed.run --nproc_per_node=5 -m training.pretrain_radar \
        --clips all --epochs 3 --batch 8 --workers 8 \
        --radar lrr1,srr0 --readout frame --frame-queries 10 \
        --experts "$experts" --out "$out" \
        || { echo "FAILED: $name"; return 1; }

    echo "[$(date '+%H:%M:%S')] === probe $name ==="
    wait_for_gpus
    # Probed on LRR clips so the number is comparable with the LRR-only encoder
    # already measured; a mixed probe set would move the target distribution as
    # well as the encoder.
    $PY -m training.probe_radar_tokens --clips 500 --workers 6 \
        --checkpoint "$out/encoder.pt" || echo "probe failed: $name"
}

run multi 0
run multi_moe 4
echo "[$(date '+%H:%M:%S')] === ENCODER ABLATION DONE ==="
echo
echo "Reference, lrr1-only encoder with the frame readout:"
echo "  lrr1_n_points 0.650   lrr1_n_moving 0.658   lrr1_max_rcs 0.761"
