#!/bin/bash
# Re-pretrain the radar encoder with a readout that is actually supervised, then
# retrain the 11-task model on it without LoRA.
#
# Two defects motivate this run, both found by probing rather than by reading the
# loss curve:
#
#   1. No pretraining loss reached the tokens the language model is handed.
#      `moving` and `box_class` stopped at the per-point features, `ego` at the
#      frame tokens. The temporal stack and the global pool therefore shipped at
#      their initialisation, and a linear probe recovered a frame's detection
#      count from the encoder output at R^2 0.31 against 0.97 from the raw
#      points. `stats` now supervises the output directly.
#
#   2. The global pool destroyed the frame axis. Every radar question names a
#      frame ("at frame 11, how many detections") and the 256 learned queries
#      were not tied to frames, so there was nothing to point at. The `frame`
#      readout emits 20 x (10 + 2) tokens in frame order instead, the two extra
#      being a masked sum -- which carries the count that softmax attention
#      averages away -- and a masked max, for the strongest-return question that
#      no averaging pool can answer.
#
# The stage is `full`, not `joint`: no adapters anywhere.
set -u
CODE=/NHNHOME/workspace/AutonomousRadarFoundationModel
PY=/NHNHOME/workspace/venv/av/bin/python
CKPT=/NHNHOME/workspace/checkpoints
cd "$CODE"

ENCODER=$CKPT/radar_encoder_frame
SAMPLES=${SAMPLES:-60000}
EVAL_SAMPLES=${EVAL_SAMPLES:-3200}
MODEL=${MODEL:-8B}
EXPERTS=${EXPERTS:-0}

wait_for_gpus() {
    for _ in $(seq 1 90); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
               | awk '{s+=$1} END {print s+0}')
        [ "$used" -lt 2000 ] && return 0
        sleep 10
    done
    echo "WARNING: GPUs still busy, continuing anyway"
}

echo "[$(date '+%H:%M:%S')] === 1/4 radar encoder, frame readout ==="
wait_for_gpus
$PY -m torch.distributed.run --nproc_per_node=5 -m training.pretrain_radar \
    --clips all --epochs 3 --batch 8 --workers 8 \
    --readout frame --frame-queries 10 --experts "$EXPERTS" \
    --out "$ENCODER" || { echo "FAILED: encoder"; exit 1; }

# The number that decides whether the rest of this is worth running. `emitted`
# is the block the language model receives; it was 0.31 / 0.26 / -0.29 before.
echo "[$(date '+%H:%M:%S')] === 2/4 probe the new encoder ==="
wait_for_gpus
$PY -m training.probe_radar_tokens --clips 500 --workers 6 \
    --checkpoint "$ENCODER/encoder.pt" || echo "probe failed"

for stage in align full; do
    out="$CKPT/vlm_${MODEL}_${stage}_fixed"
    lr=$([ "$stage" = align ] && echo 1e-4 || echo 2e-5)
    echo "[$(date '+%H:%M:%S')] === 3/4 train $MODEL / $stage (lr $lr) ==="
    wait_for_gpus
    # A lower rate than the LoRA runs used: every weight moves here, and 60k
    # samples is a small epoch to be moving 8.8 B parameters with.
    $PY -m torch.distributed.run --nproc_per_node=5 -m training.train_vlm \
        --model "$MODEL" --stage "$stage" --epochs 1 --workers 5 --lr "$lr" \
        --tasks all --all-profiles --radar-dropout 0.25 \
        --radar-checkpoint "$ENCODER/encoder.pt" --master-dtype bf16 \
        --samples "$SAMPLES" --out "$out" \
        || { echo "FAILED: $MODEL $stage"; continue; }

    echo "[$(date '+%H:%M:%S')] === 4/4 eval $MODEL / $stage ==="
    wait_for_gpus
    $PY -m training.eval_vlm --model "$MODEL" --checkpoint "$out" \
        --tasks all,ood_reasoning --all-profiles --samples "$EVAL_SAMPLES" \
        || echo "eval failed for $MODEL $stage"
done
echo "[$(date '+%H:%M:%S')] === FIXED ENCODER RUN DONE ==="

# shuffled minus full is the comparison that matters; zeroed can be passed by
# noticing the input is blank rather than by reading it.
for f in "$CKPT"/vlm_*_fixed/eval.json "$CKPT"/vlm_*_alltasks/eval.json; do
    [ -f "$f" ] || continue
    echo "--- $f"
    $PY - "$f" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
rows = []
for t in sorted(d.get("full", {})):
    f = d["full"][t]["loss"]
    s = d.get("shuffled", {}).get(t, {}).get("loss")
    if s is not None:
        rows.append((t, f, s - f))
for t, f, gap in rows:
    print(f"  {t:22s} full {f:7.4f}   shuffled-full {gap:+.4f}")
if rows:
    print(f"  {'MEAN':22s}              shuffled-full "
          f"{sum(g for _, _, g in rows)/len(rows):+.4f}")
EOF
done
