#!/bin/bash
# One configuration per device, five at a time.
#
# The sharded runs use all five devices for one configuration and reach 14.7
# sample/s. A single device holds an 8 B full fine-tune on its own -- 65.6 GiB of
# weights, gradients and bf16 Adam moments against 179 -- so five configurations
# can run at once instead. Total throughput is similar; what changes is that five
# answers arrive together rather than one at a time, which is what a sweep needs.
#
# Configurations are listed one per line as "name|flags". Lines are dispatched to
# whichever device frees up next, so the list can be longer than five.
#
#   SWEEP=seeds  ./run_parallel_sweep.sh    repeat the best recipe on 5 seeds
#   SWEEP=grid   ./run_parallel_sweep.sh    contrast weight x margin
set -u
CODE=/NHNHOME/workspace/AutonomousRadarFoundationModel
PY=/NHNHOME/workspace/venv/av/bin/python
CKPT=/NHNHOME/workspace/checkpoints
cd "$CODE"

SAMPLES=${SAMPLES:-12000}
EVAL_SAMPLES=${EVAL_SAMPLES:-2400}
MODEL=${MODEL:-8B}
SWEEP=${SWEEP:-seeds}
ENC=$CKPT/radar_encoder_multi/encoder.pt
ALIGN=$CKPT/vlm_8B_align_fixed
# Single device, so the global batch is made up with accumulation instead of
# data parallelism. 4 x 40 = 160, matching every run so far.
BATCH="--micro-batch 4 --accum 40"
TASKS=${TASKS:-all}
EVAL_TASKS=${EVAL_TASKS:-all,ood_reasoning}
DROPOUT=${DROPOUT:-0.25}

case "$SWEEP" in
  seeds)
    # Five seeds of the leading recipe. Nothing so far establishes a noise floor,
    # and the quantity being compared -- +0.60p against +0.57p against -0.11p of
    # digit accuracy -- is far too small to rank without one.
    CONFIGS=$(for s in 0 1 2 3 4; do
        echo "res_s${s}|--stage full --lr 2e-5 --seed $s --resume $ALIGN"
      done)
    ;;
  seeds_contrast)
    CONFIGS=$(for s in 0 1 2 3 4; do
        echo "resc_s${s}|--stage full --lr 2e-5 --radar-contrast 1.0 --radar-margin 0.15 --seed $s --resume $ALIGN"
      done)
    ;;
  probe)
    # The decisive one. A model that never reads the radar and emits the most
    # common digit per position scores 23.9% on radar_probe; the best run so far
    # scores 27.5%, so roughly 5% of the available information is being used.
    # These give radar_probe 36 times the exposure it had in the mixture, to
    # separate "not trained enough" from "cannot be learned through this
    # interface". `noresume` and `s1` are the controls the first two need.
    TASKS=radar_probe
    # No dropout here. It blanks the radar on a quarter of items and swaps their
    # target for a refusal, which is the right regulariser for a mixed run and
    # exactly the wrong thing when the question is whether the radar can be read
    # at all -- it would spend a quarter of the budget teaching "cannot tell".
    DROPOUT=0
    CONFIGS="probe_only|--stage full --lr 2e-5 --seed 0 --resume $ALIGN
probe_lr5e5|--stage full --lr 5e-5 --seed 0 --resume $ALIGN
probe_noresume|--stage full --lr 2e-5 --seed 0
probe_s1|--stage full --lr 2e-5 --seed 1 --resume $ALIGN
probe_align|--stage align --lr 1e-4 --seed 0"
    ;;
  compare)
    # Everything that could plausibly break the ceiling, at once, one per device.
    # `base` is the control at this budget; the rest change exactly one thing.
    #
    #   coarse   answers banded instead of exact. Tests whether emitting a
    #            quantity digit by digit is what caps it.
    #   heavy    the full mixture with the radar tasks weighted up. This is the
    #            comparison the radar_probe-only sweep should have been: that one
    #            changed exposure and removed the other tasks at the same time,
    #            and single-task training collapsed to the prior.
    #   cot      the two tasks where a radar reading is upstream of the answer,
    #            answered as {"rationale": ..., "answer": ...}.
    #   cotheavy cot plus the reweighting.
    # Every configuration is evaluated on the same superset, so `radar_probe`
    # -- which all five train on -- is a yardstick they share. Only what each
    # trains on differs, and the per-config --tasks below overrides the default.
    TASKS=all
    EVAL_TASKS=all,radar_probe_coarse,agent_traj_cot,motion_seg_cot,ood_reasoning
    COT=all,agent_traj_cot,motion_seg_cot
    HEAVY="radar_probe:6,radar_transfer:6,desc_radar:6,desc_complementarity:6,\
motion_seg:4,depth_range:4,qa:1,det_objects:1,plan_ego:1,agent_traj:1,\
track_identity:1,world_model:1,retrieval:1,desc_objects:1,desc_ego_maneuver:1,\
desc_clip_summary:1"
    HEAVYCOT="$HEAVY,agent_traj_cot:6,motion_seg_cot:6"
    CONFIGS="cmp_base|--stage full --lr 2e-5 --seed 0 --resume $ALIGN
cmp_coarse|--stage full --lr 2e-5 --seed 0 --resume $ALIGN --tasks all,radar_probe_coarse
cmp_heavy|--stage full --lr 2e-5 --seed 0 --resume $ALIGN --mixture $HEAVY
cmp_cot|--stage full --lr 2e-5 --seed 0 --resume $ALIGN --tasks $COT
cmp_cotheavy|--stage full --lr 2e-5 --seed 0 --resume $ALIGN --tasks $COT --mixture $HEAVYCOT"
    ;;
  digit)
    # The diagnosis this tests: radar reaches the hidden state that generates the
    # first answer token at R^2 0.65, collapses to 0.005 with another clip's
    # radar so it is genuinely radar-borne, and comes out of the model's mouth at
    # a correlation of 0.14. Nothing upstream is broken. The only untouched stage
    # is the loss that turns that state into digits, and cross-entropy scores 639
    # against 638 as though it were 100.
    TASKS=radar_probe
    EVAL_TASKS=radar_probe
    DROPOUT=0
    CONFIGS="dg_0|--stage full --lr 2e-5 --seed 0 --resume $ALIGN --digit-weight 0
dg_0p3|--stage full --lr 2e-5 --seed 0 --resume $ALIGN --digit-weight 0.3
dg_1|--stage full --lr 2e-5 --seed 0 --resume $ALIGN --digit-weight 1.0
dg_3|--stage full --lr 2e-5 --seed 0 --resume $ALIGN --digit-weight 3.0
dg_1_s1|--stage full --lr 2e-5 --seed 1 --resume $ALIGN --digit-weight 1.0"
    ;;
  digit_full)
    # Two questions at once. Seeds 0/1 of each arm pin an effect whose two-seed
    # spread was 0.527 against 0.378, and running it on the whole eleven-task
    # mixture rather than radar_probe alone says whether the fix survives contact
    # with the tasks that do not need a radar.
    TASKS=all
    EVAL_TASKS=all,ood_reasoning
    CONFIGS="dgf_w1_s0|--stage full --lr 2e-5 --seed 0 --resume $ALIGN --digit-weight 1.0
dgf_w1_s1|--stage full --lr 2e-5 --seed 1 --resume $ALIGN --digit-weight 1.0
dgf_w0_s0|--stage full --lr 2e-5 --seed 0 --resume $ALIGN --digit-weight 0
dgf_w0_s1|--stage full --lr 2e-5 --seed 1 --resume $ALIGN --digit-weight 0
dgf_probe_s2|--stage full --lr 2e-5 --seed 2 --resume $ALIGN --digit-weight 1.0 --tasks radar_probe"
    ;;
  combined)
    # Exposure alone failed and the distance-aware loss alone failed, for
    # complementary reasons: reweighting gave the model more radar questions but
    # a loss that still scored 639 against 638 as though it were 100, and the
    # loss worked only where the task got twelve times the exposure it has in the
    # mixture. This tests them together, with a control for each half.
    TASKS=all
    EVAL_TASKS=all,ood_reasoning
    HEAVY="radar_probe:8,radar_transfer:6,desc_radar:6,desc_complementarity:6,\
motion_seg:4,depth_range:4,qa:1,det_objects:1,plan_ego:1,agent_traj:1,\
track_identity:1,world_model:1,retrieval:1,desc_objects:1,desc_ego_maneuver:1,\
desc_clip_summary:1"
    CONFIGS="cb_both_s0|--stage full --lr 2e-5 --seed 0 --resume $ALIGN --digit-weight 1.0 --mixture $HEAVY
cb_both_s1|--stage full --lr 2e-5 --seed 1 --resume $ALIGN --digit-weight 1.0 --mixture $HEAVY
cb_heavy_only|--stage full --lr 2e-5 --seed 0 --resume $ALIGN --digit-weight 0 --mixture $HEAVY
cb_digit_24k|--stage full --lr 2e-5 --seed 0 --resume $ALIGN --digit-weight 1.0 --samples 24000
cb_probe_s3|--stage full --lr 2e-5 --seed 3 --resume $ALIGN --digit-weight 1.0 --tasks radar_probe"
    ;;
  control)
    # No new variations. The distance-aware digit loss looked like a 4.7x effect
    # on the radar-attributable correlation, but that rested on four treated
    # seeds spread from 0.220 to 0.583 against a single control at 0.070 -- and a
    # reweighted run with no digit loss at all then scored 0.167. Four more
    # controls, matched in every other respect, are what decide whether the
    # effect is real.
    TASKS=radar_probe
    EVAL_TASKS=radar_probe
    DROPOUT=0
    CONFIGS=$(for s in 1 2 3 4; do
        echo "ct_w0_s${s}|--stage full --lr 2e-5 --seed $s --resume $ALIGN --digit-weight 0"
      done
      echo "ct_w1_s4|--stage full --lr 2e-5 --seed 4 --resume $ALIGN --digit-weight 1.0")
    ;;
  grid)
    CONFIGS="w0.5m0.15|--lr 2e-5 --radar-contrast 0.5 --radar-margin 0.15 --seed 0
w2.0m0.15|--stage full --lr 2e-5 --radar-contrast 2.0 --radar-margin 0.15 --seed 0
w1.0m0.30|--stage full --lr 2e-5 --radar-contrast 1.0 --radar-margin 0.30 --seed 0
w1.0m0.50|--stage full --lr 2e-5 --radar-contrast 1.0 --radar-margin 0.50 --seed 0
w1.0m0.15lr1e5|--stage full --lr 1e-5 --radar-contrast 1.0 --radar-margin 0.15 --seed 0"
    ;;
  *) echo "unknown SWEEP=$SWEEP"; exit 1 ;;
esac

# Built after the case block: a sweep may override DROPOUT, and bash expands an
# assignment at the point it is written, not where it is used.
BASE="--epochs 1 --workers 4 --all-profiles \
--radar-dropout $DROPOUT --radar-checkpoint $ENC --master-dtype bf16"

run_one() {
    local device=$1 name=$2; shift 2
    local out="$CKPT/vlm_${MODEL}_sw_${name}"
    local logfile="sweep_${name}.log"
    {
        echo "[$(date '+%H:%M:%S')] device $device: train $name : $*"
        CUDA_VISIBLE_DEVICES=$device $PY -m training.train_vlm \
            --model "$MODEL" $BASE $BATCH --tasks "$TASKS" \
            --samples "$SAMPLES" --out "$out" "$@" \
            || { echo "FAILED train $name"; exit 1; }
        echo "[$(date '+%H:%M:%S')] device $device: eval $name"
        CUDA_VISIBLE_DEVICES=$device $PY -m training.eval_vlm \
            --model "$MODEL" --checkpoint "$out" --tasks "$EVAL_TASKS" \
            --all-profiles --samples "$EVAL_SAMPLES" --batch 4 \
            || echo "FAILED eval $name"
        echo "[$(date '+%H:%M:%S')] device $device: done $name"
    } > "$logfile" 2>&1
}

# Waits on the training processes themselves, not on script names. A shell
# wrapper that merely mentions the script in its command line matches `pgrep -f`
# too, and two stale ones held this loop open indefinitely.
echo "[$(date '+%H:%M:%S')] waiting for any training process to finish"
while pgrep -f "training\.train_vlm|training\.eval_vlm|training\.pretrain_radar" \
      > /dev/null; do sleep 60; done
for _ in $(seq 1 120); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
           | awk '{s+=$1} END {print s+0}')
    [ "$used" -lt 2000 ] && break
    sleep 10
done

device=0
pids=""
while IFS='|' read -r name flags; do
    [ -z "$name" ] && continue
    if [ "$device" -ge 5 ]; then
        # Five in flight; wait for the whole wave before starting the next.
        for pid in $pids; do wait "$pid"; done
        device=0; pids=""
    fi
    echo "[$(date '+%H:%M:%S')] launching $name on device $device"
    # </dev/null is not optional. Without it the backgrounded job inherits this
    # loop's stdin, the training process reads from the here-string, and the
    # remaining configurations are eaten -- which surfaced as launches named
    # after fragments of the previous line's flags.
    run_one "$device" "$name" $flags < /dev/null &
    pids="$pids $!"
    device=$((device + 1))
    sleep 20          # stagger the model loads so five do not read at once
done <<< "$CONFIGS"
for pid in $pids; do wait "$pid"; done

echo "[$(date '+%H:%M:%S')] === SWEEP $SWEEP DONE ==="
$PY -m training.compare_runs
