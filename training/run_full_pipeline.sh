#!/bin/bash
# Re-extract features with the SRR fix, then descriptions, then radar pretraining.
#
# The first pass excluded the short-range radar from the box-illumination test,
# so every radar_config='low' clip -- 49% of frames, 29.5 million boxes -- came
# out as 0% radar coverage and 100% camera-only. That would have made half of
# task 11's complementarity descriptions false statements.
set -u
CODE=/NHNHOME/workspace/AutonomousRadarFoundationModel
COMMON=/NHNHOME/workspace/dataset/raw_Auto_datasets/preprocessed_train_test_split/common
cd "$CODE"

echo "[$(date '+%H:%M:%S')] === 1/3 scene features over every clip (SRR fixed) ==="
python3 -m datatools.scene_features --clips all --workers 14 || exit 1

echo "[$(date '+%H:%M:%S')] === 2/3 descriptions over every clip ==="
python3 -m datatools.build_scene_descriptions \
  --features "$COMMON/scene_features_all_clips.parquet" || exit 1

echo "[$(date '+%H:%M:%S')] === 3/3 radar encoder pretraining on all clips ==="
torchrun --nproc_per_node=5 -m training.pretrain_radar \
  --clips all --epochs 3 --batch 8 --workers 6 \
  --out /NHNHOME/workspace/checkpoints/radar_encoder_all || exit 1

echo "[$(date '+%H:%M:%S')] === PIPELINE DONE ==="
