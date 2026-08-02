"""Where the data lives. The only module that knows about absolute paths.

Everything else takes roots as arguments, so the rest of the package stays
portable across machines.
"""

import os

# Raw, read-only source trees.
RAW_ROOT = os.environ.get(
    "RAVILA_RAW_ROOT", "/NHNHOME/workspace/dataset/raw_Auto_datasets")
NVIDIA_ROOT = os.path.join(RAW_ROOT, "Nvidia_AUTO")
NUSCENES_ROOT = os.path.join(RAW_ROOT, "nuScenes")

# Generated indices and per-task splits.
SPLIT_ROOT = os.environ.get(
    "RAVILA_SPLIT_ROOT", os.path.join(RAW_ROOT, "preprocessed_train_test_split"))
COMMON_DIR = os.path.join(SPLIT_ROOT, "common")

CAMERA = "camera_front_wide_120fov"
RADARS = ("radar_front_center_srr_0",
          "radar_front_center_mrr_2",
          "radar_front_center_imaging_lrr_1")
RADAR_SHORT = {"radar_front_center_srr_0": "srr0",
               "radar_front_center_mrr_2": "mrr2",
               "radar_front_center_imaging_lrr_1": "lrr1"}

# Front-sensor field of view. Both the camera (120 deg wide) and the imaging
# LRR span about +/-60 deg, so this is the sector where a front-only model can
# actually observe anything. Measured: labels reach +/-180 deg, and only ~46%
# of them fall inside this sector.
FRONT_FOV_DEG = 60.0
FRONT_MAX_RANGE_M = 300.0
