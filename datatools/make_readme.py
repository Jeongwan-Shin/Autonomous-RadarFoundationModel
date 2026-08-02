#!/usr/bin/env python3
"""Generate README.md for this directory and for each task folder.

Numbers are read back out of the generated parquet files rather than copied by
hand, so the documentation cannot drift from the splits it describes.
"""

import json
import os
import time

import pandas as pd
import pyarrow.parquet as pq

from . import paths

ROOT = paths.SPLIT_ROOT
COMMON = paths.COMMON_DIR

TASKS = [
    ("01_object_detection", "Object Detection"),
    ("02_tracking", "Tracking"),
    ("03_1_planning_ego", "Planning - Ego"),
    ("03_2_agent_trajectory", "Planning - Agent Trajectory Prediction"),
    ("04_world_model", "Action-conditioned World Model"),
    ("05_depth_estimation", "Depth Estimation"),
    ("06_doppler_motion_seg", "Doppler-based Moving Object Segmentation"),
    ("07_radar_adaptation", "Radar Adaptation (cross-radar transfer)"),
    ("08_missing_modality", "Missing-modality Robustness"),
    ("09_scenario_retrieval", "Scenario Mining / Retrieval"),
]
# Produced on another machine and dropped in. Their READMEs are authored by
# whoever generates the data, so they are summarised in the index but never
# overwritten here.
EXTERNAL = [
    ("10_radar_vision_qa", "Radar+Vision Question Answering"),
    ("11_radar_vision_scene_description", "Radar+Vision Scene Description"),
]


def rows_of(path):
    """Row count without loading the file."""
    return pq.ParquetFile(path).metadata.num_rows


def scan(directory):
    out = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".parquet"):
            continue
        path = os.path.join(directory, name)
        out.append((name[:-8], rows_of(path), os.path.getsize(path)))
    return out


def describe_contents(directory):
    """What is in a folder, including things that are not parquet.

    The external tasks ship JSON trees and tarballs rather than parquet, so a
    parquet-only listing would report them as empty.
    """
    lines = []
    for name in sorted(os.listdir(directory)):
        if name == "README.md":
            continue
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            inner = os.listdir(path)
            lines.append(f"| `{name}/` | {len(inner):,} files | directory |")
        elif name.endswith(".parquet"):
            lines.append(f"| `{name}` | {rows_of(path):,} rows | "
                         f"{os.path.getsize(path)/1e6:.1f} MB |")
        else:
            lines.append(f"| `{name}` | - | {os.path.getsize(path)/1e6:.1f} MB |")
    if not lines:
        return "_empty - awaiting data._"
    return "\n".join(["| item | contents | size |", "|---|---|---:|"] + lines)


def fmt_table(files):
    lines = ["| file | rows | size |", "|---|---:|---:|"]
    for name, rows, size in files:
        lines.append(f"| `{name}.parquet` | {rows:,} | {size/1e6:.1f} MB |")
    return "\n".join(lines)


def task_readme(folder, title):
    directory = os.path.join(ROOT, folder)
    if not os.path.isdir(directory):
        return None
    cfg_path = os.path.join(directory, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    files = scan(directory)

    body = [f"# {title}", ""]
    if cfg.get("task"):
        body += [f"**Task.** {cfg['task']}", ""]

    def block(key, label):
        nonlocal body
        v = cfg.get(key)
        if v is None:
            return
        body.append(f"## {label}")
        body.append("")
        if isinstance(v, dict):
            for k, val in v.items():
                if isinstance(val, list):
                    body.append(f"- **{k}**")
                    body.extend(f"  - {x}" for x in val)
                else:
                    body.append(f"- **{k}**: {val}")
        elif isinstance(v, list):
            body.extend(f"- {x}" for x in v)
        else:
            body.append(str(v))
        body.append("")

    block("input", "Input")
    block("output", "Output")
    block("metrics", "Metrics")

    for key in cfg:
        if key in ("task", "input", "output", "metrics"):
            continue
        block(key, key.replace("_", " ").capitalize())

    body += ["## Files", "", fmt_table(files), ""]
    body += ["Every row carries paths into the read-only raw trees under",
             "`../Nvidia_AUTO/` and `../nuScenes/`. No sensor data is copied here.",
             ""]

    path = os.path.join(directory, "README.md")
    with open(path, "w") as fh:
        fh.write("\n".join(body))
    return files


def main():
    meta = json.load(open(os.path.join(COMMON, "dataset_meta.json")))
    per_task = {}
    # Only the ten generated tasks get their README rewritten. The external ones
    # are summarised in the index instead, so their authored README survives.
    for folder, title in TASKS:
        files = task_readme(folder, title)
        if files is not None:
            per_task[folder] = files

    nv = pd.read_parquet(os.path.join(COMMON, "nvidia_clips.parquet"),
                         columns=["split", "radar_config", "n_front_radars",
                                  "has_obstacle", "has_egomotion",
                                  "has_ood_reasoning", "country", "hour_of_day"])
    scenes = pd.read_parquet(os.path.join(COMMON, "nuscenes_scenes.parquet"))

    lines = [
        "# preprocessed_train_test_split",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `build_common.py` ->",
        "`build_tracks.py` -> `build_task_splits.py` -> `make_readme.py`.",
        "",
        "Index-only. Each split file stores **paths plus metadata**, never pixels or",
        "point clouds, so this whole directory is a few hundred MB against ~3.3 TB of",
        "raw data. A loader joins on `clip_id` (Nvidia) or `sample_token` (nuScenes)",
        "and opens the referenced archive member directly.",
        "",
        "## Sources",
        "",
        "| | Nvidia_AUTO | nuScenes |",
        "|---|---|---|",
        f"| unit | 20.17 s clip | 20 s scene |",
        f"| downloaded | {len(nv):,} clips | {len(scenes)} scenes |",
        "| camera | `camera_front_wide_120fov` 1 view, 1920x1080 HEVC 30 fps | 6 cameras |",
        "| radar | 3 front radars (SRR / MRR / imaging LRR) | 5 radars |",
        "| lidar | not downloaded (upstream for ~97.5% of clips) | `LIDAR_TOP` |",
        "| boxes | `obstacle.offline`, **autolabels** | human-annotated |",
        "| ego | `egomotion.offline` 10 Hz | `ego_pose` |",
        "",
        "### Split provenance",
        "",
        "Nvidia splits come from `clip_index.parquet`; chunks are homogeneous in split",
        "(verified: all 1,838 chunks map to exactly one split), so nothing leaks.",
        "nuScenes splits come from the nuscenes-devkit scene lists",
        "(`train = train_detect + train_track`).",
        "",
        "| split | Nvidia clips | nuScenes scenes |",
        "|---|---:|---:|",
    ]
    ns_counts = scenes["split"].value_counts()
    nv_counts = nv["split"].value_counts()
    for split in ("train", "val", "test"):
        lines.append(f"| {split} | {nv_counts.get(split, 0):,} | {ns_counts.get(split, 0):,} |")

    lines += [
        "",
        "## Shared indices (`common/`)",
        "",
        fmt_table(scan(COMMON)),
        "",
        "## Tasks",
        "",
        "| # | task | folder | main split rows (train / val / test) |",
        "|---|---|---|---|",
    ]
    for folder, title in TASKS:
        files = dict((n, r) for n, r, _ in per_task.get(folder, []))
        tr = files.get("nvidia_train", files.get("nvidia_source_srr_train", 0))
        va = files.get("nvidia_val", files.get("nvidia_source_srr_val", 0))
        te = files.get("nvidia_test", files.get("nvidia_source_srr_test", 0))
        num = folder.split("_")[0] if not folder.startswith("03") else folder[:4].replace("_", "-")
        lines.append(f"| {num} | {title} | `{folder}/` | {tr:,} / {va:,} / {te:,} |")
    for folder, title in EXTERNAL:
        directory = os.path.join(ROOT, folder)
        items = [n for n in os.listdir(directory) if n != "README.md"] \
            if os.path.isdir(directory) else []
        state = f"{len(items)} items (external)" if items else "awaiting data"
        lines.append(f"| {folder.split('_')[0]} | {title} | `{folder}/` | {state} |")

    lines += [
        "",
        "Tasks 10 and 11 are produced on another machine. Nothing here generates or",
        "overwrites them, including their READMEs; they are only summarised below.",
        "",
    ]
    for folder, title in EXTERNAL:
        directory = os.path.join(ROOT, folder)
        if not os.path.isdir(directory):
            continue
        lines += [f"### {folder}", "", describe_contents(directory), ""]

    lines += [
        "",
        "## Facts worth knowing before training",
        "",
        "**Nvidia boxes are autolabels.** `obstacle.offline` carries",
        "`source='scene:obstacles:autolabels:v2'`. The only human-verified ground truth",
        f"in the release is the OOD reasoning subset: {int(nv['has_ood_reasoning'].sum()):,} "
        "of the downloaded clips",
        "(1,740 exist upstream; the rest sit in chunks without front radar). Tasks 03-1",
        "and 06 are the ones whose targets come from sensor measurement instead.",
        "",
        "**`obstacle.offline` is not frame-synchronised.** Each track samples at 10 Hz but",
        "timestamps are per object, so a target time has to be chosen and tracks",
        "interpolated onto it. Boxes use `reference_frame='rig'` at their own timestamp;",
        "`egomotion` instead uses a clip-local frame anchored at t=0. Do not mix the two.",
        "",
        "**No clip carries all three front radars.**",
        "",
        "| front radars per clip | clips |",
        "|---|---:|",
    ]
    for k, v in nv["n_front_radars"].value_counts().sort_index().items():
        lines.append(f"| {k} | {v:,} |")
    lines += [
        "",
        "`radar_config` is a property of the vehicle rig: `low` carries the front SRR,",
        "`med`/`high` carry MRR + imaging LRR. They never co-occur. Chunk-level",
        "co-presence of all three is a packaging artefact only - a chunk groups ~97 clips",
        "regardless of platform. So SRR->LRR translation has no paired supervision; the",
        "only true simultaneous pair is MRR<->LRR",
        f"({int((nv['n_front_radars'] == 2).sum()):,} clips).",
        "",
        f"**{int((nv['n_front_radars'] == 0).sum()):,} clips have no front radar at all.** They "
        "arrived because the download filter",
        "worked at chunk granularity. They are camera-only and are what makes task 08 a",
        "real missing-modality benchmark rather than a simulated one.",
        "",
        "**No HD map is available** in either download (nuScenes has 4 basemap PNGs, not",
        "the semantic map expansion), so map-conditioned planning and the official",
        "nuScenes prediction challenge are out of scope until that is fetched.",
        "",
        "**Licence.** The Nvidia data is for autonomous-vehicle use cases only.",
        "`camera_blurred_boxes` marks privacy-anonymised regions - it is not an object",
        "label and must not be used to train face or licence-plate detection.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "python3 build_common.py        # shared indices from raw metadata",
        "python3 build_tracks.py        # scan obstacle archives for track statistics",
        "python3 build_task_splits.py   # per-task splits (--only 07 for one task)",
        "python3 make_readme.py         # regenerate this file and per-task READMEs",
        "```",
        "",
    ]

    with open(os.path.join(ROOT, "README.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote README.md and {len(per_task)} task READMEs")


if __name__ == "__main__":
    main()
