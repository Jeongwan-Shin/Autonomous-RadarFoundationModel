#!/usr/bin/env python3
"""Build the shared indices every task split is derived from.

Writes to common/:
  nvidia_clips.parquet   one row per downloaded Nvidia clip: official split,
                         collection metadata, sensor availability, and the
                         archive path + member name for every artefact.
  nuscenes_*.parquet     scenes / samples / sample_data / annotations /
                         ego_pose, tagged with the official train-val-test
                         split from the nuscenes-devkit.

Nothing is copied: the indices carry paths into the read-only raw trees, so
this directory stays small.
"""

import json
import os
import sys
import time

import pandas as pd

from . import paths

ROOT = os.path.dirname(os.path.abspath(__file__))   # this package, for ns_splits.json
RAW = paths.RAW_ROOT
NV = paths.NVIDIA_ROOT
NS = paths.NUSCENES_ROOT
OUT = paths.COMMON_DIR

CAMERA = "camera_front_wide_120fov"
RADARS = ["radar_front_center_srr_0",
          "radar_front_center_mrr_2",
          "radar_front_center_imaging_lrr_1"]
RADAR_SHORT = {"radar_front_center_srr_0": "srr0",
               "radar_front_center_mrr_2": "mrr2",
               "radar_front_center_imaging_lrr_1": "lrr1"}

# Measured from the archives. Radar scan rate differs per sensor model, so it is
# recorded per radar rather than as one number. Obstacle tracks sample at 10 Hz
# but each object carries its own timestamp, so they are not frame-synchronised.
RATES = {
    "camera_hz": 30.0,
    "camera_frames": 605,
    "clip_seconds": 20.0,
    "radar_hz": {"radar_front_center_srr_0": 13.4,
                 "radar_front_center_mrr_2": 20.0,
                 "radar_front_center_imaging_lrr_1": 20.0},
    "radar_detections_per_scan_median": {"radar_front_center_srr_0": 313,
                                        "radar_front_center_mrr_2": 545,
                                        "radar_front_center_imaging_lrr_1": 989},
    "radar_max_range_m": {"radar_front_center_srr_0": 200,
                          "radar_front_center_mrr_2": 250,
                          "radar_front_center_imaging_lrr_1": 300},
    "radar_azimuth_fov_deg": {"radar_front_center_srr_0": 160,
                              "radar_front_center_mrr_2": 177,
                              "radar_front_center_imaging_lrr_1": 120},
    "radar_output_level": "detection list (no range-azimuth/Doppler tensor)",
    "obstacle_hz": 10.0,
    "obstacle_synchronised": False,
    "egomotion_hz": 10.0,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Nvidia
# --------------------------------------------------------------------------

def build_nvidia():
    log("Nvidia: reading manifest and metadata")
    manifest = json.load(open(os.path.join(NV, "manifest.json")))
    downloaded = set(manifest["chunks_complete"])
    chunk_radars = {int(k): v for k, v in manifest["chunk_radars"].items()}

    clips = pd.read_parquet(os.path.join(NV, "clip_index.parquet"))
    collection = pd.read_parquet(os.path.join(NV, "metadata/data_collection.parquet"))
    presence = pd.read_parquet(os.path.join(NV, "metadata/feature_presence.parquet"))

    df = clips[clips["chunk"].isin(downloaded)].join(collection)
    keep = ["egomotion.offline", "obstacle.offline", "lidar_top_360fov",
            "camera_intrinsics.offline", "sensor_extrinsics.offline"] + RADARS
    df = df.join(presence[keep])
    df.index.name = "clip_id"
    log(f"Nvidia: {len(df):,} clips across {df['chunk'].nunique():,} chunks")

    chunk = df["chunk"]
    tag = chunk.map(lambda c: f"chunk_{c:04d}")
    cid = df.index.to_series()

    out = pd.DataFrame(index=df.index)
    out["chunk"] = chunk
    out["split"] = df["split"]
    out["clip_is_valid"] = df["clip_is_valid"]
    for col in ["country", "month", "hour_of_day", "platform_class", "radar_config"]:
        out[col] = df[col]

    # Camera: one zip per chunk, three members per clip.
    out["camera_zip"] = "camera/" + CAMERA + "/" + CAMERA + "." + tag + ".zip"
    out["camera_video"] = cid + f".{CAMERA}.mp4"
    out["camera_timestamps"] = cid + f".{CAMERA}.timestamps.parquet"
    out["camera_blurred_boxes"] = cid + f".{CAMERA}.blurred_boxes.parquet"

    # Radar: present only on some chunks; null where the sensor is absent.
    for radar in RADARS:
        short = RADAR_SHORT[radar]
        has = chunk.map(lambda c: radar in chunk_radars.get(c, [])) & df[radar].fillna(False)
        out[f"has_{short}"] = has
        out[f"radar_{short}_zip"] = (
            "radar/" + radar + "/" + radar + "." + tag + ".zip").where(has)
        out[f"radar_{short}_member"] = (cid + f".{radar}.parquet").where(has)

    for feature, short in [("obstacle.offline", "obstacle"),
                           ("egomotion.offline", "egomotion")]:
        has = df[feature].fillna(False)
        out[f"has_{short}"] = has
        out[f"{short}_zip"] = (
            f"labels/{feature}/{feature}." + tag + ".zip").where(has)
        out[f"{short}_member"] = (cid + f".{feature}.parquet").where(has)

    # Calibration is chunk-level parquet keyed by (clip_id, sensor_name).
    out["camera_intrinsics_parquet"] = (
        "calibration/camera_intrinsics.offline/camera_intrinsics.offline."
        + tag + ".parquet")
    out["sensor_extrinsics_parquet"] = (
        "calibration/sensor_extrinsics.offline/sensor_extrinsics.offline."
        + tag + ".parquet")

    # Downloaded separately; flagged so tasks can require it later.
    out["lidar_available_upstream"] = df["lidar_top_360fov"].fillna(False)
    out["n_front_radars"] = sum(out[f"has_{RADAR_SHORT[r]}"].astype(int) for r in RADARS)

    # Human-verified OOD reasoning labels, where present.
    reasoning_path = os.path.join(NV, "reasoning/ood_reasoning.parquet")
    if os.path.exists(reasoning_path):
        reasoning = pd.read_parquet(reasoning_path)
        out["ood_cluster"] = reasoning["event_cluster"].reindex(out.index)
        out["has_ood_reasoning"] = out["ood_cluster"].notna()
        log(f"Nvidia: {out['has_ood_reasoning'].sum():,} clips carry OOD reasoning labels")
    else:
        out["ood_cluster"] = None
        out["has_ood_reasoning"] = False

    path = os.path.join(OUT, "nvidia_clips.parquet")
    out.to_parquet(path)
    log(f"Nvidia: wrote {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
    return out


# --------------------------------------------------------------------------
# nuScenes
# --------------------------------------------------------------------------

def load_json(split_dir, name):
    path = os.path.join(NS, split_dir, f"v1.0-{split_dir}", name)
    log(f"  reading {split_dir}/{name} ({os.path.getsize(path)/1e6:.0f} MB)")
    with open(path) as fh:
        return json.load(fh)


def build_nuscenes(splits):
    scene_split = {}
    for name in ("train", "val", "test"):
        for scene in splits[name]:
            scene_split[scene] = name

    scenes, samples, sample_data, annotations, ego_poses = [], [], [], [], []

    for split_dir in ("trainval", "test"):
        log(f"nuScenes: parsing {split_dir}")
        scene_rows = load_json(split_dir, "scene.json")
        log_rows = {r["token"]: r for r in load_json(split_dir, "log.json")}
        token_to_scene = {}
        for s in scene_rows:
            lg = log_rows.get(s["log_token"], {})
            token_to_scene[s["token"]] = s["name"]
            scenes.append({
                "scene_token": s["token"], "scene_name": s["name"],
                "split": scene_split.get(s["name"], "unknown"),
                "source_dir": split_dir, "nbr_samples": s["nbr_samples"],
                "description": s["description"],
                "location": lg.get("location"), "date_captured": lg.get("date_captured"),
                "vehicle": lg.get("vehicle"),
            })

        for s in load_json(split_dir, "sample.json"):
            samples.append({
                "sample_token": s["token"], "scene_token": s["scene_token"],
                "timestamp": s["timestamp"], "prev": s["prev"], "next": s["next"],
            })

        sensors = {r["token"]: r for r in load_json(split_dir, "calibrated_sensor.json")}
        sensor_meta = {r["token"]: r for r in load_json(split_dir, "sensor.json")}
        for sd in load_json(split_dir, "sample_data.json"):
            if not sd["is_key_frame"]:
                continue  # keyframes only; sweeps stay addressable via filename
            cs = sensors.get(sd["calibrated_sensor_token"], {})
            sensor = sensor_meta.get(cs.get("sensor_token"), {})
            sample_data.append({
                "sample_token": sd["sample_token"], "channel": sensor.get("channel"),
                "modality": sensor.get("modality"),
                "filename": os.path.join(split_dir, sd["filename"]),
                "ego_pose_token": sd["ego_pose_token"],
                "calibrated_sensor_token": sd["calibrated_sensor_token"],
                "width": sd.get("width"), "height": sd.get("height"),
                "source_dir": split_dir,
            })

        for a in load_json(split_dir, "sample_annotation.json"):
            annotations.append({
                "sample_token": a["sample_token"], "instance_token": a["instance_token"],
                "translation": a["translation"], "size": a["size"],
                "rotation": a["rotation"],
                "num_lidar_pts": a["num_lidar_pts"], "num_radar_pts": a["num_radar_pts"],
                "visibility_token": a.get("visibility_token"),
                "attribute_tokens": a.get("attribute_tokens"),
                "source_dir": split_dir,
            })

        for p in load_json(split_dir, "ego_pose.json"):
            ego_poses.append({
                "ego_pose_token": p["token"], "timestamp": p["timestamp"],
                "translation": p["translation"], "rotation": p["rotation"],
                "source_dir": split_dir,
            })

        # Instance -> category, resolved per split_dir then merged below.
        cats = {c["token"]: c["name"] for c in load_json(split_dir, "category.json")}
        inst = {i["token"]: cats.get(i["category_token"])
                for i in load_json(split_dir, "instance.json")}
        for row in annotations:
            if row["source_dir"] == split_dir and "category" not in row:
                row["category"] = inst.get(row["instance_token"])

    scenes_df = pd.DataFrame(scenes)
    samples_df = pd.DataFrame(samples).merge(
        scenes_df[["scene_token", "scene_name", "split", "source_dir"]],
        on="scene_token", how="left")
    sd_df = pd.DataFrame(sample_data).merge(
        samples_df[["sample_token", "scene_name", "split"]],
        on="sample_token", how="left")
    ann_df = pd.DataFrame(annotations).merge(
        samples_df[["sample_token", "scene_name", "split"]],
        on="sample_token", how="left")
    ego_df = pd.DataFrame(ego_poses)

    for name, frame in [("nuscenes_scenes", scenes_df),
                        ("nuscenes_samples", samples_df),
                        ("nuscenes_sample_data", sd_df),
                        ("nuscenes_annotations", ann_df),
                        ("nuscenes_ego_pose", ego_df)]:
        path = os.path.join(OUT, f"{name}.parquet")
        frame.to_parquet(path, index=False)
        log(f"nuScenes: wrote {name}.parquet  rows={len(frame):,}  "
            f"({os.path.getsize(path)/1e6:.1f} MB)")

    return scenes_df, samples_df, sd_df, ann_df, ego_df


def main():
    os.makedirs(OUT, exist_ok=True)
    splits = json.load(open(os.path.join(ROOT, "ns_splits.json")))
    # The devkit defines train as the union of the detect and track halves.
    splits["train"] = splits["train_detect"] + splits["train_track"]

    nv = build_nvidia()
    ns = build_nuscenes(splits)

    meta = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_root": RAW,
        "nvidia_root": NV,
        "nuscenes_root": NS,
        "nvidia_camera": CAMERA,
        "nvidia_radars": RADARS,
        "nvidia_clips": int(len(nv)),
        "nuscenes_scenes": int(len(ns[0])),
        "nuscenes_samples": int(len(ns[1])),
        "nuscenes_annotations": int(len(ns[3])),
        "sampling_rates": RATES,
        "nuscenes_split_sizes": {k: len(v) for k, v in splits.items()},
    }
    json.dump(meta, open(os.path.join(OUT, "dataset_meta.json"), "w"), indent=2)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
