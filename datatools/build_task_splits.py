#!/usr/bin/env python3
"""Derive per-task train/val/test splits from the shared indices in common/.

Every split file holds paths and metadata only -- no pixel or point data is
copied. Splits inherit the official partitioning of each source dataset
(Nvidia clip_index, nuscenes-devkit scene lists), so nothing leaks between
train and eval.

    python build_task_splits.py            # all tasks
    python build_task_splits.py --only 07  # one task
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from . import paths

ROOT = paths.SPLIT_ROOT        # where task folders are written
COMMON = paths.COMMON_DIR

NV_SPLITS = ("train", "val", "test")

# Sliding-window geometry for the trajectory tasks, in seconds.
EGO_HISTORY_S = 2.0
EGO_FUTURE_S = 3.0
EGO_STRIDE_S = 2.0
AGENT_OBS_S = 3.0
AGENT_PRED_S = 3.0

# Detection-relevant thresholds.
TRACK_MIN_OBS_TRACKING = 30       # 3 s at 10 Hz
TRACK_MIN_OBS_AGENT = 60          # 3 s observation + 3 s prediction
# Kept only so this script still runs against the legacy rig-frame track table.
# It measures displacement relative to the ego vehicle, which is the wrong frame
# for "did this object move" -- `fix_track_splits.py` overwrites tasks 02 and
# 03_2 using world-frame motion, and that is the filter actually in force.
AGENT_MIN_PATH_M = 2.0

CAMERA = "camera_front_wide_120fov"
RADAR_COLS = ["srr0", "mrr2", "lrr1"]

# nuScenes detection benchmark collapses the 23 raw categories into 10.
NUSC_DETECTION_MAP = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "trailer",
    "vehicle.construction": "construction_vehicle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.barrier": "barrier",
}

# Rough correspondence between the two label vocabularies, for cross-dataset
# transfer. Deliberately conservative: anything ambiguous is left out.
CROSS_DATASET_MAP = {
    "automobile": "car",
    "person": "pedestrian",
    "heavy_truck": "truck",
    "bus": "bus",
    "trailer": "trailer",
    "rider": "motorcycle",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def task_dir(name):
    path = os.path.join(ROOT, name)
    os.makedirs(path, exist_ok=True)
    return path


def write(frame, directory, name):
    path = os.path.join(directory, f"{name}.parquet")
    frame.to_parquet(path, index=False)
    log(f"    {name:34s} rows={len(frame):>9,}  {os.path.getsize(path)/1e6:>7.1f} MB")
    return len(frame)


def write_config(directory, config):
    with open(os.path.join(directory, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2, default=str)


def nvidia_columns(extra=()):
    base = ["clip_id", "chunk", "split", "country", "month", "hour_of_day",
            "platform_class", "radar_config", "n_front_radars",
            "camera_zip", "camera_video", "camera_timestamps",
            "camera_intrinsics_parquet", "sensor_extrinsics_parquet",
            # radar poses live only in the non-offline calibration; without
            # these a radar return cannot be lifted into the rig frame
            "radar_extrinsics_parquet", "has_radar_extrinsics",
            "radar_extrinsics_sensors"]
    for short in RADAR_COLS:
        base += [f"has_{short}", f"radar_{short}_zip", f"radar_{short}_member"]
    return base + list(extra)


def load_common():
    nv = pd.read_parquet(os.path.join(COMMON, "nvidia_clips.parquet")).reset_index()
    ns = {}
    for name in ("scenes", "samples", "sample_data", "annotations"):
        ns[name] = pd.read_parquet(os.path.join(COMMON, f"nuscenes_{name}.parquet"))
    tracks_path = os.path.join(COMMON, "nvidia_tracks.parquet")
    tracks = pd.read_parquet(tracks_path) if os.path.exists(tracks_path) else None
    return nv, ns, tracks


def nuscenes_sample_table(ns):
    """One row per keyframe with a column per sensor channel."""
    sd = ns["sample_data"]
    wide = sd.pivot_table(index="sample_token", columns="channel",
                          values="filename", aggfunc="first")
    wide.columns = [f"path_{c}" for c in wide.columns]
    counts = ns["annotations"].groupby("sample_token").size().rename("n_annotations")
    out = (ns["samples"]
           .merge(wide, on="sample_token", how="left")
           .merge(counts, on="sample_token", how="left"))
    out["n_annotations"] = out["n_annotations"].fillna(0).astype(int)
    return out


# --------------------------------------------------------------------------
# 01 Object detection
# --------------------------------------------------------------------------

def task_01(nv, ns, tracks):
    d = task_dir("01_object_detection")
    log("01 object detection")

    sel = nv[nv["has_obstacle"]].copy()
    cols = nvidia_columns(["obstacle_zip", "obstacle_member", "has_obstacle",
                           "has_egomotion", "egomotion_zip", "egomotion_member",
                           "ood_cluster", "has_ood_reasoning"])
    for split in NV_SPLITS:
        write(sel[sel["split"] == split][cols], d, f"nvidia_{split}")

    samples = nuscenes_sample_table(ns)
    ann = ns["annotations"].copy()
    ann["detection_class"] = ann["category"].map(NUSC_DETECTION_MAP)
    for split in ("train", "val"):
        write(samples[samples["split"] == split], d, f"nuscenes_{split}")
    write(samples[samples["split"] == "test"], d, "nuscenes_test_nolabels")
    write(ann[ann["detection_class"].notna()],
          d, "nuscenes_annotations_detection10")

    write_config(d, {
        "task": "3D object detection",
        "input": {
            "nvidia": ["camera_front_wide_120fov video frame",
                       "front radar detections (spherical -> rig conversion required)",
                       "camera_intrinsics.offline", "sensor_extrinsics.offline"],
            "nuscenes": ["6 cameras", "LIDAR_TOP", "5 radars"],
        },
        "output": {
            "nvidia": "3D boxes: center_xyz, size_xyz, orientation quaternion, "
                      "label_class (10 classes), frame='rig' at each timestamp",
            "nuscenes": "3D boxes: translation, size, rotation, class, velocity, attribute",
        },
        "nvidia_target_sampling_hz": 2.0,
        "nvidia_note": "obstacle.offline timestamps are per-object and asynchronous; "
                       "interpolate tracks onto the chosen target times.",
        "nuscenes_detection_classes": sorted(set(NUSC_DETECTION_MAP.values())),
        "nuscenes_class_map": NUSC_DETECTION_MAP,
        "cross_dataset_class_map": CROSS_DATASET_MAP,
        "metrics": ["mAP @ center-distance 0.5/1/2/4 m", "NDS",
                    "ATE", "ASE", "AOE (AVE/AAE need velocity+attribute, "
                    "available for nuScenes only)"],
        "labels_are_autolabels": {"nvidia": True, "nuscenes": False},
    })


# --------------------------------------------------------------------------
# 02 Tracking
# --------------------------------------------------------------------------

def task_02(nv, ns, tracks):
    d = task_dir("02_tracking")
    log("02 tracking")

    cols = nvidia_columns(["obstacle_zip", "obstacle_member",
                           "has_egomotion", "egomotion_zip", "egomotion_member"])
    if tracks is None:
        log("    !! nvidia_tracks.parquet missing - run build_tracks.py first")
    else:
        long_tracks = tracks[tracks["n_obs"] >= TRACK_MIN_OBS_TRACKING]
        per_clip = (long_tracks.groupby("clip_id")
                    .agg(n_long_tracks=("track_id", "size"),
                         max_track_s=("duration_s", "max"),
                         median_track_s=("duration_s", "median"))
                    .reset_index())
        sel = nv[nv["has_obstacle"]].merge(per_clip, on="clip_id", how="inner")
        sel = sel[sel["n_long_tracks"] >= 3]
        for split in NV_SPLITS:
            write(sel[sel["split"] == split][cols + ["n_long_tracks", "max_track_s",
                                                     "median_track_s"]],
                  d, f"nvidia_{split}")
        write(long_tracks, d, "nvidia_long_tracks")

    scenes = ns["scenes"]
    inst = (ns["annotations"].groupby(["scene_name", "instance_token"])
            .size().rename("n_obs").reset_index())
    per_scene = (inst.groupby("scene_name")
                 .agg(n_instances=("instance_token", "size"),
                      median_instance_obs=("n_obs", "median")).reset_index())
    sc = scenes.merge(per_scene, on="scene_name", how="left")
    for split in ("train", "val"):
        write(sc[sc["split"] == split], d, f"nuscenes_{split}")
    write(sc[sc["split"] == "test"], d, "nuscenes_test_nolabels")

    write_config(d, {
        "task": "3D multi-object tracking",
        "input": "sensor sequence (same modalities as detection)",
        "output": "per-frame 3D boxes with a persistent identity "
                  "(nvidia: track_id, nuscenes: instance_token)",
        "nvidia_filter": {
            "min_obs_per_track": TRACK_MIN_OBS_TRACKING,
            "min_long_tracks_per_clip": 3,
            "reason": "median track is only ~2.4 s; short tracks cannot exercise re-ID",
        },
        "metrics": ["AMOTA", "AMOTP", "MOTA", "MOTP", "IDS", "FRAG"],
        "labels_are_autolabels": {"nvidia": True, "nuscenes": False},
    })


# --------------------------------------------------------------------------
# 03-1 Ego planning
# --------------------------------------------------------------------------

def enumerate_ego_windows(clips):
    """Sliding (history, future) windows inside each 20 s clip."""
    starts = np.arange(EGO_HISTORY_S,
                       20.0 - EGO_FUTURE_S + 1e-9,
                       EGO_STRIDE_S)
    rows = []
    for t0 in starts:
        w = clips.copy()
        w["t_ref_s"] = float(t0)
        w["hist_start_s"] = float(t0 - EGO_HISTORY_S)
        w["fut_end_s"] = float(t0 + EGO_FUTURE_S)
        rows.append(w)
    return pd.concat(rows, ignore_index=True).sort_values(["clip_id", "t_ref_s"])


def task_03_1(nv, ns, tracks):
    d = task_dir("03_1_planning_ego")
    log("03-1 ego planning")

    cols = nvidia_columns(["egomotion_zip", "egomotion_member",
                           "obstacle_zip", "obstacle_member", "has_obstacle",
                           "ood_cluster", "has_ood_reasoning"])
    sel = nv[nv["has_egomotion"]][cols]
    for split in NV_SPLITS:
        write(enumerate_ego_windows(sel[sel["split"] == split]), d, f"nvidia_{split}")

    # nuScenes: keyframes at 2 Hz; a 3 s horizon needs 6 following samples.
    samples = nuscenes_sample_table(ns).sort_values(["scene_name", "timestamp"])
    samples["idx_in_scene"] = samples.groupby("scene_name").cumcount()
    samples["scene_len"] = samples.groupby("scene_name")["idx_in_scene"].transform("max") + 1
    need = int(EGO_FUTURE_S * 2)
    hist = int(EGO_HISTORY_S * 2)
    ok = samples[(samples["idx_in_scene"] >= hist) &
                 (samples["idx_in_scene"] <= samples["scene_len"] - 1 - need)]
    for split in ("train", "val"):
        write(ok[ok["split"] == split], d, f"nuscenes_{split}")

    write_config(d, {
        "task": "open-loop ego trajectory planning",
        "input": "past frames (+ radar) over the history window",
        "output": "future ego waypoints in the clip-local frame "
                  "(nvidia egomotion is anchored at t=0 with zero yaw)",
        "window_seconds": {"history": EGO_HISTORY_S, "future": EGO_FUTURE_S,
                           "stride": EGO_STRIDE_S},
        "nvidia_egomotion_hz": 10.0,
        "nuscenes_keyframe_hz": 2.0,
        "metrics": ["L2 displacement @ 1s/2s/3s", "collision rate against obstacle boxes"],
        "why_this_first": "egomotion is a sensor-derived measurement, not an autolabel, "
                          "so it is the most trustworthy target in the Nvidia release.",
    })


# --------------------------------------------------------------------------
# 03-2 Agent trajectory prediction
# --------------------------------------------------------------------------

def task_03_2(nv, ns, tracks):
    d = task_dir("03_2_agent_trajectory")
    log("03-2 agent trajectory prediction")

    if tracks is None:
        log("    !! nvidia_tracks.parquet missing - run build_tracks.py first")
    else:
        agents = tracks[(tracks["n_obs"] >= TRACK_MIN_OBS_AGENT) &
                        (tracks["path_len_m"] >= AGENT_MIN_PATH_M)].copy()
        keys = nv[["clip_id", "split", "chunk", "country", "hour_of_day",
                   "radar_config", "obstacle_zip", "obstacle_member",
                   "egomotion_zip", "egomotion_member",
                   "camera_zip", "camera_video"]]
        agents = agents.drop(columns=["chunk"]).merge(keys, on="clip_id", how="inner")
        for split in NV_SPLITS:
            write(agents[agents["split"] == split], d, f"nvidia_{split}")

    ann = ns["annotations"]
    inst = (ann.groupby(["instance_token", "scene_name", "split", "category"])
            .size().rename("n_obs").reset_index())
    inst = inst[inst["n_obs"] >= int((AGENT_OBS_S + AGENT_PRED_S) * 2)]
    for split in ("train", "val"):
        write(inst[inst["split"] == split], d, f"nuscenes_{split}")

    write_config(d, {
        "task": "surrounding-agent trajectory prediction",
        "input": "past track of the agent (+ ego trajectory, + scene context)",
        "output": "future agent trajectory, optionally K modes with probabilities",
        "window_seconds": {"observation": AGENT_OBS_S, "prediction": AGENT_PRED_S},
        "nvidia_filter": {
            "min_obs_per_track": TRACK_MIN_OBS_AGENT,
            "min_path_length_m": AGENT_MIN_PATH_M,
            "reason": "6 s of observation is required for a 3 s + 3 s split, and "
                      "parked objects would otherwise dominate the set.",
        },
        "metrics": ["minADE_k", "minFDE_k", "MissRate_k"],
        "limitation": "no HD map is available in either download, so map-conditioned "
                      "multimodal prediction is out of scope for now.",
    })


# --------------------------------------------------------------------------
# 04 Action-conditioned world model
# --------------------------------------------------------------------------

def task_04(nv, ns, tracks):
    d = task_dir("04_world_model")
    log("04 action-conditioned world model")

    cols = nvidia_columns(["egomotion_zip", "egomotion_member",
                           "camera_blurred_boxes", "ood_cluster", "has_ood_reasoning"])
    sel = nv[nv["has_egomotion"]].copy()
    sel["is_night"] = (sel["hour_of_day"] >= 20) | (sel["hour_of_day"] <= 5)
    sel["region"] = np.where(sel["country"] == "United States", "north_america", "other")
    keep = cols + ["is_night", "region"]
    for split in NV_SPLITS:
        write(sel[sel["split"] == split][keep], d, f"nvidia_{split}")
    for name, mask in [("nvidia_eval_night", sel["is_night"]),
                       ("nvidia_eval_day", ~sel["is_night"])]:
        write(sel[mask & (sel["split"] == "val")][keep], d, name)

    write_config(d, {
        "task": "action-conditioned future video prediction (driving world model)",
        "input": "K past frames + action sequence (future ego trajectory / velocity)",
        "output": "future frames, or future latent states",
        "video": {"codec": "hevc", "resolution": [1920, 1080], "fps": 30,
                  "frames_per_clip": 605, "seconds": 20.17},
        "action_source": "labels/egomotion.offline at 10 Hz",
        "metrics": ["FVD", "LPIPS",
                    "action fidelity: re-estimated trajectory vs conditioning action"],
        "eval_slices": ["day", "night", "north_america", "other"],
        "note": "camera_blurred_boxes marks privacy-anonymised regions. It is not an "
                "object label and must not be used as a face/plate detection target.",
    })


# --------------------------------------------------------------------------
# 05 Depth estimation
# --------------------------------------------------------------------------

def task_05(nv, ns, tracks):
    d = task_dir("05_depth_estimation")
    log("05 depth estimation")

    cols = nvidia_columns(["egomotion_zip", "egomotion_member"])
    sel = nv[nv["has_egomotion"]].copy()
    for split in NV_SPLITS:
        write(sel[sel["split"] == split][cols], d, f"nvidia_{split}")
    # Radar gives sparse but metric range, so it is the only in-dataset check.
    with_radar = sel[sel["n_front_radars"] > 0]
    write(with_radar[with_radar["split"] == "val"][cols], d, "nvidia_val_radar_verifiable")

    # nuScenes carries LiDAR, which is a far better depth reference.
    samples = nuscenes_sample_table(ns)
    depth_cols = [c for c in samples.columns
                  if c in ("sample_token", "scene_name", "split", "timestamp",
                           "path_CAM_FRONT", "path_LIDAR_TOP", "path_RADAR_FRONT")]
    for split in ("train", "val"):
        write(samples[samples["split"] == split][depth_cols], d, f"nuscenes_{split}")

    write_config(d, {
        "task": "monocular depth estimation",
        "input": "single RGB frame (self-supervised training uses neighbouring frames)",
        "output": "per-pixel metric depth [m]",
        "why_nvidia_helps": "egomotion supplies metric-scale pose, so pose does not have "
                            "to be learned jointly and the usual scale ambiguity of "
                            "self-supervised monodepth disappears.",
        "supervision": {
            "nvidia": "self-supervised photometric loss with known pose; "
                      "sparse metric check from radar 'distance'",
            "nuscenes": "LIDAR_TOP projected into CAM_FRONT gives dense-ish depth GT",
        },
        "metrics": ["Abs Rel", "Sq Rel", "RMSE", "delta < 1.25"],
        "note": "Nvidia LiDAR was not downloaded; it exists upstream for ~97.5% of clips "
                "(column lidar_available_upstream in common/nvidia_clips.parquet).",
    })


# --------------------------------------------------------------------------
# 06 Doppler motion segmentation
# --------------------------------------------------------------------------

def task_06(nv, ns, tracks):
    d = task_dir("06_doppler_motion_seg")
    log("06 doppler motion segmentation")

    cols = nvidia_columns(["egomotion_zip", "egomotion_member",
                           "obstacle_zip", "obstacle_member", "has_obstacle"])
    sel = nv[nv["has_egomotion"] & (nv["n_front_radars"] > 0)]
    for split in NV_SPLITS:
        write(sel[sel["split"] == split][cols], d, f"nvidia_{split}")

    samples = nuscenes_sample_table(ns)
    radar_cols = [c for c in samples.columns if c.startswith("path_RADAR")]
    keep = ["sample_token", "scene_name", "split", "timestamp",
            "path_CAM_FRONT", "path_LIDAR_TOP"] + radar_cols
    for split in ("train", "val"):
        write(samples[samples["split"] == split][keep], d, f"nuscenes_{split}")

    write_config(d, {
        "task": "moving vs static segmentation from radar Doppler",
        "stage_1": "compensate radial_velocity with ego motion; residual |v| above a "
                   "threshold marks a moving detection -- no manual labels needed",
        "stage_2": "distil the radar-derived motion signal into a camera model to get "
                   "an image-space moving-object mask",
        "input": "radar scans (+ egomotion) and optionally the RGB frame",
        "output": "per-detection moving/static label, or per-pixel motion mask",
        "radial_velocity_range_m_s": [-23.01, 25.26],
        "why_valuable": "the only Nvidia target here comes from sensor measurement, so it "
                        "sidesteps the autolabel noise that limits tasks 01 and 02.",
        "nuscenes_note": "nuScenes radar already provides ego-compensated vx_comp/vy_comp "
                         "plus a dyn_prop field, useful as a sanity reference.",
        "metrics": ["IoU (moving class)", "precision/recall on detections"],
    })


# --------------------------------------------------------------------------
# 07 Radar adaptation
# --------------------------------------------------------------------------

def task_07(nv, ns, tracks):
    d = task_dir("07_radar_adaptation")
    log("07 radar adaptation")

    cols = nvidia_columns(["obstacle_zip", "obstacle_member",
                           "egomotion_zip", "egomotion_member"])
    # radar_config is a per-clip property of the vehicle: 'low' rigs carry the
    # front SRR, 'med'/'high' rigs carry MRR + imaging LRR. The two never
    # co-occur on one clip, so source->target has no paired supervision.
    source = nv[nv["radar_config"] == "low"]
    target = nv[nv["radar_config"].isin(["med", "high"])]
    paired = nv[nv["has_mrr2"] & nv["has_lrr1"]]

    for name, frame in [("source_srr", source), ("target_mrr_lrr", target)]:
        for split in NV_SPLITS:
            write(frame[frame["split"] == split][cols], d, f"nvidia_{name}_{split}")
    # The only genuine simultaneous pair in the release: mid vs long range.
    for split in NV_SPLITS:
        write(paired[paired["split"] == split][cols], d, f"nvidia_paired_mrr_lrr_{split}")

    write_config(d, {
        "task": "cross-radar transfer and cross-range translation",
        "domains": {
            "source": "radar_config='low' -> front SRR only",
            "target": "radar_config='med'/'high' -> front MRR + imaging LRR",
            "paired_mrr_lrr": "clips carrying MRR and imaging LRR at the same time",
        },
        "counts": {"source": int(len(source)), "target": int(len(target)),
                   "paired_mrr_lrr": int(len(paired))},
        "clips_with_all_three_front_radars": 0,
        "important": "SRR never co-occurs with MRR/LRR on the same clip, because "
                     "radar_config is a property of the vehicle rig. Chunk-level "
                     "co-presence of all three sensors is only a packaging artefact - a "
                     "chunk groups ~97 clips regardless of platform. Therefore SRR -> LRR "
                     "translation cannot be supervised with paired data; it has to be "
                     "posed as unpaired translation or unsupervised domain adaptation.",
        "why_this_split_is_real": "the domains correspond to two physical vehicle "
                                  "generations (hyperion_8 vs 8.1 sensor suites), not to "
                                  "an artificial perturbation.",
        "use_cases": [
            "unpaired / unsupervised domain adaptation of a detector from SRR to MRR+LRR",
            "supervised MRR <-> LRR translation on nvidia_paired_mrr_lrr_* (true "
            "simultaneous observations of the same scene)",
            "sensor-invariant radar representation learning across both domains",
        ],
        "metrics": ["target-domain detection mAP", "Chamfer distance (paired MRR<->LRR)",
                    "downstream detector trained on translated radar"],
    })


# --------------------------------------------------------------------------
# 08 Missing-modality robustness
# --------------------------------------------------------------------------

def task_08(nv, ns, tracks):
    d = task_dir("08_missing_modality")
    log("08 missing-modality robustness")

    sel = nv.copy()
    sel["sensor_profile"] = (
        np.where(sel["has_srr0"], "S", "") + np.where(sel["has_mrr2"], "M", "")
        + np.where(sel["has_lrr1"], "L", ""))
    sel["sensor_profile"] = sel["sensor_profile"].replace("", "none")
    cols = nvidia_columns(["obstacle_zip", "obstacle_member", "has_obstacle",
                           "egomotion_zip", "egomotion_member", "sensor_profile"])
    for split in NV_SPLITS:
        write(sel[sel["split"] == split][cols], d, f"nvidia_{split}")

    profile = (sel.groupby(["sensor_profile", "split"]).size()
               .rename("n_clips").reset_index())
    write(profile, d, "profile_counts")

    write_config(d, {
        "task": "robustness to genuinely absent sensors",
        "input": "whatever subset of sensors the clip actually carries",
        "output": "the downstream task output, plus a degradation curve per profile",
        "sensor_profile_legend": {"S": "front SRR", "M": "front MRR",
                                 "L": "front imaging LRR", "none": "camera only"},
        "profiles_present": profile.groupby("sensor_profile")["n_clips"].sum().to_dict(),
        "why_this_beats_masking": "most papers simulate dropout by masking inputs. Here "
                                  "the sensors are missing in the real collection, so the "
                                  "absence pattern is the deployment distribution.",
        "metrics": ["per-profile task metric", "worst-profile metric", "degradation slope"],
    })


# --------------------------------------------------------------------------
# 09 Scenario mining / retrieval
# --------------------------------------------------------------------------

def task_09(nv, ns, tracks):
    d = task_dir("09_scenario_retrieval")
    log("09 scenario mining / retrieval")

    corpus_cols = ["clip_id", "chunk", "split", "country", "month", "hour_of_day",
                   "platform_class", "radar_config", "n_front_radars",
                   "camera_zip", "camera_video", "ood_cluster", "has_ood_reasoning"]
    corpus = nv[corpus_cols].copy()
    corpus["is_night"] = (corpus["hour_of_day"] >= 20) | (corpus["hour_of_day"] <= 5)
    write(corpus, d, "nvidia_corpus_all")
    for split in NV_SPLITS:
        write(corpus[corpus["split"] == split], d, f"nvidia_corpus_{split}")

    # OOD reasoning clips are the only human-verified relevance signal available.
    reasoning_path = os.path.join(paths.NVIDIA_ROOT,
                                  "reasoning/ood_reasoning.parquet")
    if os.path.exists(reasoning_path):
        raw = pd.read_parquet(reasoning_path)
        rows = []
        for clip_id, r in raw.dropna(subset=["events"]).iterrows():
            for e in json.loads(r["events"]):
                rows.append({"clip_id": clip_id, "split": r["split"],
                             "event_cluster": r["event_cluster"],
                             "event_start_frame": e["event_start_frame"],
                             "event_start_timestamp_us": e["event_start_timestamp"],
                             "coc_text": e["coc"]})
        events = pd.DataFrame(rows)
        events["clip_downloaded"] = events["clip_id"].isin(set(nv["clip_id"]))
        for split in ("train", "val"):
            write(events[events["split"] == split], d, f"nvidia_ood_queries_{split}")
        write(events, d, "nvidia_ood_queries_all")

    scenes = ns["scenes"][["scene_token", "scene_name", "split", "location",
                           "description", "nbr_samples"]]
    for split in ("train", "val", "test"):
        write(scenes[scenes["split"] == split], d, f"nuscenes_scene_text_{split}")

    write_config(d, {
        "task": "scenario mining and text-to-clip retrieval",
        "input": "a text query (or an example clip)",
        "output": "ranked clip list",
        "corpus_size": int(len(corpus)),
        "relevance_signals": {
            "ood_reasoning": "human-verified OOD cluster + Chain-of-Causation text; "
                             "only signal that is not machine-generated",
            "metadata": "country / month / hour_of_day / platform_class / radar_config",
            "nuscenes_scene_description": "free-text scene descriptions usable as "
                                          "weak text supervision for transfer",
        },
        "ood_cluster_note": "distribution is heavily skewed; the top 3 clusters are 86% "
                            "of labels and the bottom 4 have <30 clips each, so treat "
                            "rare clusters as few-shot or binary OOD detection.",
        "metrics": ["Recall@k", "mAP", "nDCG"],
        "official_challenge": "nvidia/AV-Causal-Scenario-Retrieval-Challenge",
    })


# --------------------------------------------------------------------------
# 10, 11 -- data produced externally
# --------------------------------------------------------------------------

# These two tasks are built on another machine and dropped in here. This script
# owns neither, so it only creates the directory and seeds a README while the
# directory is still empty. Once anything lands, it is left completely alone --
# an earlier version rewrote the README unconditionally and clobbered the real
# one after the QA data arrived.
EXTERNAL_TASKS = [
    ("10_radar_vision_qa", "Radar+Vision Question Answering",
     "Question answering over paired radar and camera input."),
    ("11_radar_vision_scene_description", "Radar+Vision Scene Description",
     "Free-text scene description generated from paired radar and camera input."),
]


def has_content(directory):
    return any(name != "README.md" for name in os.listdir(directory))


def task_10_11():
    for name, title, desc in EXTERNAL_TASKS:
        d = task_dir(name)
        if has_content(d):
            log(f"    {name}: has data, left untouched")
            continue
        with open(os.path.join(d, "README.md"), "w") as fh:
            fh.write(f"""# {title}

**Status: awaiting data, produced on another machine.**

{desc}

This directory is empty for now. `build_task_splits.py` does not generate it and
will stop touching this file as soon as any data lands here.

## Joining with the rest of the splits

`../common/nvidia_clips.parquet` carries, for every one of the 177,891
downloaded clips: the official split, collection metadata, and the archive path
plus member name for the camera video and each front radar. Join on `clip_id` so
this task stays aligned with tasks 01-09 and no clip leaks across splits.

Note that Nvidia's official split is not the same thing as a split chosen when
generating this task's data. If the two disagree, clips used for training here
can reappear in another task's validation set.
""")
        log(f"    {name}: empty, seeded README")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="task prefix, e.g. 07 (default: all)")
    args = ap.parse_args(argv)

    log("loading common indices")
    nv, ns, tracks = load_common()
    log(f"  nvidia clips {len(nv):,} | nuscenes samples {len(ns['samples']):,} "
        f"| tracks {'-' if tracks is None else format(len(tracks), ',')}")

    builders = {
        "01": task_01, "02": task_02, "03_1": task_03_1, "03_2": task_03_2,
        "04": task_04, "05": task_05, "06": task_06, "07": task_07,
        "08": task_08, "09": task_09,
    }
    # Tasks 02 and 03_2 filter on whether an object moved, which only makes
    # sense in the world frame. Once the world-frame track table exists,
    # fix_track_splits.py owns those two folders; writing them here first with
    # the legacy rig-frame table would just be overwritten, and would leave the
    # wrong filter in place if the follow-up step were skipped.
    world_tracks = os.path.join(COMMON, "nvidia_tracks_world.parquet")
    owned_elsewhere = {"02", "03_2"} if os.path.exists(world_tracks) else set()

    for key, fn in builders.items():
        if args.only and not key.startswith(args.only):
            continue
        if key in owned_elsewhere and not args.only:
            log(f"{key}: skipped, owned by fix_track_splits (world-frame motion)")
            continue
        fn(nv, ns, tracks)

    if not args.only:
        log("10, 11 (external)")
        task_10_11()

    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
