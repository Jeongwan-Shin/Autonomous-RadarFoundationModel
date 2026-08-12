#!/usr/bin/env python3
"""Dataset that turns a clip into padded radar point tensors plus free labels.

Nothing is materialised to disk. Reading a clip's radar, obstacle and egomotion
archives measures 71.8 clip/s single-threaded, so 14 workers deliver about
1,005 clip/s against the ~80 clip/s a 5-GPU step actually consumes. Writing the
29 million point rows to parquet first would have cost a gigabyte and a pipeline
stage for no throughput gain.

Per point, eight channels. The last four matter as much as the geometry:

  x, y, z            rig frame, metres
  radial_velocity    raw Doppler
  doppler_residual   after ego compensation and fold correction. This is what
                     separates movers from the static world without any label.
  rcs                the only cue that distinguishes a stopped car from ground
                     clutter, since a stopped car has zero Doppler residual
  snr                confidence
  range              sqrt(x^2 + y^2); angular resolution scales with it, so the
                     model should see it explicitly

Two supervision signals come for free, computed rather than annotated:

  is_moving      doppler_residual > 1 m/s
  box_class      which labelled box the point falls inside, if any

The second is sparse on purpose, and it is worth knowing how sparse before
relying on it: measured over 800 frames, only 3.9% of returns land inside any
box, and a car carries a median of 3.8 of them. Pedestrians average 1.8 and
protruding objects essentially never appear. Class-conditioned grounding is
therefore trainable for vehicles and not much else.
"""

import io
import os
import zipfile

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from datatools import paths
from datatools.geometry import (doppler_residual, extrinsics_to_matrix,
                               points_in_box, sensor_to_rig,
                               spherical_to_sensor_xyz)
from training.radar_encoder import SENSOR_IDS

SENSOR_NAME = {"srr0": "radar_front_center_srr_0",
               "mrr2": "radar_front_center_mrr_2",
               "lrr1": "radar_front_center_imaging_lrr_1"}

# 채널 정의는 radar_encoder 에 있다 -- 인코더가 그 이름으로 기하 채널을
# 골라야 하고, radar_data 가 이미 radar_encoder 를 참조하므로 방향은
# 하나뿐이다.
from training.radar_encoder import CHANNELS


CLASSES = ("automobile", "person", "heavy_truck", "bus", "trailer", "rider",
           "protruding_object", "animal", "stroller", "other_vehicle",
           "train_or_tram_car")
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}

# Scales chosen from the measured ranges so every channel lands near unit
# variance: range reaches 300 m, rcs spans -60..+60 dBsm, snr 11..50,
# radial velocity +/-28 m/s.
# sin/cos of the bearing are already in [-1, 1].
NORMALISE = np.array([50.0, 50.0, 100.0, 1.0, 1.0, 15.0, 5.0, 20.0],
                     dtype=np.float32)

# The encoder's bearing objective takes atan2(y, x) straight off the normalised
# tensor, which is only the same angle while these two scales are equal.
assert NORMALISE[CHANNELS.index("x")] == NORMALISE[CHANNELS.index("y")]

# Where the labelled objects are, as a coarse bird's-eye grid the encoder is
# asked to fill in. This is the pretraining target that carries position.
#
# The first attempt asked for the bearing histogram of the *returns*, which
# needs no label at all -- and turned out to teach almost nothing: measured
# over 60 clips, predicting the dataset-average histogram already costs 2.064
# against a floor of 2.019, a spread of 0.046. Most returns are road, kerb and
# clutter, so the distribution barely moves between scenes. The objects do
# move: the same measurement over their bearings gives 2.308 against 1.044, a
# spread of 1.264, twenty-seven times wider.
#
# Twelve azimuth cells of 10 degrees over the sector and four range rings of
# 10 m out to 40 m -- the same 40 m the listing tasks now use. Mean occupancy
# is 0.094, so the grid is sparse enough to be a real question.
OCC_AZ_BINS, OCC_RANGE_BINS = 12, 4
OCC_MAX_RANGE_M = 40.0

MOVING_THRESHOLD_MS = 1.0
BOX_MARGIN = 1.3
BOX_TIME_WINDOW_S = 0.12      # a 0.6 s stale box is 6 m out of place at 10 m/s


def read_member(root, zip_rel, member):
    with zipfile.ZipFile(os.path.join(root, zip_rel)) as zf:
        return pd.read_parquet(io.BytesIO(zf.read(member)))


class RadarClipDataset(Dataset):
    """One item per clip: `n_frames` radar scans at 1 Hz with per-point labels.

    Frames follow the same 1 Hz index as the QA and the descriptions, so an item
    lines up with any text built for the same clip.
    """

    def __init__(self, clip_ids, radar="lrr1", n_frames=20, max_points=1024,
                 nvidia_root=paths.NVIDIA_ROOT, common_dir=paths.COMMON_DIR,
                 with_boxes=True, thin_to=None):
        self.clip_ids = list(clip_ids)
        self.radar = radar
        self.n_frames = n_frames
        self.max_points = max_points
        self.nvidia_root = nvidia_root
        self.with_boxes = with_boxes
        # (low, high) returns per scan to thin down to, or None to leave a scan
        # as the sensor gave it. Training only -- see `_thin`.
        self.thin_to = thin_to
        index = pd.read_parquet(os.path.join(common_dir, "nvidia_clips.parquet"))
        self.index = index.loc[[c for c in self.clip_ids if c in index.index]]
        self.clip_ids = list(self.index.index)

    def __len__(self):
        return len(self.clip_ids)

    def _thin(self, mask, index):
        """Drop returns at random so density is a nuisance, not a cue.

        This rig's LRR gives a median 815 returns per scan; the ARS408 that
        nuScenes carries gives 125, measured over 400 scans. An encoder that
        only ever saw dense scans reads a sparse one as 85% empty, and that is
        a larger shift than the two channels the two rigs do not share.
        Thinning each scan to a random count makes the number of returns
        uninformative, so the same weights read either sensor.

        Log-uniform rather than uniform: the interesting end is the sparse one,
        and a uniform draw over 80..815 would put four fifths of its mass above
        250, where the model already works.

        Points are dropped by clearing the mask rather than by rewriting the
        arrays, so the per-point labels beside them stay aligned.
        """
        low, high = self.thin_to
        rng = np.random.default_rng(index * 9973 + 17)
        for f in range(mask.shape[0]):
            present = np.flatnonzero(mask[f])
            if len(present) <= low:
                continue
            target = int(round(np.exp(rng.uniform(
                np.log(low), np.log(min(high, len(present)))))))
            drop = rng.choice(present, len(present) - target, replace=False)
            mask[f, drop] = False

    def _ego(self, row):
        ego = read_member(self.nvidia_root, row["egomotion_zip"],
                          row["egomotion_member"])
        t = ego["timestamp"].to_numpy() / 1e6
        xy = ego[["x", "y"]].to_numpy()
        velocity = np.gradient(xy, t, axis=0)
        quat = ego[["qx", "qy", "qz", "qw"]].to_numpy()
        return t, velocity, quat, ego

    def _boxes(self, row, ego):
        if (not self.with_boxes or not row.get("has_obstacle", False)
                or not isinstance(row.get("obstacle_zip"), str)):
            return None
        obstacle = read_member(self.nvidia_root, row["obstacle_zip"],
                               row["obstacle_member"])
        return None if obstacle.empty else obstacle

    def __getitem__(self, i):
        return self.build(i, until_s=None)

    def window(self, i, until_s, rate_hz=None):
        """The last `n_frames` scans ending at `until_s`, at the sensor's own rate.

        The default sampling takes one scan per second and throws away 95% of
        what the sensor produced -- imaging LRR runs at 20 Hz, so a 20 s clip
        holds ~399 scans and only 20 survive. For a question about one instant
        that trade is backwards: the twenty scans immediately before that instant
        carry the object's actual Doppler track, and the nineteen seconds of
        other context carry nothing the question asks about.

        Twenty scans is one second on LRR and MRR (20 Hz) and about 1.6 s on SRR
        (12.7 Hz). Counting scans rather than fixing the duration keeps the
        tensor full for every sensor instead of padding SRR with blanks.
        """
        return self.build(i, until_s=float(until_s), rate_hz=rate_hz)

    def build(self, i, until_s=None, rate_hz=None):
        clip_id = self.clip_ids[i]
        row = self.index.loc[clip_id]
        short = self.radar
        # egomotion is required, not optional: the Doppler residual is computed
        # against it and the ego channels come from it. 2.5% of clips have no
        # egomotion or obstacle labels, and reaching those mid-epoch killed a
        # run at step 100 with a None path handed to os.path.join.
        if (not row.get("has_egomotion", False)
                or not isinstance(row.get("egomotion_zip"), str)):
            return self._empty(clip_id)

        # Radar is optional so a clip that genuinely carries no front radar --
        # sensor profile 'none', 17,130 clips -- still yields its ego motion and
        # video instead of being dropped whole. Task 08 is about exactly those
        # clips, and zeroing their ego alongside their radar would have blanked a
        # modality that is in fact present.
        radar = rotation = translation = None
        scans = np.array([])
        if (row.get(f"has_{short}", False)
                and row.get("has_radar_extrinsics", False)
                and isinstance(row.get("radar_extrinsics_parquet"), str)):
            extrinsics = pd.read_parquet(
                os.path.join(self.nvidia_root, row["radar_extrinsics_parquet"]))
            if clip_id in extrinsics.index.get_level_values(0):
                rotation, translation = extrinsics_to_matrix(
                    extrinsics.loc[(clip_id, SENSOR_NAME[short])])
                radar = read_member(self.nvidia_root, row[f"radar_{short}_zip"],
                                    row[f"radar_{short}_member"])
                scans = np.sort(radar["timestamp"].unique())

        ego_t, ego_v, ego_q, _ = self._ego(row)
        obstacle = self._boxes(row, None)

        F, P, C = self.n_frames, self.max_points, len(CHANNELS)
        features = np.zeros((F, P, C), dtype=np.float32)
        mask = np.zeros((F, P), dtype=bool)
        moving = np.zeros((F, P), dtype=np.int64)
        box_class = np.full((F, P), -1, dtype=np.int64)   # -1 = not in any box
        occupancy = np.zeros((F, OCC_AZ_BINS * OCC_RANGE_BINS), dtype=np.float32)
        ego_state = np.zeros((F, 3), dtype=np.float32)    # speed, accel, yaw rate

        speed = np.linalg.norm(ego_v, axis=1)
        accel = np.gradient(speed, ego_t)
        qx, qy, qz, qw = ego_q.T
        yaw = np.unwrap(np.arctan2(2 * (qw * qz + qx * qy),
                                  1 - 2 * (qy ** 2 + qz ** 2)))
        yaw_rate = np.gradient(yaw, ego_t)

        # Frame f is at t = f seconds by default; in windowed mode it is the
        # f-th of the twenty scans that end at `until_s`, so the frame axis
        # spans one second instead of twenty.
        if until_s is None or len(scans) == 0:
            times = [float(f) for f in range(F)]
        elif rate_hz:
            # A fixed grid ending at `until_s`: F samples at `rate_hz`, each
            # taking the nearest real scan. This decouples the window's duration
            # from the sensor's rate, so 20 frames at 4 Hz is five seconds on
            # every rig -- LRR at 20 Hz and SRR at 12.7 Hz alike -- which
            # counting scans backwards would not give.
            grid = until_s - np.arange(F - 1, -1, -1) / float(rate_hz)
            times = [float(scans[int(np.argmin(np.abs(scans / 1e6 - g)))]) / 1e6
                     for g in grid]
        else:
            earlier = scans[scans / 1e6 <= until_s + 1e-6]
            chosen = (earlier[-F:] if len(earlier) >= F
                      else scans[:F] if len(scans) >= F else scans)
            times = [float(s) / 1e6 for s in chosen]
            times = [times[0]] * (F - len(times)) + times

        for f in range(F):
            t_s = times[f]
            j = int(np.argmin(np.abs(ego_t - t_s)))
            ego_state[f] = (speed[j], accel[j], np.degrees(yaw_rate[j]))
            if len(scans) == 0:
                continue
            pick = scans[int(np.argmin(np.abs(scans / 1e6 - t_s)))]
            scan = radar[radar["timestamp"] == pick]
            if scan.empty:
                continue

            sensor_xyz = spherical_to_sensor_xyz(scan["azimuth"].to_numpy(),
                                                 scan["elevation"].to_numpy(),
                                                 scan["distance"].to_numpy())
            rig = sensor_to_rig(sensor_xyz, rotation, translation)
            v_rig = Rotation.from_quat(ego_q[j]).as_matrix().T @ np.array(
                [ego_v[j, 0], ego_v[j, 1], 0.0])
            residual = doppler_residual(rig, scan["radial_velocity"].to_numpy(),
                                        v_rig, scan["doppler_ambiguity"].to_numpy())

            n = min(len(rig), P)
            if len(rig) > P:
                # Keep the informative returns rather than an arbitrary prefix:
                # everything above the Doppler threshold, then the strongest
                # reflectors. Truncating blindly would drop movers, which are
                # only ~10% of the cloud.
                # lexsort takes the last key as primary. Negate to sort
                # descending; the bool has to become numeric first because numpy
                # rejects unary minus on booleans.
                is_mover = (residual > MOVING_THRESHOLD_MS).astype(np.int8)
                order = np.lexsort((-scan["rcs"].to_numpy(), -is_mover))[:P]
            else:
                order = np.arange(n)

            rng = np.hypot(rig[order, 0], rig[order, 1])
            # The bearing, handed over directly. `atan2(y, x)` is not
            # something a linear layer can form, and the one thing this model
            # keeps failing at is azimuth -- so it is given as sin and cos
            # rather than left to be inferred from x and y. Both are computed
            # identically from nuScenes, so the shared six-channel contract
            # becomes a shared eight.
            az = np.arctan2(rig[order, 1], rig[order, 0])
            stack = np.stack([rig[order, 0], rig[order, 1], rng,
                              np.sin(az), np.cos(az),
                              scan["radial_velocity"].to_numpy()[order],
                              residual[order],
                              scan["rcs"].to_numpy()[order]], axis=1)
            features[f, :n] = (stack / NORMALISE).astype(np.float32)
            mask[f, :n] = True
            moving[f, :n] = (residual[order] > MOVING_THRESHOLD_MS).astype(np.int64)

            if obstacle is not None:
                near = obstacle[(obstacle["timestamp_us"] / 1e6 - t_s).abs()
                                <= BOX_TIME_WINDOW_S].drop_duplicates("track_id")
                for box in near.itertuples():
                    inside = points_in_box(
                        rig[order],
                        np.array([box.center_x, box.center_y, box.center_z]),
                        [box.size_x, box.size_y, box.size_z],
                        [box.orientation_x, box.orientation_y,
                         box.orientation_z, box.orientation_w], BOX_MARGIN)
                    if inside.any():
                        box_class[f, :n][inside] = CLASS_INDEX.get(
                            box.label_class, len(CLASSES))
                    # Every labelled object in the sector marks its cell,
                    # whether or not the radar happens to illuminate it -- the
                    # question is where things are, not where the returns are.
                    rng_b = float(np.hypot(box.center_x, box.center_y))
                    az_b = float(np.degrees(np.arctan2(box.center_y,
                                                       box.center_x)))
                    if rng_b < OCC_MAX_RANGE_M and abs(az_b) < 60.0:
                        ai = min(int((az_b + 60.0) / (120.0 / OCC_AZ_BINS)),
                                 OCC_AZ_BINS - 1)
                        ri = min(int(rng_b / (OCC_MAX_RANGE_M / OCC_RANGE_BINS)),
                                 OCC_RANGE_BINS - 1)
                        occupancy[f, ri * OCC_AZ_BINS + ai] = 1.0

        if self.thin_to is not None:
            self._thin(mask, i)

        return {"clip_id": clip_id,
                "points": torch.from_numpy(features),
                "mask": torch.from_numpy(mask),
                "is_moving": torch.from_numpy(moving),
                "box_class": torch.from_numpy(box_class),
                "occupancy": torch.from_numpy(occupancy),
                "ego_state": torch.from_numpy(ego_state),
                # Which sensor these points came from, for the routed experts.
                # A clip whose radar could not be read is `none`, not `lrr1`.
                "sensor": torch.tensor(
                    SENSOR_IDS[short] if radar is not None else SENSOR_IDS["none"])}

    def _empty(self, clip_id):
        F, P, C = self.n_frames, self.max_points, len(CHANNELS)
        return {"clip_id": clip_id,
                "points": torch.zeros(F, P, C),
                "mask": torch.zeros(F, P, dtype=torch.bool),
                "is_moving": torch.zeros(F, P, dtype=torch.long),
                "box_class": torch.full((F, P), -1, dtype=torch.long),
                "occupancy": torch.zeros(F, OCC_AZ_BINS * OCC_RANGE_BINS),
                "ego_state": torch.zeros(F, 3),
                "sensor": torch.tensor(SENSOR_IDS["none"])}


def collate(batch):
    out = {"clip_id": [b["clip_id"] for b in batch]}
    for key in ("points", "mask", "is_moving", "box_class", "occupancy",
                "ego_state", "sensor"):
        out[key] = torch.stack([b[key] for b in batch])
    return out
