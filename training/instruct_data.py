#!/usr/bin/env python3
"""Multi-task instruction dataset: video + radar + ego + text, one sample per item.

Every one of the eleven tasks is posed the same way -- the sensors are identical
and only the instruction changes -- so a single set of weights answers all of
them and the task is a property of the prompt, not of an output head.

  det_objects     01  class, range and azimuth of each road user ahead
  track_step      02  detections at t-4..t-1 in, objects at t out, ids carried
  plan_ego        03-1 future ego waypoints, (x, y) metres, forward-positive
  agent_traj      03-2 one agent's future range/azimuth over 3 s
  world_model     04  the scene 3 s on, conditioned on the ego action
  depth_range     05  nearest object, and the nearest the radar confirms
  motion_seg      06  moving vs stationary from the Doppler residual
  radar_transfer  07  LRR in, MRR statistics out -- the only genuinely paired
                      cross-radar supervision in the release
  (08)                not a task of its own: sensor profiles are real (ML / S /
                      none), so the loader feeds whichever radar a clip carries
                      and says so in the prompt. `radar_dropout` masks it on top.
  retrieval       09  the scenario tags that should retrieve this clip
  qa              10  39,158 multiple-choice items, every numeric claim recomputed
  desc_*          11  scene descriptions; `radar` and `complementarity` are the
                      two kinds that put a loss on the radar pathway
  ood_reasoning       2,077 human-verified Chain-of-Causation labels, held out of
                      training and used only for evaluation

Tasks 01-06 are pre-serialised by `datatools.frame_objects`; the geometry needs
three archives per frame, which does not belong in a dataloader worker.

Two of task 11's six kinds are deliberately not trained. `context` is
"Spain, at night, summer" -- country and month are not observable from a forward
camera, so it teaches the model to invent them. `qa_summary` describes the
dataset rather than the scene.

Two leakage rules are enforced here rather than left to the training script.

First, ego context is truncated at the frame a question asks about. Half the QA
set is `Anticipation` or `Planning`, and handing the model ego motion for the
whole 20 s clip would answer those outright: the future it is asked to predict is
in its own input. The same truncation is what keeps `plan_ego` honest -- its
target is the ego's own future, which is in the ego stream verbatim.
"""

import glob
import json
import os
import random
import time

import numpy as np
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from transformers.video_utils import VideoMetadata

from datatools import paths
from datatools.frame_objects import COT_INSTRUCTION
from training.connector import radar_prompt_block
from training.radar_encoder import SENSOR_IDS
from training.radar_data import RadarClipDataset
from training import frame_cache
from training.video_frames import (FRAME_HEIGHT, FRAME_WIDTH,
                                   clip_frames, pad_frames)

N_FRAMES = 20
RADAR_TOKENS = 256

SYSTEM = ("You are a driving-scene assistant. You receive 20 video frames at 1 Hz, "
          "a matching sequence of forward-radar observations, and the ego "
          "vehicle's own motion. Answer from what the sensors show. Range is in "
          "metres and azimuth in degrees, positive to the left of the ego "
          "heading. A line stating which sensors are present precedes every "
          "question; if a sensor is absent, do not report what it would have "
          "seen.")

# Answers that cannot be produced without a radar. When the radar is absent --
# a real sensor profile, or `radar_dropout` -- the target is replaced rather than
# kept, because training the original answer against a blank radar is precisely
# how a model learns to recite radar statistics it never read.
# Tasks answered from one instant rather than from the whole clip. They take a
# single video frame and the twenty radar scans immediately before that instant
# -- one second on LRR and MRR -- instead of twenty frames sampled at 1 Hz.
#
# The default input throws away 95% of the radar: the sensor runs at 20 Hz and
# only one scan per second survives. For a question about what is ahead *now*,
# the nineteen seconds of other context answer nothing the question asks, while
# the discarded scans carry the Doppler track that does. The vision side costs
# 78 tokens instead of 1,560 for the same reason -- one instant needs one image.
INSTANT_TASKS = frozenset({"det_objects_azdeg", "det_objects_3dbbox",
                           "det_objects_azdeg_cot",
                           "det_objects_3dbbox_cot"})

# Tracking sees a history, not an instant and not the whole clip: five seconds
# of it. The vision side is five frames at 1 Hz and the radar twenty scans at
# 4 Hz, so both cover the same five seconds at the resolution each is useful at
# -- the camera to recognise, the radar to measure how things are moving.
# (seconds, radar Hz, video frames) per task. Twenty radar scans always, so the
# rate follows the duration and the encoder's input shape never changes: five
# seconds at 4 Hz for tracking, two at 10 Hz for planning. The camera is sampled
# at 1 Hz either way, because that is what the video's keyframes are.
WINDOWS = {}
for _t in ("track_step_azdeg", "track_step_bbox",
           "track_step_azdeg_cot", "track_step_bbox_cot"):
    WINDOWS[_t] = (5, 4, 5)
for _t in ("plan_ego_xy", "plan_ego_control", "plan_ego_xy_cot",
           "plan_ego_control_cot", "agent_traj_azdeg", "agent_traj_bbox",
           "agent_traj_xy", "agent_traj_azdeg_cot", "agent_traj_bbox_cot",
           "agent_traj_xy_cot",
           "motion_seg_azdeg", "motion_seg_bbox",
           "motion_seg_azdeg_cot", "motion_seg_bbox_cot"):
    WINDOWS[_t] = (2, 10, 2)
WINDOW_TASKS = frozenset(WINDOWS)

# How large a frame each task is shown. Every task used 384x216, which the
# processor rounds to 384x224 and turns into a 12x7 grid of vision tokens --
# one token per 32x32 pixels, so one token spans 10 degrees of the 120 degree
# field. A car is 10.3 deg wide at 10 m and 3.4 deg at 30 m, so beyond about
# ten metres an object is a fraction of a single token, and 91.8% of labelled
# objects are narrower than one. Asking for azimuth to a degree off a grid that
# coarse is asking for something the representation cannot hold, and measured
# on the trained model the generated |azimuth| averaged 6.6 deg against 27.4 in
# the truth -- less than one token cell of spread.
#
# The budget was spent on frames rather than resolution, but the tasks that
# need azimuth barely use frames. `det_objects` reads one frame and so spends
# 84 of the 3,584 tokens; at 1152x648 it spends 720, a fifth of the budget, and
# the grid becomes 3.3 deg. `track_step` reads five, so it stops at 768x432.
# `qa` and the descriptions read all twenty and stay where they are.
FRAME_SIZE = {}
for _t in INSTANT_TASKS:                       # one frame: 84 -> 720 tokens
    FRAME_SIZE[_t] = (1152, 648)
    FRAME_SIZE[f"{_t}_cot"] = (1152, 648)
for _t in ("plan_ego_xy", "plan_ego_control", "agent_traj_azdeg",
           "agent_traj_bbox", "agent_traj_xy", "motion_seg_azdeg",
           "motion_seg_bbox"):                 # two frames: 84 -> 720
    FRAME_SIZE[_t] = (1152, 648)
    FRAME_SIZE[f"{_t}_cot"] = (1152, 648)
for _t in ("track_step_azdeg", "track_step_bbox"):   # five frames: 252 -> 1008
    FRAME_SIZE[_t] = (768, 432)
    FRAME_SIZE[f"{_t}_cot"] = (768, 432)
DEFAULT_FRAME_SIZE = (384, 216)

# Whatever the loops above decided, a task that reads all twenty frames stays
# small. At 1152 that is 7,200 vision tokens against a 3,584 budget, and the
# processor does not fail cleanly: the sequence is truncated and then the video
# token count no longer matches the text, so training dies at the first batch
# with a shape mismatch rather than at the table that caused it.
for _t in list(FRAME_SIZE):
    _n = 1 if _t in INSTANT_TASKS else (WINDOWS[_t][2] if _t in WINDOWS else 20)
    if _n >= 20:
        FRAME_SIZE[_t] = DEFAULT_FRAME_SIZE

# The budget is checked here rather than discovered at the first step.
_TOKENS = {(384, 216): 42, (768, 432): 168, (1152, 648): 360}   # per frame, 20-frame rate
for _t, _sz in FRAME_SIZE.items():
    _n = 1 if _t in INSTANT_TASKS else (WINDOWS[_t][2] if _t in WINDOWS else 20)
    _v = _TOKENS[_sz] * max(_n, 2)      # temporal patching pairs frames
    assert _v + 256 < 3200, f"{_t}: {_v} vision tokens leaves no room"

RADAR_ONLY_TASKS = frozenset({
    "radar_probe", "radar_probe_coarse", "radar_transfer",
    "motion_seg_azdeg", "motion_seg_bbox",
    "radar_structure", "radar_objects", "det_objects_cot", "depth_range_cot",
    "motion_seg_cot", "agent_traj_cot", "depth_range", "desc_radar",
    "desc_complementarity",
})
NO_RADAR_ANSWER = "Radar unavailable, so this cannot be determined."

# A batch costs what its *longest* item costs. The loss builds logits of shape
# [batch, padded length, 151,672] and upcasts them to fp32, which is 600 kB per
# token, and every item in the batch is padded to the longest one. One 4,096-
# token item among sixteen therefore asks for 37 GiB in a single allocation --
# which is what killed a run at step 1,630 after six and a half hours.
#
# Measured over all 33.4 M items: 01-06 top out at 2,887 characters and the
# descriptions at 342, while `qa_cot` has a tail reaching 10,825. Fourteen items
# exceed 4,000 characters and every one of them is `qa_cot`. Cutting there costs
# 14 items in 33,429,895 and bounds the longest sample at about 3,480 tokens,
# under the 3,584 the trainer now allows -- so nothing is truncated either.
#
# Characters rather than tokens because this runs over every item at load time
# and tokenising 33 M strings to discard fourteen of them is not a trade.
MAX_ITEM_CHARS = 4000

SENSOR_LABEL = {"lrr1": "imaging long-range radar",
                "mrr2": "mid-range radar",
                "srr0": "short-range radar"}

# Tasks pre-serialised by datatools.frame_objects into one parquet.
# `det_objects` split in two: the input is identical and the instruction picks
# the representation, which is the premise of the model stated as a task pair.
# 04 world_model, 05 depth_range and 09 retrieval are held back from this
# build. They are still defined elsewhere; they are simply not in the task set
# this dataset was generated for, and leaving them in the lists would have the
# loader ask for rows that are not there.
FRAME_ITEM_TASKS = ("det_objects_azdeg", "det_objects_3dbbox",
                    "track_step_azdeg", "track_step_bbox", "plan_ego_xy",
                    "plan_ego_control", "agent_traj_azdeg",
                    "agent_traj_bbox", "agent_traj_xy", "motion_seg_azdeg",
                    "motion_seg_bbox")

# Same questions, but the answer is {"rationale": ..., "answer": ...} and the
# rationale has to state the radar reading first. Only two tasks qualify: a
# rationale is worth training when the radar quantity sits upstream of the
# answer, and for `radar_probe` the quantity *is* the answer, so a rationale
# there would just be the answer written twice. Validated on 1,296 items -- the
# direction the rationale claims agrees with the actual range change 92% of the
# time for closing and 80% for receding, against 52% for always guessing.
COT_TASKS = ("agent_traj_azdeg_cot", "agent_traj_bbox_cot",
             "agent_traj_xy_cot",
             "motion_seg_azdeg_cot", "motion_seg_bbox_cot", "det_objects_azdeg_cot",
             "det_objects_3dbbox_cot", "track_step_azdeg_cot",
             "track_step_bbox_cot", "plan_ego_xy_cot",
             "plan_ego_control_cot", "qa_cot")

# What `--tasks all` means. `ood_reasoning` is not in it: 2,077 events are the
# only human-verified text in the release, and they are worth more as an
# evaluation set nothing was fitted to.
# `all` has to reach the CoT variants too. It did not, and nothing said so: the
# eleven `_cot` tasks are 10,848,192 items -- 32% of the training data and the
# whole point of the redefinition, which was that every answer arrives after a
# stated reason. A run with `--tasks all` trained on none of them and reported
# no error, because a task nobody asks for is simply absent from the mixture.
ALL_TRAIN_TASKS = (FRAME_ITEM_TASKS + COT_TASKS
                   + ("qa", "description", "radar_probe"))


def expand_tasks(spec):
    """Comma-separated task list where `all` stands for every trained task.

    `all,ood_reasoning` is the evaluation form: everything that was trained,
    plus the held-out human-verified set.
    """
    out = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        out.extend(ALL_TRAIN_TASKS if token == "all" else [token])
    return tuple(dict.fromkeys(out))

# Ego channels are quantised rather than printed as floats: 32 bins each, edges
# taken from measured percentiles. Feeding "9.34 m/s" as text spends several
# tokens on digits the model has to learn to parse.
EGO_BINS = 32
EGO_EDGES = {
    "speed": np.linspace(0.0, 32.0, EGO_BINS + 1),
    "accel": np.linspace(-4.0, 4.0, EGO_BINS + 1),
    "yaw": np.linspace(-25.0, 25.0, EGO_BINS + 1),
}


def quantise(value, key):
    return int(np.clip(np.digitize(value, EGO_EDGES[key]) - 1, 0, EGO_BINS - 1))


def ego_text(ego_state, up_to_frame, times=None):
    """Ego motion as bin indices, truncated at `up_to_frame` inclusive.

    `times`, when given, labels each reading with its own offset in seconds
    instead of a bare index. `t0..t19` meant one second apart in a clip-level
    task and a tenth of a second in a two-second window -- the same notation
    for two different things, and the header was the only place that said
    which. Reading it as twenty seconds of context is the natural mistake, and
    for `plan_ego`, whose answer is the ego's own future, that mistake looks
    exactly like a leak.
    """
    rows = []
    for f in range(min(up_to_frame + 1, len(ego_state))):
        speed, accel, yaw = ego_state[f].tolist()
        label = f"t{f}" if times is None else f"{times[f]:+.0f}s"
        rows.append(f"{label}:s{quantise(speed,'speed')}"
                    f"a{quantise(accel,'accel')}y{quantise(yaw,'yaw')}")
    return " ".join(rows)


def frame_reference(text):
    """The frame a question or description is about, or the last frame."""
    import re
    found = re.findall(r"[Ff]rames?\s+(\d+)", text or "")
    if not found:
        return N_FRAMES - 1
    # Frames are 1-indexed in the text, 0-indexed here.
    return int(np.clip(int(found[0]) - 1, 0, N_FRAMES - 1))


# Relative weight per task in the training mixture. `qa` is camera-grounded --
# its rationales cite obstacle and egomotion values, never radar returns -- so it
# is held to the same weight as any single description kind rather than being
# allowed to dominate. `radar` and `complementarity` are doubled because they are
# the only text that puts a loss on the radar pathway at all.
# Five of these keys named tasks that no longer exist -- `agent_traj_cot` and
# `motion_seg_cot` were renamed when each task split into an azdeg and a bbox
# form, so the two kinds the weights were meant to double got the default 1.0
# instead. A key that matches nothing is silent: the mixture still sums, still
# samples, and simply weights something else.
#
# Each `_cot` variant now carries its plain twin's weight. They are the same
# question with the reason spoken aloud, so weighting them apart would be a
# claim about which of the two the model should prefer, and there is no
# measurement behind such a claim.
# Within a task, balance the exposure of subgroups the prompt distinguishes.
#
# `plan_ego` carries a navigation command, and 88.2% of its anchors are
# STRAIGHT (887,470 of 1,006,678, against LEFT 58,478 and RIGHT 60,730). A
# model that never reads the command therefore loses almost nothing, which
# defeats the point of conditioning on it. Cutting the data to 1:1:1 would fix
# that by deleting 83% of the task -- 1,006,678 items down to 175,434 -- and
# would freeze the ratio into the parquet.
#
# Weighting the draw instead keeps every item and makes the ratio a number that
# can be changed without a rebuild. 2:1:1 rather than 1:1:1 because straight
# driving is not an artefact to be corrected away: it is what the road mostly
# is, and a model shown turns a third of the time learns a prior that will make
# it turn when it should not.
STRATA = {
    "plan_ego_xy":          {"STRAIGHT": 2, "LEFT": 1, "RIGHT": 1},
    "plan_ego_control":     {"STRAIGHT": 2, "LEFT": 1, "RIGHT": 1},
    "plan_ego_xy_cot":      {"STRAIGHT": 2, "LEFT": 1, "RIGHT": 1},
    "plan_ego_control_cot": {"STRAIGHT": 2, "LEFT": 1, "RIGHT": 1},
}

DEFAULT_MIXTURE = {
    # radar-dependent: the only items whose answer changes when the radar does
    "radar_probe": 2.0,
    "motion_seg_azdeg": 2.0,
    "motion_seg_bbox": 2.0,
    "motion_seg_azdeg_cot": 2.0,
    "motion_seg_bbox_cot": 2.0,
    "desc_radar": 2.0,
    "desc_complementarity": 2.0,
    # camera- and ego-grounded
    "qa": 2.0,
    "qa_cot": 2.0,
    "det_objects_azdeg": 1.5,
    # 3D 상자가 이 작업의 주력이 되었으므로 배치의 30% 를 준다. 7.5% 였는데,
    # 그것은 스물여섯 종에 고르게 나눈 결과이지 무엇을 보이려는지와는 무관한
    # 숫자였다. 나머지 가중치 합이 37.0 이므로 두 종에 각각 7.93 이면
    # 2*7.93 / (37.0 + 2*7.93) = 30.0% 다.
    "det_objects_3dbbox": 7.93,
    "det_objects_azdeg_cot": 1.5,
    "det_objects_3dbbox_cot": 7.93,
    "plan_ego_xy": 1.5,
    "plan_ego_control": 1.5,
    "plan_ego_xy_cot": 1.5,
    "plan_ego_control_cot": 1.5,
    "agent_traj_azdeg": 1.5,
    "agent_traj_bbox": 1.5,
    "agent_traj_xy": 1.5,
    "agent_traj_azdeg_cot": 1.5,
    "agent_traj_bbox_cot": 1.5,
    "agent_traj_xy_cot": 1.5,
    "track_step_azdeg": 1.0,
    "track_step_bbox": 1.0,
    "track_step_azdeg_cot": 1.0,
    "track_step_bbox_cot": 1.0,
    "desc_objects": 1.0,
    "desc_ego_maneuver": 1.0,
    "desc_clip_summary": 1.0,
    # Evaluation-only, and kept so an explicit `--tasks` can still weight them.
    "radar_transfer": 2.0,
    "ood_reasoning": 1.0,
}


# `val` is folded into `train`. The clip-level split was inherited from the
# release and gives train 86,607 / val 54,163 / test 37,121 -- a third of the
# data sitting in a validation set nothing selects on. Model selection here uses
# `test` (the human-checked QA clips and the clips held out with them), so the
# validation third was simply unused. `test` is untouched.
SPLITS = {"train": ("train", "val"), "val": ("val",), "test": ("test",)}


def in_split(series, split):
    """Boolean mask for the splits a requested split covers."""
    return series.isin(SPLITS.get(split, (split,)))


QA_TRAIN_DIR = "10_radar_vision_qa/qa_train"
QA_TEST_DIR = "10_radar_vision_qa/qa_gt"


def qa_holdout():
    """The clips task 10 is tested on, excluded from training everywhere.

    These are the 139 clips of `qa_gt` -- 2,019 questions a person checked,
    which is the only human-verified text in the release. They supersede the
    sha1-selected 99 clips that stood in for a test set while there was none;
    those clips are back in training, and nothing was ever evaluated on them.

    The exclusion applies to every task, not only QA. A held-out clip still
    produces detection, tracking and description items, and leaving those in
    training would mean the model had already seen the clip's video and radar
    before being questioned about it -- the split would be nominal.
    """
    root = os.path.join(paths.SPLIT_ROOT, QA_TEST_DIR)
    if not os.path.isdir(root):
        return frozenset()
    return frozenset(json.load(open(f))["clip_id"]
                     for f in glob.glob(os.path.join(root, "*.json")))


def usable_clips(split, radar="lrr1", all_profiles=False, clips=None):
    """Which clips a split may draw from, and which radar each one carries.

    Kept out of `InstructDataset.__init__` because it is policy, not plumbing,
    and more than one thing needs to agree with it -- the loader that builds
    batches and `datatools.dataset_summary`, which reports how large the data
    is. Two copies of this would drift and the reported numbers would stop
    describing what was trained on.

    Returns `(index, {clip_id: radar or None})`.
    """
    if clips is None:
        clips = pd.read_parquet(
            os.path.join(paths.COMMON_DIR, "nvidia_clips.parquet"))
    # Everything a sample needs that is never optional: both label streams, the
    # ego stream and the camera archive.
    base = clips[clips["has_egomotion"].fillna(False)
                 & clips["has_obstacle"].fillna(False)
                 & clips["camera_zip"].notna()
                 & in_split(clips["split"], split)]
    has_extrinsics = base["has_radar_extrinsics"].fillna(False)
    primary = base[base[f"has_{radar}"].fillna(False) & has_extrinsics]

    # Sensor profiles are a property of the vehicle rig, not a perturbation:
    # 'low' rigs carry only the SRR, so `all_profiles` reaches for it rather
    # than dropping those clips.
    #
    # Clips with no front radar at all are a different case and are excluded
    # everywhere. They are 17,130 of 177,891 and were 9.6% of the items. The
    # answer was never in doubt -- it comes from the obstacle labels and
    # egomotion -- but the rationale said "no radar return" for every object,
    # which claims the sensor looked and saw nothing where the truth is that
    # there is no sensor. Blanking a radar that was really there is
    # `radar_dropout`, and that path replaces the target with a refusal rather
    # than leaving a rationale that describes returns nobody has.
    radar_of = {c: radar for c in primary.index}
    if all_profiles:
        fallback = base[~base.index.isin(primary.index)]
        srr = fallback[fallback["has_srr0"].fillna(False)
                       & fallback["has_radar_extrinsics"].fillna(False)]
        for c in srr.index:
            radar_of[c] = "srr0"
    usable = pd.Index(radar_of.keys())

    # Task 10 has no val or test clips of its own, so a fixed set is carved out
    # of its training clips. They leave training for every task and come back
    # only as `test`.
    holdout = qa_holdout()
    if holdout:
        if split == "train":
            usable = usable.difference(pd.Index(sorted(holdout)))
        elif split == "test":
            usable = usable.union(
                pd.Index([c for c in holdout if c in clips.index]))
    # The holdout union adds clips back by id alone, so the radar requirement is
    # re-applied after it rather than only before.
    carries = clips.index[clips["has_radar_extrinsics"].fillna(False)
                          & (clips["has_lrr1"].fillna(False)
                             | clips["has_mrr2"].fillna(False)
                             | clips["has_srr0"].fillna(False))]
    usable = usable.intersection(carries)
    radar_of = {c: r for c, r in radar_of.items() if c in set(usable)}
    for clip in usable:
        radar_of.setdefault(clip, radar)
    return usable, radar_of


def load_items(tasks, split, limit_per_task=0, usable_clips=None, forms=None):
    """Flat list of {clip_id, task, prompt, target, frame} records.

    `usable_clips` filters as items are built rather than afterwards. Filtering
    later meant a head-truncating limit picked mostly SRR-equipped clips, which
    were then dropped for lacking the long-range radar -- descriptions collapsed
    from 12,000 to 2,350 and QA ended up at 68% of the mixture.

    `forms` restricts `radar_objects` to a subset of its question forms, so a run
    can train on some and leave the rest as instruments. Training every form
    would repeat the mistake `radar_probe` embodies: a model graded on exactly
    what it was optimised for tells you nothing about whether it can read radar.
    """
    items = []
    root = paths.SPLIT_ROOT
    keep = None if usable_clips is None else set(usable_clips)

    if "qa" in tasks or "qa_cot" in tasks:
        # Two directories now: `qa_train` is the 1,999 original clips merged with
        # the 8,000 added later, both run through the same verification and
        # correction, and `qa_gt` is the human-checked test set. They share no
        # clip, so the split is a directory rather than a filter.
        holdout = qa_holdout()
        folder = QA_TEST_DIR if split == "test" else QA_TRAIN_DIR
        # The QA documents all record split='train' whatever clip they describe,
        # so their own field cannot separate train from val. The clip-level
        # split in `nvidia_clips` is what every other task uses and is the one
        # that keeps a clip's items together on one side of the line.
        clip_split = pd.read_parquet(
            os.path.join(paths.COMMON_DIR, "nvidia_clips.parquet"),
            columns=["split"])["split"]
        for path in sorted(glob.glob(os.path.join(root, folder, "*.json"))):
            doc = json.load(open(path))
            if split != "test":
                if doc["clip_id"] in holdout:
                    continue
                if clip_split.get(doc["clip_id"]) not in SPLITS.get(split, (split,)):
                    continue
            if keep is not None and doc["clip_id"] not in keep:
                continue
            for item in doc.get("qa", []):
                options = "\n".join(f"{k}. {v}" for k, v in sorted(item["options"].items()))
                verified = item.get("verification", {}).get("status") == "agrees"
                if "qa" in tasks:
                    items.append({
                        "clip_id": doc["clip_id"], "task": "qa",
                        "prompt": f"{item['question']}\n{options}\n"
                                  "Answer with the letter.",
                        "target": item["answer"],
                        "frame": frame_reference(item["question"]),
                        "verified": verified,
                    })
                # The QA release ships a worked rationale with every question --
                # "the ego is at 5.74 m/s, the sedan at 5.22, the difference is
                # 0.52" -- and here the reasoning is genuinely upstream: the
                # letter is whichever option the computed number lands on.
                #
                # Everything is emitted except what the checker actively
                # contradicts, which after the correction pass is nothing. The
                # earlier rule kept only `agrees`, but that verdict means "had
                # numbers and they matched", not "is correct": 62% of the set is
                # `unchecked` because its reasoning is qualitative -- "the object
                # directly ahead in its lane is the red automobile" -- and
                # discarding it would have taught the model that reasoning is
                # arithmetic. Sampled, only 13.7% of the unchecked rationales
                # contain no number at all; the rest were unchecked because the
                # extractor had no pattern for their form.
                unsound = (item.get("verification", {}).get("status")
                           == "disagrees")
                if "qa_cot" in tasks and not unsound and item.get("rationale"):
                    items.append({
                        "clip_id": doc["clip_id"], "task": "qa_cot",
                        "prompt": f"{item['question']}\n{options}\n"
                                  "Answer with the letter." + COT_INSTRUCTION,
                        "target": json.dumps({"rationale": item["rationale"],
                                              "answer": item["answer"]}),
                        "frame": frame_reference(item["question"]),
                        "verified": verified,
                    })

    if "description" in tasks:
        frame_path = os.path.join(root, "11_radar_vision_scene_description",
                                  "descriptions_frame.parquet")
        if os.path.exists(frame_path):
            frame_df = pd.read_parquet(frame_path)
            frame_df = frame_df[in_split(frame_df["split"], split)]
            if keep is not None:
                frame_df = frame_df[frame_df["clip_id"].isin(keep)]
            asked = {
                "objects": "List the road users in the forward sector.",
                "ego_maneuver": "Describe what the ego vehicle is doing.",
                "radar": "Describe what the forward radar observes.",
                "complementarity": "Which labelled objects does the radar see, "
                                   "and which are visible to the camera only?",
            }
            for row in frame_df.itertuples():
                items.append({
                    "clip_id": row.clip_id, "task": f"desc_{row.kind}",
                    "prompt": f"At frame {row.frame}. {asked.get(row.kind, 'Describe the scene.')}",
                    "target": row.text, "frame": int(row.frame) - 1,
                    "verified": True,
                })

        # Clip-level descriptions. `context` and `qa_summary` are skipped: see
        # the module docstring.
        clip_path = os.path.join(root, "11_radar_vision_scene_description",
                                 "descriptions_clip.parquet")
        if os.path.exists(clip_path):
            clip_df = pd.read_parquet(clip_path)
            clip_df = clip_df[in_split(clip_df["split"], split)
                              & (clip_df["kind"] == "clip_summary")]
            if keep is not None:
                clip_df = clip_df[clip_df["clip_id"].isin(keep)]
            for row in clip_df.itertuples():
                items.append({
                    "clip_id": row.clip_id, "task": "desc_clip_summary",
                    "prompt": "Summarise the whole 20 s clip: what the ego "
                              "vehicle did and what was around it.",
                    "target": row.text, "frame": N_FRAMES - 1,
                    "verified": True,
                })

    if "radar_probe_coarse" in tasks:
        # The same three questions banded instead of exact. Every recipe tried so
        # far lands 2-4 points above the 23.9% a per-position digit prior scores
        # on `radar_probe`, whatever is done to the training. One explanation is
        # that the radar is unreadable through this interface; the other is that
        # emitting "638" digit by digit is a hard way to express a quantity the
        # encoder only knows to R^2 0.71. Bands separate the two: if precision was
        # the barrier these scores go up, and if it was not they do not.
        features = pd.read_parquet(
            os.path.join(paths.COMMON_DIR, "scene_features_all_clips.parquet"),
            columns=["clip_id", "frame", "lrr1_n_points", "lrr1_n_moving",
                     "lrr1_max_rcs", "n_boxes_with_radar", "n_boxes_camera_only"])
        features = features[features["lrr1_n_points"].notna()]
        if keep is not None:
            features = features[features["clip_id"].isin(keep)]

        def band(value, step):
            return int(round(value / step) * step)

        probes = [
            ("Roughly how many radar detections are there at this frame, and "
             "roughly how many are moving? Answer to the nearest 50 and 10.",
             lambda r: f"about {band(r.lrr1_n_points, 50)} detections, "
                       f"about {band(r.lrr1_n_moving, 10)} moving."),
            ("Roughly how strong is the strongest radar return at this frame? "
             "Answer to the nearest 5 dBsm.",
             lambda r: f"about {band(r.lrr1_max_rcs, 5)} dBsm."),
            ("Roughly how many objects ahead does the radar illuminate, and "
             "roughly how many are camera-only? Answer to the nearest 5.",
             lambda r: f"about {band(r.n_boxes_with_radar, 5)} illuminated, "
                       f"about {band(r.n_boxes_camera_only, 5)} camera-only."),
        ]
        for row in features.itertuples():
            question, answer = probes[int(row.frame) % len(probes)]
            items.append({
                "clip_id": row.clip_id, "task": "radar_probe_coarse",
                "prompt": f"At frame {int(row.frame)}. {question}",
                "target": answer(row), "frame": int(row.frame) - 1,
                "verified": True,
            })

    if "radar_probe" in tasks:
        # The description tasks can be answered from a memorised prior: every
        # `radar` sentence has the same skeleton and only the numbers move, so a
        # model that guesses "about 700 returns, about 10% moving" scores well
        # without reading the radar at all. That is exactly what the first 8B run
        # did -- shuffling the radar cost only +0.011 nll. These ask for the
        # numbers alone, where the skeleton carries no information and the answer
        # is different for every clip.
        features = pd.read_parquet(
            os.path.join(paths.COMMON_DIR, "scene_features_all_clips.parquet"),
            columns=["clip_id", "frame", "lrr1_n_points", "lrr1_n_moving",
                     "lrr1_max_rcs", "n_boxes_with_radar", "n_boxes_camera_only"])
        features = features[features["lrr1_n_points"].notna()]
        if keep is not None:
            features = features[features["clip_id"].isin(keep)]
        probes = [
            ("How many radar detections are there at this frame, and how many "
             "are moving?",
             lambda r: f"{int(r.lrr1_n_points)} detections, "
                       f"{int(r.lrr1_n_moving)} moving."),
            ("What is the strongest radar return at this frame, in dBsm?",
             lambda r: f"{r.lrr1_max_rcs:.0f} dBsm."),
            ("How many objects ahead does the radar illuminate, and how many are "
             "camera-only?",
             lambda r: f"{int(r.n_boxes_with_radar)} illuminated, "
                       f"{int(r.n_boxes_camera_only)} camera-only."),
        ]
        for row in features.itertuples():
            question, answer = probes[int(row.frame) % len(probes)]
            items.append({
                "clip_id": row.clip_id, "task": "radar_probe",
                "prompt": f"At frame {int(row.frame)}. {question}",
                "target": answer(row), "frame": int(row.frame) - 1,
                "verified": True,
            })

    wanted_frame_items = [t for t in FRAME_ITEM_TASKS + COT_TASKS if t in tasks]
    if wanted_frame_items:
        path = os.path.join(paths.COMMON_DIR, "instruct_items_tasks01_06.parquet")
        if os.path.exists(path):
            cols = ["clip_id", "task", "frame", "prompt", "target", "split"]
            have = set(pq.ParquetFile(path).schema.names)
            if "strat" in have:
                cols.append("strat")
            built = pd.read_parquet(path, columns=cols)
            if "strat" not in built.columns:
                built["strat"] = ""
            built = built[in_split(built["split"], split)
                          & built["task"].isin(wanted_frame_items)]
            if keep is not None:
                built = built[built["clip_id"].isin(keep)]
            for row in built.itertuples():
                items.append({
                    "clip_id": row.clip_id, "task": row.task,
                    "prompt": f"At frame {int(row.frame)}. {row.prompt}",
                    "target": row.target, "frame": int(row.frame) - 1,
                    "verified": True, "strat": row.strat or "",
                })

    if "radar_objects" in tasks:
        # Object-level, and trainable -- this is the one difference from
        # `radar_structure`. Measured on 382,610 frames with both feature sets
        # on the same rows, the nearest illuminated object's range, bearing and
        # radial velocity explain 0.581 of `depth_range`, 0.289 of
        # `det_objects` and 0.156 of `motion_seg`, against 0.048 / 0.023 / 0.014
        # for the scan-level summary. A reward can only shape what it scores, so
        # this is the granularity worth scoring.
        #
        # `forms` exists so some question forms can be trained while the rest
        # stay instruments: training all of them would repeat the `radar_probe`
        # mistake of grading a model on what it was optimised for.
        path = os.path.join(paths.COMMON_DIR, "radar_object_probes.parquet")
        built = pd.read_parquet(path, columns=["clip_id", "frame", "form",
                                               "prompt", "target", "split"])
        built = built[in_split(built["split"], split)]
        if forms is not None:
            built = built[built["form"].isin(forms)]
        if keep is not None:
            built = built[built["clip_id"].isin(keep)]
        for row in built.itertuples():
            items.append({
                "clip_id": row.clip_id, "task": "radar_objects",
                "prompt": row.prompt, "target": row.target,
                "frame": int(row.frame) - 1, "verified": True,
            })

    if "radar_structure" in tasks:
        # Evaluation only, and deliberately absent from ALL_TRAIN_TASKS: these
        # ask what `head_stats` never taught the encoder to write down, so a
        # score here is transfer. Training on them would turn them into another
        # memorised skeleton and destroy the only honest instrument we have.
        path = os.path.join(paths.COMMON_DIR, "radar_structure_probes.parquet")
        built = pd.read_parquet(path,
                                columns=["clip_id", "frame", "prompt", "target",
                                         "split"])
        built = built[in_split(built["split"], split)]
        if keep is not None:
            built = built[built["clip_id"].isin(keep)]
        for row in built.itertuples():
            items.append({
                "clip_id": row.clip_id, "task": "radar_structure",
                "prompt": row.prompt, "target": row.target,
                "frame": int(row.frame) - 1, "verified": True,
            })

    if "radar_transfer" in tasks:
        # The one paired cross-radar signal that exists: MRR and imaging LRR
        # observe the same scene at the same instant on med/high rigs. SRR never
        # co-occurs with either, so SRR -> LRR cannot be posed this way at all.
        features = pd.read_parquet(
            os.path.join(paths.COMMON_DIR, "scene_features_all_clips.parquet"),
            columns=["clip_id", "frame", "mrr2_n_points", "mrr2_n_moving",
                     "mrr2_p90_range_m", "lrr1_n_points"])
        features = features[features["mrr2_n_points"].notna()
                            & features["lrr1_n_points"].notna()]
        if keep is not None:
            features = features[features["clip_id"].isin(keep)]
        probes = [
            ("The mid-range radar covers this sector at lower resolution. How "
             "many detections would it report, and how many moving?",
             lambda r: f"{int(r.mrr2_n_points)} detections, "
                       f"{int(r.mrr2_n_moving)} moving."),
            ("How far out does the mid-range radar reach at this frame, at the "
             "90th percentile of its detections?",
             lambda r: f"{r.mrr2_p90_range_m:.0f} m."),
        ]
        for row in features.itertuples():
            question, answer = probes[int(row.frame) % len(probes)]
            items.append({
                "clip_id": row.clip_id, "task": "radar_transfer",
                "prompt": f"At frame {int(row.frame)}. {question}",
                "target": answer(row), "frame": int(row.frame) - 1,
                "verified": True,
            })

    if "retrieval" in tasks:
        corpus = pd.read_parquet(
            os.path.join(root, "09_scenario_retrieval/nvidia_corpus_all.parquet"),
            columns=["clip_id", "split", "is_night", "ood_cluster"])
        corpus = corpus[in_split(corpus["split"], split)]
        if keep is not None:
            corpus = corpus[corpus["clip_id"].isin(keep)]
        # Tags are built only from what the sensors can show. `country` is in the
        # corpus and is not: a forward camera cannot see which country it is in,
        # so training it would reward guessing from the release's geography.
        anchor = 11
        features = pd.read_parquet(
            os.path.join(paths.COMMON_DIR, "scene_features_all_clips.parquet"),
            columns=["clip_id", "frame", "ego_speed_ms", "n_tracks_in_fov",
                     "n_moving_in_fov", "nearest_class"])
        features = features[features["frame"] == anchor].set_index("clip_id")
        for row in corpus.itertuples():
            if row.clip_id not in features.index:
                continue
            scene = features.loc[row.clip_id]
            tags = ["night" if row.is_night else "daytime",
                    f"ego {scene.ego_speed_ms:.0f} m/s"]
            if int(scene.n_tracks_in_fov) > 0:
                tags.append(f"{int(scene.n_tracks_in_fov)} objects ahead, "
                            f"{int(scene.n_moving_in_fov)} moving")
                if isinstance(scene.nearest_class, str):
                    tags.append(f"nearest {scene.nearest_class}")
            else:
                tags.append("no objects ahead")
            if isinstance(row.ood_cluster, str):
                tags.append(row.ood_cluster.lower().replace("_", " "))
            items.append({
                "clip_id": row.clip_id, "task": "retrieval",
                "prompt": "Write the scenario tags that should retrieve this clip.",
                "target": "; ".join(tags), "frame": anchor - 1,
                "verified": True,
            })

    if "ood_reasoning" in tasks:
        path = os.path.join(root, "09_scenario_retrieval/nvidia_ood_queries_all.parquet")
        if os.path.exists(path):
            events = pd.read_parquet(path)
            events = events[in_split(events["split"], split)]
            if keep is not None:
                events = events[events["clip_id"].isin(keep)]
            for row in events.itertuples():
                items.append({
                    "clip_id": row.clip_id, "task": "ood_reasoning",
                    "prompt": "What should the ego vehicle do here, and why?",
                    "target": row.coc_text,
                    "frame": int(np.clip(row.event_start_frame // 30, 0, N_FRAMES - 1)),
                    "verified": True,
                })
    return [i for i in items
            if len(i["prompt"]) + len(i["target"]) <= MAX_ITEM_CHARS]


class InstructDataset(Dataset):
    def __init__(self, tasks=("qa", "description", "ood_reasoning"), split="train",
                 processor=None, tokenizer=None, radar="lrr1",
                 n_frames=N_FRAMES, radar_tokens=RADAR_TOKENS,
                 verified_only=False, samples=0, mixture=None, seed=0,
                 max_text_tokens=512, nvidia_root=paths.NVIDIA_ROOT,
                 all_profiles=False, radar_dropout=0.0, forms=None):
        self.clips = pd.read_parquet(
            os.path.join(paths.COMMON_DIR, "nvidia_clips.parquet"))
        usable, self.radar_of = usable_clips(split, radar, all_profiles,
                                             clips=self.clips)

        self.items = load_items(tasks, split, usable_clips=usable,
                                forms=forms)
        if verified_only:
            self.items = [i for i in self.items if i["verified"]]
        self.available = self._counts(self.items)
        if samples:
            self.items = self._mix(self.items, samples,
                                   mixture or DEFAULT_MIXTURE, seed)
        self.processor = processor
        self.tokenizer = tokenizer
        self.n_frames = n_frames
        self.radar_tokens = radar_tokens
        self.max_text_tokens = max_text_tokens
        self.nvidia_root = nvidia_root
        self.radar_dropout = radar_dropout
        self.seed = seed

        # One reader per radar type actually in play. Clips with no front radar
        # still go through a reader -- it returns their ego motion and a blank
        # point cloud, which is the honest representation of that profile.
        clip_ids = sorted({i["clip_id"] for i in self.items})
        groups = {}
        for clip_id in clip_ids:
            groups.setdefault(self.radar_of.get(clip_id) or radar, []).append(clip_id)
        self.radar_ds = {name: RadarClipDataset(ids, radar=name, n_frames=n_frames)
                         for name, ids in groups.items()}
        self.radar_pos = {}
        for name, dataset in self.radar_ds.items():
            for position, clip_id in enumerate(dataset.clip_ids):
                self.radar_pos[clip_id] = (name, position)
        self.items = [i for i in self.items if i["clip_id"] in self.radar_pos]

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _counts(items):
        counts = {}
        for item in items:
            counts[item["task"]] = counts.get(item["task"], 0) + 1
        return counts

    def task_counts(self):
        return self._counts(self.items)

    @staticmethod
    def _mix(items, samples, mixture, seed):
        """Draw `samples` items with the requested task proportions.

        Sampled rather than truncated: taking the head of each task biases
        towards whatever happens to sort first, which here was one vehicle
        platform. A task with fewer items than its share gets all of them and the
        remainder is redistributed, so a small task cannot silently shrink the
        epoch.
        """
        rng = random.Random(seed)
        buckets = {}
        for item in items:
            task = item["task"]
            strat = item.get("strat") or ""
            key = (task, strat) if strat and task in STRATA else task
            buckets.setdefault(key, []).append(item)
        # A stratified task splits its own weight across its subgroups, so
        # balancing inside a task never changes how much of the epoch that task
        # gets against the others.
        weights = {}
        for key in buckets:
            if isinstance(key, tuple):
                task, strat = key
                table = STRATA[task]
                share = table.get(strat, 1.0) / (sum(table.values()) or 1.0)
                weights[key] = mixture.get(task, 1.0) * share
            else:
                weights[key] = mixture.get(key, 1.0)
        total_weight = sum(weights.values()) or 1.0

        chosen, shortfall = [], 0
        picked = set()          # identity, since the records are plain dicts
        for task, bucket in sorted(buckets.items(), key=str):
            want = int(samples * weights[task] / total_weight)
            take = min(want, len(bucket))
            shortfall += want - take
            drawn = rng.sample(bucket, take)
            chosen.extend(drawn)
            picked.update(id(i) for i in drawn)
        if shortfall:
            leftover = [i for _, b in sorted(buckets.items(), key=str) for i in b
                        if id(i) not in picked]
            rng.shuffle(leftover)
            chosen.extend(leftover[:shortfall])
        # Shuffle so consecutive items -- and therefore every batch -- carry a
        # mixture of tasks rather than runs of one.
        rng.shuffle(chosen)
        return chosen

    def __getitem__(self, i):
        item = self.items[i]
        clip_id = item["clip_id"]
        name, position = self.radar_pos[clip_id]
        instant = item["task"] in INSTANT_TASKS
        window = item["task"] in WINDOW_TASKS
        if instant:
            # `frame` is 0-indexed and one frame is one second, so the instant
            # asked about is t = frame seconds.
            radar = self.radar_ds[name].window(position, float(item["frame"]))
        elif window:
            seconds, radar_hz, n_frames = WINDOWS[item["task"]]
            radar = self.radar_ds[name].window(position, float(item["frame"]),
                                               rate_hz=radar_hz)
        else:
            radar = self.radar_ds[name][position]
        # Work out which frames this item needs before decoding anything. The
        # old order decoded all twenty and then sliced, so a `det_objects` item
        # that uses one frame paid for twenty. With the cache the wanted indices
        # are read individually and the rest are never touched.
        last = self.n_frames - 1
        if instant:
            wanted = [min(int(item["frame"]), last)]
        elif window:
            end = min(int(item["frame"]), last)
            wanted = list(range(max(0, end - n_frames + 1), end + 1))
        else:
            wanted = list(range(self.n_frames))
        frames = frame_cache.load(clip_id, wanted)
        if frames is None:
            # No cache for this clip -- decode the mp4 and slice as before.
            # 같은 이유로 여기도 재시도한다. 한 클립을 못 읽었다고 2백만
            # 샘플짜리 학습을 죽일 이유가 없고, 그렇다고 조용히 빈 화면을
            # 넣으면 프레임이 스무 장 다 같았던 그 결함처럼 숨는다.
            decoded = None
            for attempt in range(4):
                try:
                    decoded = pad_frames(clip_frames(self.nvidia_root,
                                                     self.clips.loc[clip_id]),
                                         self.n_frames)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(0.2 * 2 ** attempt)
            frames = [decoded[i] for i in wanted]

        # The cache holds one size; each task is shown the size its token
        # budget allows. Upscaling a 384-wide frame recovers nothing the
        # decoder threw away, so the cache is built at the largest size any
        # task asks for and every other task scales down from it.
        want_size = FRAME_SIZE.get(item["task"], DEFAULT_FRAME_SIZE)
        if frames and frames[0].size != want_size:
            frames = [f.resize(want_size, Image.BILINEAR) for f in frames]

        points, mask = radar["points"], radar["mask"]
        present = self.radar_of.get(clip_id)
        # Dropout on top of the real profiles. Seeded on the index so an item
        # keeps its condition across epochs and ranks, rather than flickering
        # between sensed and blind and averaging the two away.
        if present is not None and self.radar_dropout:
            if random.Random(self.seed * 1_000_003 + i).random() < self.radar_dropout:
                present = None
        if present is None:
            points, mask = torch.zeros_like(points), torch.zeros_like(mask)

        sensors = "camera, ego motion"
        sensors += f", {SENSOR_LABEL[present]}" if present else ", no radar"

        target = item["target"]
        if present is None and item["task"] in RADAR_ONLY_TASKS:
            target = NO_RADAR_ANSWER

        if instant:
            # One sample, at the instant asked about. The windowed ego state is
            # twenty readings over one second and they barely differ -- measured,
            # 0.06 to 0.12 m/s of spread against 4.3 m/s across the whole clip --
            # so the other nineteen repeat the same fact for about 200 tokens.
            # What the question needs from the ego is the speed that the Doppler
            # residual is computed against, and that is a single number.
            ego = ego_text(radar["ego_state"][-1:], 0)
            ego_header = f"Ego motion (binned, at t={item['frame']}s)"
        elif window:
            # One reading per second, not one per radar scan. The radar grid is
            # 10 Hz for a two-second window, and twenty ego readings a tenth of
            # a second apart say the same thing twenty times: measured across a
            # one-second window the spread is 0.06-0.12 m/s against 4.3 m/s
            # across a clip. Subsampling to 1 Hz keeps the trend the prediction
            # actually needs and drops about 170 tokens.
            state = radar["ego_state"]
            step = max(1, int(round(radar_hz)))
            keep = list(range(len(state) - 1, -1, -step))[::-1]
            offsets = [(i - (len(state) - 1)) / float(radar_hz) for i in keep]
            ego = ego_text(state[keep], len(keep) - 1, times=offsets)
            ego_header = (f"Ego motion (binned, 1 Hz, offsets in seconds from "
                          f"t={item['frame']}s)")
        else:
            # Ego motion stops at the frame in question; see the module docstring.
            ego = ego_text(radar["ego_state"], item["frame"],
                           times=[i - item["frame"]
                                  for i in range(item["frame"] + 1)])
            ego_header = ("Ego motion (binned, 1 Hz, offsets in seconds from "
                          f"t={item['frame']}s)")

        user = (f"{radar_prompt_block(self.radar_tokens)}\n"
                f"Sensors present: {sensors}.\n"
                f"{ego_header}: {ego}\n"
                f"{item['prompt']}")
        return {"clip_id": clip_id, "task": item["task"],
                "frames": frames, "user": user, "target": target,
                "points": points, "mask": mask,
                # Routed experts key off this, so it has to follow the dropout:
                # a blanked radar is `none`, not the sensor the rig carries.
                "sensor": torch.tensor(SENSOR_IDS[present] if present
                                       else SENSOR_IDS["none"])}


def build_collate(processor, tokenizer, max_length=3584):
    """Chat-template the batch and mask the loss to the answer span."""

    def collate(batch):
        texts, videos = [], []
        for sample in batch:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "video"},
                                             {"type": "text", "text": sample["user"]}]},
                {"role": "assistant", "content": [{"type": "text",
                                                   "text": sample["target"]}]},
            ]
            texts.append(processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False))
            videos.append(sample["frames"])

        # Without metadata the processor assumes fps=24, so it would read the 20
        # keyframes as 0.83 s of footage instead of 20 s. Qwen3-VL feeds
        # timestamps into its positional encoding, so that error would misplace
        # every temporal claim the text makes -- "at frame 12" would refer to
        # half a second in.
        metadata = [VideoMetadata(total_num_frames=len(v), fps=1.0,
                                  width=v[0].width, height=v[0].height,
                                  duration=float(len(v)),
                                  video_backend="manual",
                                  frames_indices=list(range(len(v))))
                    for v in videos]
        encoded = processor(text=texts, videos=videos, video_metadata=metadata,
                           return_tensors="pt", padding=True, truncation=True,
                           max_length=max_length)

        # Supervise the assistant turn only. Everything before the final
        # generation header is context, and training on it would teach the model
        # to reproduce prompts.
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        header = tokenizer("<|im_start|>assistant\n",
                           add_special_tokens=False)["input_ids"]
        for b in range(labels.shape[0]):
            ids = encoded["input_ids"][b].tolist()
            start = _find_last(ids, header)
            if start is None:
                labels[b, :] = -100
            else:
                labels[b, :start + len(header)] = -100
        encoded["labels"] = labels
        encoded["points"] = torch.stack([s["points"] for s in batch])
        encoded["radar_mask"] = torch.stack([s["mask"] for s in batch])
        encoded["sensor"] = torch.stack([s["sensor"] for s in batch])
        encoded["task"] = [s["task"] for s in batch]
        # Carried so a saved generation can be traced to the clip it describes.
        encoded["clip_id"] = [s.get("clip_id") for s in batch]
        return encoded

    return collate


def _find_last(haystack, needle):
    for start in range(len(haystack) - len(needle), -1, -1):
        if haystack[start:start + len(needle)] == needle:
            return start
    return None
