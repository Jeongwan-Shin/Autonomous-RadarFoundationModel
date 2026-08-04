# Autonomous Radar Foundation Model

A radar + vision + text foundation model. Video, radar and ego motion are a
fixed input; **only the text instruction changes** to select among the tasks.
Nothing has a task-specific output head — every answer is generated as text.

Data stays out of this tree: everything here reads the read-only raw datasets
and writes indices into the split directory.

| | path |
|---|---|
| code (this repo) | `/NHNHOME/workspace/AutonomousRadarFoundationModel` |
| raw data | `/NHNHOME/workspace/dataset/raw_Auto_datasets/{Nvidia_AUTO,nuScenes}` |
| task splits | `/NHNHOME/workspace/dataset/raw_Auto_datasets/preprocessed_train_test_split` |
| checkpoints | `/NHNHOME/workspace/checkpoints` |

Both dataset roots are overridable with `RAVILA_RAW_ROOT` and
`RAVILA_SPLIT_ROOT`; `datatools/paths.py` is the only module that hardcodes
anything. Checkpoint and model paths are hardcoded in `training/train_vlm.py`
and would need editing to run elsewhere.

## Layout

```
datatools/                    data preparation, all offline
  paths.py                    dataset roots, sensor names, front-FOV constants
  geometry.py                 sensor <-> rig <-> world transforms, ego velocity,
                              Doppler residual, FOV mask, box containment
  scene_features.py           per-frame scalar features for every clip
  frame_objects.py            per-object text targets for tasks 01-06, plus the
                              chain-of-thought variants, in one archive pass
  radar_structure.py          evaluation-only probes the encoder is NOT trained
                              on -- see "The instrument problem" below
  make_qa_holdout.py          sha1-stable QA test split (99 clips / 1,940 items)
  build_report.py             the first report, as a PDF
  build_report_v2.py          the training/RLVR report, every number read from
                              artefacts on disk rather than typed in

training/
  radar_encoder.py            point-cloud transformer; 240 tokens per clip
  connector.py                encoder tokens -> language-model embedding space
  instruct_data.py            the multi-task loader; per-clip sensor routing,
                              radar dropout, holdout handling
  train_vlm.py                align / joint / full(FSDP2) supervised training
  train_grpo.py               GRPO with verifiable rewards
  task_scorers.py             one scorer per task, and the reward derived from
                              it -- they do not share a metric
  select_seed.py              choose a seed on val, report it on test
  rescore_generations.py      re-derive metrics from saved generations, no GPU
  eval_all_tasks.py           generate, score, and compare against a
                              shuffled-radar control
  probe_pipeline.py           linear probes at four points in the stack
```

## How the model reads radar

The language model never sees raw points. `radar_encoder.py` compresses each
scan into 240 tokens, `connector.py` maps them into the embedding space, and a
forward hook writes them into the sequence. The hook is not a stylistic choice:
passing `inputs_embeds` disables Qwen3-VL's video-token scatter, so the video
would silently stop reaching the model.

`RadarEncoder` is trained with an auxiliary head, `head_stats`, that predicts
five scalars from the emitted tokens (`log_n_points`, `log_n_moving`,
`max_rcs`, `mean_rcs`, `max_range`). This is load-bearing — without it the
readout and temporal stack receive no gradient at all — and it is also the
source of the measurement problem below.

## The instrument problem

The headline metric for most of this project was **radar contribution**: the
correlation of the generated number with the truth, minus the same model fed
another clip's radar. The subtraction matters, because the camera alone
predicts much of every quantity here.

`radar_probe` reported +0.54, which read as "the model reads the radar". Split
by question form, it does not say that:

| form | quantity asked | supervised by `head_stats`? | contribution |
|---|---|---|---|
| detections | `n_points`, `n_moving` | yes | +0.84 |
| rcs | `max_rcs` | yes | +0.80 |
| illuminated | returns associated to boxes | **no** | **+0.00** |

Two of the three forms ask for exactly the scalars the encoder was trained to
write into its tokens. The measurement was circular. `radar_structure.py` exists
to fix this: five questions no supervised statistic answers, each with its
contamination — the largest correlation between its answer and any supervised
scalar — measured rather than assumed.

| probe | question | contamination | contribution |
|---|---|---|---|
| bearing | azimuth of the nearest return | 0.035 | +0.07 |
| lateral | share of returns left of centre | 0.123 | +0.05 |
| spread | angular span of the scan | 0.458 | +0.08 |
| closing | speed of the fastest approaching return | 0.562 | +0.17 |
| near_far | share of returns within 20 m | 0.690 | +0.21 |

The two clean probes read zero. Scalar summaries of the radar reach the
language model; angular and spatial structure does not. Contribution rises with
contamination, which is what that explanation predicts.

These probes are deliberately absent from `ALL_TRAIN_TASKS`. Training on them
would turn them into one more memorised skeleton and destroy the only honest
instrument in the repo.

## Facts the code encodes

These were measured, not assumed, and each one changed a design decision.

**Radar poses live only in the non-offline calibration.**
`calibration/sensor_extrinsics.offline/` has 7 cameras and the lidar, no radar.
The non-offline variant adds 8 radar entries. With it, box centres sit a median
**1.31 m** from their nearest radar return; the best guessed convention reached
only 10.6 m. `fetch_radar_extrinsics.py` exists solely to get this.

**Motion must be judged in the world frame.** `obstacle.offline` boxes use
`reference_frame='rig'`, which is ego-centric, so a parked car sweeps past while
a car matching the ego speed looks frozen. World- and rig-frame displacement
correlate at **0.202**. Measured on 14,844 tracks: **71.6% are stationary** in
the world frame versus **0.4%** by the rig measure. The first version of the
agent-trajectory filter used rig displacement and therefore selected almost
exactly the wrong tracks.

**Labels cover 360 degrees; this download sees a forward sector.** Both the
front wide camera and the imaging LRR span about +/-60 deg, so only **~46%** of
labels are observable. Only **20%** of tracks stay inside that sector for most
of their life. Training against out-of-sector labels penalises the model for
physics and teaches a language model to invent objects behind the car, so
`front_fov_mask()` is applied before labels are used.

**Doppler prunes 95% of returns without any label.** A stationary return's
radial velocity is the negative projection of ego velocity onto the
sensor-to-target ray; the residual isolates real motion. On imaging LRR a
1.0 m/s threshold keeps **47 of 954** returns per scan. Useful three ways at
once: token pruning, free supervision, and the moving/static task itself.

**Radar grounds vehicles, not pedestrians.** Fraction of in-FOV boxes
containing at least one return: `heavy_truck` 85.7%, `automobile` 67.1%,
`rider` 29.0%, `person` **20.2%**, `protruding_object` **0.0%**. Any
text-to-radar grounding supervision built from boxes is only trustworthy for
vehicle classes.

**Radar is a detection list, not a tensor.** No range-azimuth or range-Doppler
heatmap is available. Scan rate and density differ per sensor: SRR 13.4 Hz /
313 returns, MRR 20 Hz / 545, imaging LRR 20 Hz / 989.

**No clip carries all three front radars.** `radar_config` is a property of the
vehicle rig: `low` has the front SRR, `med`/`high` have MRR + imaging LRR, and
they never co-occur. So SRR -> LRR translation has no paired supervision; the
only genuine simultaneous pair is MRR <-> LRR. `radar_transfer` is built on
that one pair.

**Nvidia boxes are autolabels** (`source='scene:obstacles:autolabels:v2'`). The
only human-verified ground truth is the OOD reasoning subset: 2,077 events over
1,198 downloaded clips. `egomotion` is a sensor measurement, which is why the
ego-planning and Doppler tasks carry the most trustworthy targets.

## Evaluation

Tasks do not share a metric, so `task_scorers.py` holds one scorer each and
`eval_all_tasks.py` runs generation, scoring, and the shuffled-radar control
together. Four things in there are worth knowing because each one was a bug
that made results look better than they were:

- **Per-question-form correlation.** Pooling questions whose answers differ in
  magnitude by 100x makes the correlation detect *which question was asked*.
  Pooled, `radar_probe` read 0.994; split by form, 0.724.
- **The shuffled control has to actually shuffle.** At batch size 1,
  `roll(shifts=1, dims=0)` points at the item itself, so "full" and "shuffled"
  were identical to four decimals.
- **Range-only tasks need their own threshold.** Task 02 has no azimuth, so
  matching it on `(x, y)` at 2 m was meaningless.
- **Ridge probes need their penalty tuned.** A fixed penalty reported R^2 -0.5
  for a quantity a properly tuned probe put at +0.67.

## Training

```bash
# supervised, 5 GPUs, no LoRA
torchrun --nproc_per_node=5 -m training.train_vlm \
    --model 8B --stage full --tasks all --all-profiles \
    --radar-checkpoint .../encoder.pt --samples 300000 --out .../vlm_8B_long_base

# GRPO with verifiable rewards, one GPU per arm
torchrun --nproc_per_node=1 -m training.train_grpo \
    --init .../vlm_8B_long_base --task radar_probe,radar_transfer,depth_range \
    --reward all_numbers --steps 4000 --out .../grpo_multi

# evaluate every task, with the radar control
python -m training.eval_all_tasks --checkpoint .../grpo_multi \
    --tasks all --split test --items 500
```

GRPO here is RLVR, not RLHF: the answers were computed from the labels, so a
program checks them and no preference model is involved. Group-relative
normalisation replaces the value network, which would otherwise be a second 8 B
model in memory, and the clipped ratio against the sampling policy removes the
need for a frozen reference copy.

### One reward per task

`--reward per_task` derives each item's reward from the scorer its task is
*evaluated* with, in `task_scorers.py`. This is what lets the whole instruction
set be trained rather than the numeric corner of it. The earlier rewards all
read numbers out of the answer text, so only the three tasks whose answer is a
number could be trained at all; on `det_objects`, whose answer is a list of
objects with a class, a range and a bearing, such a reward grades the first
integer in the list, which is not the task.

| task | reward | built from |
|---|---|---|
| `det_objects`, `track_identity`, `motion_seg` | `reward_objects` | detection F1, then class / motion / range accuracy on the matched objects |
| `plan_ego` | `reward_waypoints` | horizon coverage x displacement decay, half credit at 1 m |
| `agent_traj` | `reward_trajectory` | range decay (5 m) and bearing decay (10 deg), averaged |
| `radar_probe`, `radar_transfer`, `depth_range`, `world_model`, `radar_objects` | `reward_quantity` | every number in the answer, by relative error |
| `retrieval` | `reward_tags` | tag-set F1 |
| `qa` | `reward_choice` | exact letter |
| `desc_*` | **none** | free text -- any reward invented for it is a preference model, which is the thing RLVR exists to avoid |

Deriving the reward from the scorer keeps the two from drifting: an answer that
scores well is by construction an answer the reward paid for. `reward_objects`
splits its credit half on F1 and half on the details of the matched objects,
because F1 alone is satisfied by a list of plausible objects at invented ranges.
A task with no checkable answer returns `None` and the trainer refuses to start
rather than optimise a number someone made up.

### Choosing a seed

Seeds land in different places -- two GRPO runs on identical settings went
0.378 -> 0.725 -> 0.391 and 0.330 -> 0.366 -> 0.549 -- and for a model you
intend to ship, taking the best of them is the right call. What is not is taking
the best of them *on the test set*: selecting the maximum of N draws shifts the
estimate up by about the spread between seeds, measured here at 0.028, which is
the size of the effects being argued about. `select_seed.py` separates the two
steps, choosing on `val` and reporting the single winner on `test`. It selects
on radar contribution rather than raw correlation, since a checkpoint can top a
`full`-correlation table by getting better at guessing from the camera.

**What GRPO did and did not do.** Measured against a matched supervised control
at the same scale, it prevented catastrophic forgetting — continued SFT
collapsed `radar_transfer` from 0.838 to 0.166 while every GRPO arm held ~0.85.
It did **not** raise radar contribution; the +0.55 that GRPO runs report was
already present in the checkpoint they started from. Across three seeds the four
reward shapes are indistinguishable: the spread between rewards is 0.026 against
a seed standard deviation of 0.028.

## Order of operations

```bash
# one-off: radar poses (~90 MB, rate-limited, resumable)
python3 -m datatools.fetch_radar_extrinsics --verify

# world-frame track statistics (~80 min over 1,838 archives)
python3 -m datatools.rescan_tracks --workers 14

# rebuild the splits that depended on the old rig-frame filter
python3 -m datatools.fix_track_splits

# text targets for tasks 01-06, and the evaluation-only structure probes
python3 -m datatools.frame_objects --clips all --workers 14
python3 -m datatools.radar_structure --workers 14
```

## Open questions

- **Is the angular information missing from the encoder, or unreadable by the
  language model?** The probes above show the model cannot *write* an azimuth.
  They do not distinguish "the encoder discarded it" from "the language model
  cannot read it". Running `probe_pipeline.py` against azimuth rather than
  `max_rcs` separates the two, and that determines where the next intervention
  belongs.
- If the encoder is at fault, adding an angular term to `head_stats` is the
  obvious candidate — but then `radar_structure` must leave the evaluation set,
  or it becomes circular in exactly the way `radar_probe` did. Some probes would
  have to be supervised and the rest kept back as instruments.
