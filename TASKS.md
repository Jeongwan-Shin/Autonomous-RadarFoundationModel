# Tasks

One model, one input, one instruction slot. Video, radar and ego motion are the
same tensors for every row below; the text instruction is what selects the task,
and the answer is always generated text — no task has an output head.

Every example here is a real row from the built data, not a mock-up.

---

## What a task is made of

| part | meaning |
|---|---|
| **instruction** | the text that selects the task. Prefixed with `At frame N.` where the answer is about one instant |
| **rationale** | for `_cot` variants: the evidence, stated before the answer. Always a quantity computed from the labels, so a checker can verify it |
| **answer** | what is scored |
| **reward** | the RLVR reward, derived from the same scorer the task is evaluated with (`training/task_scorers.py`) |

`_cot` rows are emitted as `{"rationale": ..., "answer": ...}` and rewarded as
`0.7 x answer + 0.3 x rationale`. Format alone earns nothing: an empty rationale
inside valid JSON scores zero on both halves.

---

## 01 · det_objects — detection as an enumeration

> **Instruction** · At frame 6. List every road user in the forward sector with its class, range and azimuth.
>
> **Answer** · `automobile 39 m az -40 deg stationary; automobile 61 m az +47 deg stationary; automobile 103 m az +17 deg stationary`

**Reward** `reward_objects` — detection F1 by greedy nearest-first matching at
2 m, then half the credit split across class, motion and range accuracy on the
objects that matched. F1 alone is satisfied by a list of plausible objects at
invented ranges, which is why the split exists.

**`det_objects_cot` rationale** · `camera only: automobile at 39 m; automobile at 61 m; automobile at 103 m.`

The radar illuminates 85.7% of heavy trucks and 20.2% of people, so which
objects the sensor actually supports is a real distinction and is upstream of
the list. Frames where the radar confirms nothing still say so, because
"the camera is alone here" is itself the evidence.

---

## 02 · track_identity — the same objects, keyed and aged

> **Instruction** · At frame 6. Give the track id, class, range and age of every object you are tracking ahead.
>
> **Answer** · `#75 automobile 39 m visible 0.0 s; #41 automobile 61 m visible 5.0 s; #59 automobile 103 m visible 3.3 s`

**Reward** `reward_objects`, same as 01. This is identity persistence, not
re-detection: the same object must keep its id across frames.

**`track_identity_cot` rationale** · `#75 first seen at t=5.0 s, now t=5 s; #41 first seen at t=0.0 s, now t=5 s; #59 first seen at t=1.7 s, now t=5 s. Age is the difference between the two.`

The cleanest chain in the set — the answer is literally the subtraction the
rationale sets up.

---

## 03-1 · plan_ego — where the ego goes

> **Instruction** · At frame 6. Predict the ego vehicle's path over the next 3 seconds as (x, y) offsets in metres.
>
> **Answer** · `+1s (+14.5, +0.1); +2s (+29.2, +0.4); +3s (+44.0, +0.5)`

**Reward** `reward_waypoints` — horizon coverage times a displacement decay,
half credit at 1 m. Ego waypoints live within a couple of metres over 3 s, so
the tolerance is set to that scale.

**`plan_ego_cot` rationale** · `The ego vehicle is travelling at 14.4 m/s and will hold speed and go straight. At that speed it covers about 14 m per second.`

No radar here. Where the ego goes is a function of its own speed and yaw rate,
both measured by the egomotion stream, and 14.4 x 3 = 43 recovers the answer's
forward offset. This is the one task whose evidence is not a radar reading, and
it is kept precisely for that contrast.

---

## 03-2 · agent_traj — where one other agent goes

> **Instruction** · At frame 6. Track #68 is a automobile at 41 m, azimuth +6 deg. Where will it be over the next 3 seconds?
>
> **Answer** · `+1s 31 m az +14 deg; +2s 23 m az +38 deg`

**Reward** `reward_trajectory` — range decay at 5 m and bearing decay at 10 deg,
averaged, scaled by how many horizons were answered.

**`agent_traj_cot` rationale** · `The radar puts 13 returns on track #68 at 41 m, median radial velocity -10.7 m/s, so it is closing.`

The strongest chain in the set: radial velocity times time *is* the range
change. 41 m closing at 10.7 m/s reaches 31 m after a second, which is the
answer. Nothing but the radar supplies that velocity.

---

## 04 · world_model — the scene 3 s on, conditioned on the ego action

> **Instruction** · At frame 6. The ego vehicle will hold speed and go straight over the next 3 seconds. What will the forward scene look like then?
>
> **Answer** · `2 automobiles; 0 moving; nearest automobile at 62 m`

**Reward** `reward_quantity` — every number in the answer, by relative error.

**`world_model_cot` rationale** · `Now: automobile at 39 m; automobile at 61 m; automobile at 103 m. The ego covers about 43 m in 3 s, so ranges change by that much plus each object's own motion.`

Check the arithmetic: the objects at 39 m and 61 m fall behind the ego's 43 m of
travel and leave the sector; 103 - 43 = 60, and the answer says the nearest is
then at 62 m. Two objects remain. The rationale determines the answer.

---

## 05 · depth_range — nearest, and nearest the radar confirms

> **Instruction** · At frame 6. How far is the nearest object ahead, and the nearest one the radar confirms?
>
> **Answer** · `nearest automobile at 16 m; nearest radar-confirmed automobile at 16 m`

**Reward** `reward_quantity`.

**`depth_range_cot` rationale** · `automobile at 16 m: 9 returns; automobile at 20 m: 9 returns; person at 20 m: 1 returns.`

The pairing is the point: the camera ranges the nearest thing, the radar
confirms a possibly different one, and only the return counts separate the two
claims. Frames where no object carries a return emit **no** CoT row — there the
rationale could only restate the answer's second clause, which teaches padding
rather than reasoning.

---

## 06 · motion_seg — moving vs stationary, by Doppler

> **Instruction** · At frame 6. Which objects ahead are moving and which are stationary? Use the radar Doppler.
>
> **Answer** · `moving: automobile 41 m az +6 deg (13 radar returns). stationary: heavy_truck 5 m az -41 deg (18 radar returns), ...`

**Reward** `reward_objects` — the same matcher as 01, so a right verdict on the
wrong object earns nothing.

**`motion_seg_cot` rationale** · `heavy_truck at 5 m: 18 returns, 0 of them still moving once the ego's own motion is removed; ... An object whose returns keep a residual above 1 m/s is moving; one whose returns are explained entirely by the ego's own motion is not.`

This rationale was wrong until it was fixed and the fix matters. It used to cite
the **measured** radial velocity — "heavy_truck at 5 m, mean radial -5.1 m/s" —
next to a rule about the **residual** left after the ego's own motion is
removed. A stationary object's measured radial is just the ego's speed projected
onto it, so applying the stated rule to the stated number gives "moving" where
the answer says "stationary". A model following that reasoning answers wrongly.
It now cites the ego-compensated count, which is what the verdict actually rests
on.

---

## 09 · retrieval — scenario tags

> **Instruction** · Write the scenario tags that should retrieve this clip.
>
> **Answer** · `daytime; ego 16 m/s; 4 objects ahead, 4 moving; nearest rider`

**Reward** `reward_tags` — tag-set F1.

**No CoT, deliberately.** The answer is already a list of measured facts, so a
rationale would restate it word for word. A rationale is worth training only
where it sits upstream of the answer.

---

## 10 · qa — five-option multiple choice

> **Instruction** · At Frame 20, what is the pedestrian on the right's lateral position relative to the ego's current path?
> A. Far to the left. / B. Slightly to the left. / C. Directly in front. / D. Slightly to the right. / E. Far to the right.
>
> **Answer** · `E`

**Reward** `reward_choice` — exact letter.

**`qa_cot` rationale** · `At Frame 20, the ego vehicle is at pos=(211.98,61.37) and the pedestrian on the right is at pos=(223.86,57.78). The pedestrian's y-coordinate (57.78) is significantly less than the ego's y-coordinate (61.37), indicating a position to the right ...`

The QA release ships a worked rationale with every one of its 39,158 questions
and it was going unused while the model was trained on the letter alone. Only
the **14,252** carrying an `agrees` verification verdict are emitted; the other
24,906 were never checked, and training a chain of reasoning on unverified
arithmetic teaches unverified arithmetic.

Two scoring bugs lived here until recently, both of which fed the reward as well
as the report:

- The options run **A–E**, and the scorer matched only `ABCD`. Every question
  whose answer was E — **411 of 1,940 test questions, 21.2%** — was marked wrong
  no matter what the model wrote, capping the task at 78.8%.
- Scanning for the first character in `ABCDE` also matched the **A** inside
  `"Answer:"`, so a model that prefixed its choice was graded on the prefix.

---

## 11 · description — the scene in sentences

Six kinds, all generated from measured quantities:

| kind | example answer |
|---|---|
| `desc_objects` | `Ahead: 2 cars. The closest is a car at 20 m. All 2 are moving.` |
| `desc_radar` | `the short-range radar returns 244 detections, 170 of which are not explained by the ego's own motion` |
| `desc_ego_maneuver` | `The ego vehicle is travelling at 29.7 m/s (107 km/h), holding a steady course.` |
| `desc_complementarity` | `None of the 2 labelled objects in the forward sector produces a radar return` |
| `desc_clip_summary` | `Over 20 s the ego vehicle covers 596 m, with speed between 29.5 and 30.9 m/s, brakes hard in 3 of them.` |

**Reward** `reward_description` — half the numbers by relative error, half the
claim words by set F1, minus 0.35 for each direct contradiction
(`all`/`none`, `moving`/`stationary`, `closing`/`receding`,
`accelerating`/`braking`, `left`/`right`).

These were once excluded from RLVR on the grounds that free text has no
checkable answer. That was wrong: every description states its measurements
outright, and grading the claims while ignoring the prose is a verifiable
reward, not a preference model. Numbers alone would let a model recite the right
counts in a sentence that says the opposite — "None are moving" and "All 2 are
moving" share their numbers — which is what the contradiction term is for.

| generated | reward |
|---|---|
| exact | 1.000 |
| `None are moving` for `All 2 are moving` | 0.358 |
| right words, wrong numbers | 0.500 |
| both wrong | 0.000 |

**No CoT**, for the same reason as 09.

---

## Not trained

| task | why |
|---|---|
| **07 radar_adaptation** (`radar_transfer`) | dropped from training, kept as an evaluation-only instrument. Having never been trained makes it a cleaner generalisation test |
| **08 missing_modality** | never a separate task. 7% of clips carry no front radar, and the loader routes them by sensor profile, replacing radar-dependent answers with `Radar unavailable, so this cannot be determined.` Training the original answer against a blank radar is exactly how a model learns to recite radar statistics it never read |

## Diagnostic tasks, added here, not in the original eleven

| task | purpose |
|---|---|
| `radar_probe` | asks the radar's scan statistics directly. Intended as a plumbing check; later found circular, because two of its three question forms ask for the exact scalars `head_stats` trains the encoder to emit |
| `radar_structure` | five scan-level questions no supervised statistic answers. **Evaluation only** |
| `radar_objects` | per-object radar geometry — the granularity the perception tasks actually depend on, by a factor of 10-12 in explained variance. Trainable, but its forms are split so some stay instruments |

Probes that get trained stop being instruments. That is the lesson `radar_probe`
paid for, and it is why `radar_structure` is absent from `ALL_TRAIN_TASKS` and
why `radar_objects` is trained on a subset of its forms.
