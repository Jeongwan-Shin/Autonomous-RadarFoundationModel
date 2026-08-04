#!/usr/bin/env python3
"""Generate notebooks/inspect_data.ipynb.

The notebook is generated rather than hand-edited so it stays in step with the
code it inspects, and so a broken cell can be fixed at the source instead of
inside JSON.

`source` is written as a single string, not a list of lines. nbformat accepts
both, but a list whose entries lack their trailing newline gets concatenated
into one unparseable line on the next write -- which is exactly what happened
the first time this notebook was built.

    python notebooks/build_inspect_notebook.py
"""

import json
import os
import sys

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "id": f"md{len(CELLS):02d}",
                  "metadata": {}, "source": text.strip("\n")})


def code(text):
    CELLS.append({"cell_type": "code", "id": f"code{len(CELLS):02d}",
                  "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.strip("\n")})


md("""
# 데이터 점검

빌드된 instruction 데이터를 눈으로 확인하는 노트북입니다. 학습 전에

- 태스크별로 몇 개가 있는지
- 프롬프트와 정답이 실제로 어떻게 생겼는지
- CoT의 rationale이 answer를 실제로 뒷받침하는지
- `det_objects`가 받는 1프레임 비전 + 1초 레이더 창이 의도대로인지

를 확인합니다. GPU가 필요 없습니다.
""")

code("""
import os, sys, json, re, textwrap

sys.path.insert(0, os.path.dirname(os.getcwd()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datatools import paths

COMMON = paths.COMMON_DIR
ITEMS = os.path.join(COMMON, "instruct_items_tasks01_06.parquet")
pd.set_option("display.width", 170)
pd.set_option("display.max_colwidth", 90)

print("split root :", paths.SPLIT_ROOT)
print("common     :", COMMON)
""")

md("## 1. 어떤 파일이 있고 얼마나 큰가")

code("""
files = ["instruct_items_tasks01_06.parquet", "scene_features_all_clips.parquet",
         "radar_object_probes.parquet", "radar_structure_probes.parquet",
         "nvidia_clips.parquet", "qa_holdout_clips.json"]
rows = []
for f in files:
    p = os.path.join(COMMON, f)
    rows.append({"file": f, "exists": os.path.exists(p),
                 "MB": round(os.path.getsize(p) / 1e6, 1) if os.path.exists(p) else None})
pd.DataFrame(rows)
""")

md("""
## 2. 태스크별 아이템 수

태스크 01~06과 그 CoT 변형이 한 파일에 들어 있고 `task` 컬럼으로 나뉩니다.
""")

code("""
items = pd.read_parquet(ITEMS, columns=["clip_id", "task", "frame", "split"])
print(f"{len(items):,} rows, {items.clip_id.nunique():,} clips")

(items.pivot_table(index="task", columns="split", values="clip_id",
                   aggfunc="count", fill_value=0)
      .assign(total=lambda d: d.sum(axis=1))
      .sort_values("total", ascending=False))
""")

md("""
## 3. 앵커 프레임

각 태스크가 클립의 어느 시점에서 만들어졌는지입니다. `frame`은 1-indexed이고 t = frame − 1초.

- `det_objects`는 **3/6/9/12/15/18초** — 미래를 묻지 않으므로 3초 지평선 제약이 없습니다
- 나머지는 **5/10/15초** — 3초 예측이 클립 안에 들어가야 합니다
""")

code("""
anchors = (items.groupby("task")["frame"].agg(lambda s: sorted(s.unique()))
                .to_frame("frames"))
anchors["seconds"] = anchors.frames.map(lambda fs: [f - 1 for f in fs])
anchors["per_clip"] = (items.groupby("task").size()
                       / items.groupby("task")["clip_id"].nunique()).round(2)
anchors
""")

md("## 4. 프롬프트와 정답 살펴보기\n\n`TASK`를 바꿔 가며 확인하세요.")

code("""
TASK = "det_objects"
N = 3

full = pd.read_parquet(ITEMS)
for _, r in full[full.task == TASK].head(N).iterrows():
    print("=" * 100)
    print(f"clip {r.clip_id}   frame {r.frame} (t={r.frame - 1}s)   split {r.split}")
    print("PROMPT :", textwrap.fill(r.prompt, 96, subsequent_indent=" " * 9))
    print("TARGET :", textwrap.fill(r.target, 96, subsequent_indent=" " * 9))
""")

md("""
## 5. CoT: rationale이 answer를 뒷받침하는가

`_cot` 태스크는 `{"rationale": ..., "answer": ...}` 형식입니다. rationale은 감상이 아니라
라벨에서 계산된 증거여야 하고, 그것이 answer를 결정해야 합니다. 근거를 따라갔을 때
답이 나오지 않으면 그 사슬은 잘못된 것이고, 보상을 걸면 모델이 그 잘못된 사슬을 배웁니다.
""")

code("""
cot = sorted(t for t in full.task.unique() if t.endswith("_cot"))
print("CoT 태스크:", cot)

for task in cot:
    r = full[full.task == task].iloc[0]
    try:
        d = json.loads(r.target)
    except json.JSONDecodeError:
        print(f"{task}: JSON 아님")
        continue
    print("\\n" + "=" * 100)
    print(f"### {task}")
    print("Q :", textwrap.fill(r.prompt, 96, subsequent_indent=" " * 4))
    print("R :", textwrap.fill(d["rationale"], 96, subsequent_indent=" " * 4))
    print("A :", textwrap.fill(d["answer"], 96, subsequent_indent=" " * 4))
""")

md("""
## 6. det_objects의 입력

`det_objects`만 입력이 다릅니다.

| | 일반 태스크 | det_objects |
|---|---|---|
| 비전 | 20프레임 (1 Hz × 20 s) | **1프레임** (질문한 순간) |
| 레이더 | 20스캔 (1 Hz × 20 s) | **20스캔** (질문 직전 약 1 s, 센서 원래 속도) |

레이더 원본은 LRR·MRR이 20 Hz라 20초 클립에 약 400스캔이 있는데, 기본 모드는 초당
1개만 남기고 95%를 버립니다. 한 순간을 묻는 질문에는 그 반대가 맞습니다.
""")

code("""
from datatools.frame_objects import read_member

clips = pd.read_parquet(os.path.join(COMMON, "nvidia_clips.parquet"))
lrr = clips[clips.has_lrr1.fillna(False) & clips.has_radar_extrinsics.fillna(False)]
row = lrr.loc[lrr.index[0]]

radar = read_member(paths.NVIDIA_ROOT, row.radar_lrr1_zip, row.radar_lrr1_member)
scans = np.sort(radar.timestamp.unique()) / 1e6
period = float(np.median(np.diff(scans)))
print(f"클립 전체 스캔 {len(scans)}개, 주기 {period:.3f} s ({1 / period:.1f} Hz)")
print()

for until in (3, 6, 9, 12, 15, 18):
    w = scans[scans <= until][-20:]
    print(f"  t={until:>2}s 창 : {len(w):>2}개  {w.min():6.3f}..{w.max():6.3f}"
          f"  span {w.max() - w.min():.3f} s")

default = np.array([scans[np.argmin(np.abs(scans - f))] for f in range(20)])
print(f"\\n  기본 모드  : {len(default)}개  {default.min():6.3f}..{default.max():6.3f}"
      f"  span {default.max() - default.min():.3f} s")
""")

md("""
## 7. 레이더 스캔 한 장 (BEV)

자차가 원점, 위쪽이 전방입니다. 색은 도플러 잔차 — 자차 운동을 제거하고도 남은 속도이고,
1 m/s를 넘으면 실제로 움직이는 것으로 판정합니다. 빨간 점선이 라벨이 사용하는 ±60° 섹터입니다.
""")

code("""
from datatools.frame_objects import ego_frame, radar_scan, SENSOR_NAME

clip_id = lrr.index[0]
ego = read_member(paths.NVIDIA_ROOT, row.egomotion_zip, row.egomotion_member)
derived = ego_frame(ego)
ext = pd.read_parquet(os.path.join(paths.NVIDIA_ROOT, row.radar_extrinsics_parquet))
scan = radar_scan(radar, ext.loc[clip_id], SENSOR_NAME["lrr1"], 12.0, ego, derived)

rig, residual, moving = scan["rig"], scan["residual"], scan["moving"]
fig, ax = plt.subplots(figsize=(7, 7))
sc = ax.scatter(rig[:, 1], rig[:, 0], c=np.abs(residual), s=6, cmap="viridis",
                vmin=0, vmax=8)
ax.scatter(0, 0, marker="^", s=180, c="red", label="ego")
for deg in (-60, 60):
    a = np.radians(deg)
    ax.plot([0, 200 * np.sin(a)], [0, 200 * np.cos(a)], "r--", lw=0.8, alpha=0.5)
ax.set_xlabel("left (m)")
ax.set_ylabel("forward (m)")
ax.set_title(f"imaging LRR @ t=12 s - {len(rig)} returns, {int(moving.sum())} moving")
ax.set_xlim(-120, 120)
ax.set_ylim(-10, 200)
ax.set_aspect("equal")
ax.grid(alpha=0.25)
ax.legend(loc="upper right")
plt.colorbar(sc, ax=ax, label="|Doppler residual| (m/s)")
plt.tight_layout()
plt.show()
""")

md("## 8. 비디오 프레임\n\n`det_objects`가 실제로 받는 한 장과, 일반 태스크가 받는 20장의 일부입니다.")

code("""
from training.video_frames import clip_frames, pad_frames

frames = pad_frames(clip_frames(paths.NVIDIA_ROOT, row))
SECONDS = 12

fig, axes = plt.subplots(2, 1, figsize=(13, 6))
axes[0].imshow(frames[SECONDS])
axes[0].set_title(f"det_objects 입력 - t={SECONDS}s, 한 장")
axes[1].imshow(np.concatenate([np.asarray(f) for f in frames[:6]], axis=1))
axes[1].set_title("일반 태스크 입력 - 20장 중 앞 6장 (1 Hz)")
for a in axes:
    a.axis("off")
plt.tight_layout()
plt.show()
""")

md("""
## 9. 보상 함수를 실제 아이템에 적용

RLVR 보상은 평가 채점기에서 유도했습니다. 정답을 그대로 넣으면 1.0이 나와야 하고,
숫자를 망가뜨리면 떨어져야 합니다. 새 태스크를 추가했다면 여기서 먼저 확인하세요.
""")

code("""
from training.task_scorers import reward_for

def double_numbers(text):
    \"\"\"형식은 그대로 두고 숫자만 2배로 -- 내용만 틀린 답을 만든다.\"\"\"
    return re.sub(r"\\d+(?:\\.\\d+)?",
                  lambda m: str(round(float(m.group()) * 2, 1)), text)

rows = []
for task in sorted(full.task.unique()):
    fn = reward_for(task)
    r = full[full.task == task].iloc[0]
    rows.append({"task": task,
                 "reward": fn.__name__ if fn else "없음 (검증 불가)",
                 "정답": round(fn(r.target, r.target), 3) if fn else None,
                 "숫자 2배": round(fn(double_numbers(r.target), r.target), 3) if fn else None})
pd.DataFrame(rows)
""")

md("""
## 10. 레이더 프로브의 오염도

평가 전용 프로브입니다. **오염도**는 프로브 정답이 인코더가 지도학습한 스칼라
(`n_points`, `n_moving`, `max_rcs`)와 갖는 최대 상관입니다. 이 값이 크면 모델이 그
스칼라만 읽어도 프로브를 풀 수 있으므로, 그 프로브로는 레이더 이해를 잴 수 없습니다.
`radar_probe`가 그렇게 순환 측정이 됐습니다.
""")

code("""
NUM = re.compile(r"-?\\d+(?:\\.\\d+)?")
feat = pd.read_parquet(os.path.join(COMMON, "scene_features_all_clips.parquet"),
                       columns=["clip_id", "frame", "lrr1_n_points",
                                "lrr1_n_moving", "lrr1_max_rcs"])
SUPERVISED = ["lrr1_n_points", "lrr1_n_moving", "lrr1_max_rcs"]

out = []
for name in ("radar_structure_probes.parquet", "radar_object_probes.parquet"):
    path = os.path.join(COMMON, name)
    if not os.path.exists(path):
        continue
    pr = pd.read_parquet(path)
    m = (pr[pr.split == "test"].merge(feat, on=["clip_id", "frame"])
                               .dropna(subset=["lrr1_n_points"]))
    m = m.assign(first=m.target.map(lambda s: float(NUM.findall(s)[0])))
    for form, g in m.groupby("form"):
        worst = max(abs(float(np.corrcoef(g["first"], g[c])[0, 1]))
                    for c in SUPERVISED)
        out.append({"probe": name.split("_")[1], "form": form, "n": len(g),
                    "오염도": round(worst, 3),
                    "판정": "깨끗" if worst < 0.3 else "부분적" if worst < 0.7 else "오염"})
pd.DataFrame(out).sort_values("오염도")
""")


def main():
    nb = {"cells": CELLS,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "inspect_data.ipynb")
    with open(out, "w") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print(f"wrote {out}  ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
