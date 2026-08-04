#!/usr/bin/env python3
"""Generate notebooks/datatools/det_object.ipynb.

Written as a generator rather than by hand for the same reason as the other
notebook: `source` goes in as one string, because a list of lines without their
trailing newlines gets concatenated into a single unparseable line the next time
anything writes the file.

    python notebooks/datatools/build_det_object_notebook.py
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
# 01 · det_objects — 정답은 어디서 어떻게 만들어지는가

`det_objects`의 정답 한 줄

```
automobile 34 m az +13 deg moving; automobile 68 m az +9 deg stationary; ...
```

이 문장이 원시 아카이브에서 나오기까지의 과정을 단계별로 보여줍니다. 먼저 **손으로 만든
장난감 예제**로 각 변환이 무엇을 하는지 확인하고, 그다음 **실제 클립**에 같은 함수를 적용해
결과를 비교합니다.

GPU가 필요 없고, 장난감 예제 부분은 원시 데이터 없이도 돌아갑니다.
""")

code("""
import os, sys, json, textwrap

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 170)
pd.set_option("display.max_colwidth", 70)
np.set_printoptions(precision=2, suppress=True)
print("ready")
""")

md("""
## 0. 원재료 두 가지

| 아카이브 | 주기 | 무엇이 들어있나 |
|---|---|---|
| `obstacle.offline` | 10 Hz | 3D 박스: `track_id`, `center_x/y/z`, `size_*`, `orientation_*`, `label_class`, `reference_frame='rig'` |
| `egomotion` | 10 Hz | 자차 위치·자세: `x, y, z, qx, qy, qz, qw` |

두 가지 사실이 이후 모든 단계를 결정합니다.

**① 박스 좌표계가 `rig`입니다.** 자차 기준이라는 뜻이고, 자차가 움직이면 정지 물체도
좌표가 변합니다.

**② 박스는 오토라벨입니다** (`source = 'scene:obstacles:autolabels:v2'`). 사람이 검증한
텍스트는 QA(태스크 10)뿐입니다.
""")

md("""
## 1. 장난감 데이터 만들기

자차가 **+x 방향으로 10 m/s로 직진**하는 3초짜리 장면을 만듭니다. 물체는 셋:

| 트랙 | 정체 | 월드에서 |
|---|---|---|
| 100 | 주차된 차 | 안 움직임 |
| 200 | 같은 속도로 앞서 가는 차 | 10 m/s로 이동 |
| 300 | 왼쪽에 선 사람 | 안 움직임 |

핵심은 **rig 좌표에서 보면 100·300이 뒤로 흘러가고 200은 멈춰 보인다**는 것입니다.
실제 데이터의 함정이 그대로 재현됩니다.
""")

code("""
HZ, DURATION, EGO_SPEED = 10, 3.0, 10.0
t = np.arange(0, DURATION, 1 / HZ)

# egomotion: +x 로 등속 직진, 회전 없음
ego = pd.DataFrame({
    "timestamp": (t * 1e6).astype(np.int64),
    "x": EGO_SPEED * t, "y": 0.0, "z": 0.0,
    "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
})

# 물체의 월드 좌표 궤적
world = {
    100: ("automobile", lambda tt: (40.0 + 0 * tt,  6.0 + 0 * tt)),   # 주차
    200: ("automobile", lambda tt: (70.0 + 10 * tt, 1.0 + 0 * tt)),   # 같은 속도
    300: ("person",     lambda tt: (25.0 + 0 * tt, -8.0 + 0 * tt)),   # 서 있는 사람
}

rows = []
for tid, (cls, path) in world.items():
    wx, wy = path(t)
    # 회전이 없으므로 rig = world - ego_position
    rows.append(pd.DataFrame({
        "timestamp_us": (t * 1e6).astype(np.int64), "track_id": tid,
        "center_x": wx - EGO_SPEED * t, "center_y": wy, "center_z": 0.8,
        "size_x": 4.2, "size_y": 1.8, "size_z": 1.6,
        "orientation_x": 0.0, "orientation_y": 0.0,
        "orientation_z": 0.0, "orientation_w": 1.0,
        "label_class": cls, "reference_frame": "rig",
    }))
obstacle = pd.concat(rows, ignore_index=True)

print(f"obstacle {len(obstacle)} 행, ego {len(ego)} 행")
obstacle[obstacle.timestamp_us.isin([0, int(2e6)])].sort_values(["timestamp_us", "track_id"])[
    ["timestamp_us", "track_id", "label_class", "center_x", "center_y"]]
""")

md("""
## 2. `boxes_world()` — 거리·방위각·이동 판정

세 가지를 계산합니다.

**거리와 방위각은 rig 좌표에서 바로** 나옵니다. 자차 기준 질문이니 그게 맞습니다.

```python
range_m     = hypot(center_x, center_y)
azimuth_deg = degrees(atan2(center_y, center_x))     # + 가 왼쪽
```

**이동 판정만 월드 좌표에서** 합니다. egomotion으로 자세를 보간해 박스를 월드로 옮긴 뒤,
트랙의 첫 관측과 마지막 관측 사이 변위가 `STATIONARY_M = 2.0` m 이상이면 `moved = True`.
""")

code("""
from datatools.frame_objects import boxes_world, STATIONARY_M

boxes = boxes_world(obstacle, ego)
print(f"STATIONARY_M = {STATIONARY_M} m\\n")

summary = (boxes.groupby(["track_id", "label_class"])
                .agg(rig_first=("center_x", "first"), rig_last=("center_x", "last"),
                     wx_first=("wx", "first"), wx_last=("wx", "last"),
                     moved=("moved", "first")))
summary["rig_변위"] = (summary.rig_last - summary.rig_first).round(1)
summary["world_변위"] = (summary.wx_last - summary.wx_first).round(1)
summary[["rig_변위", "world_변위", "moved"]]
""")

md("""
### 왜 월드 프레임이어야 하는가

위 표를 보세요.

- 트랙 **100**(주차) · **300**(사람): rig 변위 **-30 m**, world 변위 **0 m** → 정지
- 트랙 **200**(같은 속도): rig 변위 **0 m**, world 변위 **+30 m** → 이동

rig 변위로 판정했다면 **정확히 반대로** 라벨링됩니다. 실제 데이터에서 이 오류를 측정한
결과가 이렇습니다.

| | 월드 기준 | rig 기준 |
|---|---|---|
| 정지 비율 (14,844 트랙) | **71.6%** | **0.4%** |
| 두 변위의 상관 | \\multicolumn{2}{c}{0.202} |

에이전트 궤적 필터의 첫 버전이 rig 변위를 썼고, 그래서 거의 정확히 **틀린 트랙들**을
골라내고 있었습니다.
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
colors = {100: "tab:blue", 200: "tab:orange", 300: "tab:green"}
for tid, g in boxes.groupby("track_id"):
    lbl = f"#{tid} {g.label_class.iloc[0]}"
    axes[0].plot(g.t_s, g.center_x, "o-", ms=3, color=colors[tid], label=lbl)
    axes[1].plot(g.t_s, g.wx, "o-", ms=3, color=colors[tid], label=lbl)
axes[0].set_title("rig 좌표 (자차 기준) — 주차 차가 흘러간다")
axes[1].set_title("world 좌표 — 실제로 움직이는 것은 #200 뿐")
for a in axes:
    a.set_xlabel("t (s)")
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
axes[0].set_ylabel("x (m)")
plt.tight_layout()
plt.show()
""")

md("""
## 3. `visible_at(boxes, t_s)` — 그 순간, 그 섹터

네 가지를 합니다.

```python
near = boxes[|t_s - box.t_s| <= 0.15]                  # 라벨이 10 Hz라 정확히 t초가 없을 수 있음
near = near[|azimuth| <= 60  and  range <= 300]        # 전방 섹터
near = near.sort_values("_dt").drop_duplicates("track_id")   # 트랙당 시간상 최근접 관측
return near.sort_values("range_m")                     # 가까운 순
```

**±60° 절단이 중요합니다.** 라벨은 360°를 덮지만 전방 광각 카메라와 imaging LRR은 약
±60°만 봅니다. 관측 가능한 라벨은 **46%**뿐이고, 섹터 밖 라벨로 학습하면 언어모델이
**차 뒤의 물체를 지어내도록** 배웁니다.
""")

code("""
from datatools.frame_objects import visible_at, BOX_TIME_WINDOW_S
from datatools import paths

print(f"시간 창 ±{BOX_TIME_WINDOW_S} s, 섹터 ±{paths.FRONT_FOV_DEG}°, "
      f"최대 {paths.FRONT_MAX_RANGE_M:.0f} m\\n")

for t_s in (0.0, 2.0):
    here = visible_at(boxes, t_s)
    print(f"--- t = {t_s} s : {len(here)} 개")
    print(here[["track_id", "label_class", "range_m", "azimuth_deg", "moved"]]
          .round(1).to_string(index=False))
    print()
""")

code("""
# 섹터 밖으로 나가는 물체를 보기 위해, 왼쪽 사람을 자차 옆으로 가져와 본다
demo = obstacle.copy()
demo.loc[demo.track_id == 300, "center_y"] = -20.0

boxes_demo = boxes_world(demo, ego)
rows = []
for t_s in np.arange(0, 3.0, 0.5):
    g = boxes_demo[(boxes_demo.t_s - t_s).abs() <= BOX_TIME_WINDOW_S]
    g = g.sort_values("t_s").drop_duplicates("track_id")
    for _, r in g.iterrows():
        rows.append({"t_s": t_s, "track": r.track_id, "range": round(r.range_m, 1),
                     "az": round(r.azimuth_deg, 1),
                     "섹터 안": abs(r.azimuth_deg) <= paths.FRONT_FOV_DEG})
pd.DataFrame(rows).pivot(index="t_s", columns="track", values=["az", "섹터 안"])
""")

md("""
## 4. 상위 8개 절단

`MAX_LISTED = 8`. 가까운 순으로 자릅니다.

이건 **재현율 상한을 만듭니다** — 앞에 물체가 12개 있어도 정답에는 8개만 적히므로,
모델이 9번째를 맞혀도 오답(false positive)으로 채점됩니다. 답 길이를 통제하기 위한
의도적 절충이고, 알고 쓰는 것이 중요합니다.
""")

code("""
from datatools.frame_objects import MAX_LISTED
print("MAX_LISTED =", MAX_LISTED)
""")

md("""
## 5. `describe_object()` — 문장으로

```python
f"{label_class} {range_m:.0f} m az {azimuth_deg:+.0f} deg" + (" moving" | " stationary")
```

세미콜론으로 이어 붙이면 정답이 완성됩니다.

거리와 방위각을 **정수로 반올림**하는 것도 선택입니다. 소수점을 남기면 채점기가 34.2와
34.4를 다른 답으로 보게 되는데, 오토라벨의 정확도가 거기까지 가지 않습니다.
""")

code("""
from datatools.frame_objects import describe_object

t_s = 2.0
here = visible_at(boxes, t_s)
listed = here.head(MAX_LISTED)

answer = "; ".join(describe_object(r) for r in listed.itertuples())
print(f"PROMPT : At frame {int(t_s) + 1}. List every road user in the forward "
      f"sector with its class, range and azimuth.")
print(f"TARGET : {answer}")
""")

md("""
## 6. rationale — 레이더 증거 붙이기

CoT 변형(`det_objects_cot`)은 답 앞에 **레이더가 무엇을 뒷받침하는지**를 붙입니다.
`radar_hits(scan, rows)`가 박스마다 그 안에 들어온 반사점을 세고, 세 값을 돌려줍니다.

```
(반사점 수, 자차 운동 제거 후에도 움직이는 반사점 수, 중앙값 시선속도)
```

이 구분이 근거가 되는 이유는 **레이더가 물체를 가려서 보기 때문**입니다.

| 클래스 | 반사점을 갖는 비율 |
|---|---|
| heavy_truck | 85.7% |
| automobile | 67.1% |
| rider | 29.0% |
| person | **20.2%** |
| protruding_object | **0.0%** |

즉 박스 기반 레이더 지도(supervision)는 **차량 클래스에서만 신뢰할 수 있습니다.**
""")

code("""
from datatools.geometry import points_in_box
from datatools.frame_objects import BOX_MARGIN

# 장난감 스캔: #100 위에 6점, #200 위에 3점, 사람(#300)에는 0점
rng = np.random.default_rng(0)
target_rows = visible_at(boxes, 2.0).set_index("track_id")
pts = []
for tid, n in ((100, 6), (200, 3)):
    c = target_rows.loc[tid]
    pts.append(np.column_stack([
        c.center_x + rng.normal(0, 0.8, n),
        c.center_y + rng.normal(0, 0.5, n),
        np.full(n, 0.8)]))
pts.append(rng.uniform([-5, -30, 0], [120, 30, 1], size=(40, 3)))   # 잡음
rig_pts = np.vstack(pts)

for tid in (100, 200, 300):
    c = target_rows.loc[tid]
    inside = points_in_box(rig_pts,
                           np.array([c.center_x, c.center_y, c.center_z]),
                           [c.size_x, c.size_y, c.size_z],
                           [c.orientation_x, c.orientation_y,
                            c.orientation_z, c.orientation_w], BOX_MARGIN)
    print(f"track #{tid} ({c.label_class:10s}) : 박스 안 반사점 {int(inside.sum())}개")
print(f"\\nBOX_MARGIN = {BOX_MARGIN} (박스를 이만큼 키워서 셈)")
""")

md("""
## 7. 실제 클립으로 같은 함수 돌리기

여기서부터는 원시 아카이브가 필요합니다. 위에서 본 것과 **똑같은 함수들**이 실제
데이터에 적용됩니다.
""")

code("""
from datatools.frame_objects import read_member, DET_ANCHOR_S

clips = pd.read_parquet(os.path.join(paths.COMMON_DIR, "nvidia_clips.parquet"))
usable = clips[clips.has_obstacle.fillna(False) & clips.has_egomotion.fillna(False)]
row = usable.loc[usable.index[0]]
clip_id = usable.index[0]

real_ob = read_member(paths.NVIDIA_ROOT, row.obstacle_zip, row.obstacle_member)
real_ego = read_member(paths.NVIDIA_ROOT, row.egomotion_zip, row.egomotion_member)
print(f"clip {clip_id}")
print(f"obstacle {len(real_ob):,} 행, {real_ob.track_id.nunique()} 트랙, "
      f"클래스 {sorted(real_ob.label_class.unique())}")
print(f"source   {real_ob.source.iloc[0]}")
print(f"앵커     {DET_ANCHOR_S} 초")
""")

code("""
real_boxes = boxes_world(real_ob, real_ego)

for seconds in DET_ANCHOR_S:
    here = visible_at(real_boxes, float(seconds))
    if here is None or here.empty:
        print(f"t={seconds:>2}s : 전방 섹터에 물체 없음")
        continue
    listed = here.head(MAX_LISTED)
    answer = "; ".join(describe_object(r) for r in listed.itertuples())
    print(f"t={seconds:>2}s : 섹터 안 {len(here):>2}개 (상위 {len(listed)}개 기재)")
    print("        ", textwrap.fill(answer, 92, subsequent_indent=" " * 9))
""")

md("""
## 8. 빌드된 파일과 대조

위에서 손으로 만든 문자열이 실제 파일의 `target`과 같아야 합니다. 다르면 노트북이
파이프라인을 잘못 재현하고 있다는 뜻입니다.
""")

code("""
ITEMS = os.path.join(paths.COMMON_DIR, "instruct_items_tasks01_06.parquet")
built = pd.read_parquet(ITEMS)
mine = built[(built.clip_id == clip_id) & (built.task == "det_objects")]

if mine.empty:
    print("이 클립의 det_objects 행이 없습니다 (재생성 전이거나 필터로 빠짐)")
else:
    ok = 0
    for _, r in mine.sort_values("frame").iterrows():
        here = visible_at(real_boxes, float(r.frame - 1))
        rebuilt = ("No road users in the forward sector." if here is None or here.empty
                   else "; ".join(describe_object(x)
                                  for x in here.head(MAX_LISTED).itertuples()))
        same = rebuilt == r.target
        ok += same
        print(f"  frame {r.frame:>2} (t={r.frame - 1:>2}s) : {'일치' if same else '불일치'}")
        if not same:
            print("    파일 :", r.target[:88])
            print("    재현 :", rebuilt[:88])
    print(f"\\n{ok}/{len(mine)} 일치")
""")

md("""
## 9. 알아두어야 할 한계 세 가지

**① 라벨이 오토라벨입니다.** `source = 'scene:obstacles:autolabels:v2'`. 거리·방위각의
절대 정확도는 그 품질에 묶여 있습니다. 사람이 검증한 것은 QA(태스크 10)뿐입니다.

**② `moved`는 트랙 전체 기준입니다.** 첫 관측과 마지막 관측의 월드 변위로 판정하므로,
클립 초반에 서 있다가 후반에 출발한 차는 **모든 시점에서** `moving`으로 표기됩니다.
`det_objects`가 3/6/9/12/15/18초 여섯 시점으로 늘어나면서 이 근사가 더 눈에 띕니다 —
여섯 시점의 이동 표기가 전부 같기 때문입니다.

시점별 순간 속도로 바꿀 수 있지만, 그러면 태스크 06(`motion_seg`, 도플러 기반)과 판정
기준이 갈라집니다. 아래에서 실제로 얼마나 문제인지 세어 봅니다.

**③ 상위 8개 절단이 재현율 상한을 만듭니다.** 앞에 물체가 12개면 정답은 8개까지만
적히므로, 모델이 9번째를 맞혀도 오답으로 채점됩니다.
""")

code("""
# ② 가 실제로 얼마나 흔한가: 클립 안에서 정지->이동으로 바뀌는 트랙의 비율
sample = usable.index[:60]
changed = total = 0
for cid in sample:
    r = usable.loc[cid]
    try:
        ob = read_member(paths.NVIDIA_ROOT, r.obstacle_zip, r.obstacle_member)
        eg = read_member(paths.NVIDIA_ROOT, r.egomotion_zip, r.egomotion_member)
        b = boxes_world(ob, eg)
    except Exception:
        continue
    for tid, g in b.groupby("track_id"):
        if len(g) < 20:
            continue
        total += 1
        half = len(g) // 2
        d1 = np.hypot(g.wx.iloc[half] - g.wx.iloc[0], g.wy.iloc[half] - g.wy.iloc[0])
        d2 = np.hypot(g.wx.iloc[-1] - g.wx.iloc[half], g.wy.iloc[-1] - g.wy.iloc[half])
        if (d1 < 1.0) != (d2 < 1.0):        # 전반/후반의 이동 여부가 다르다
            changed += 1

print(f"트랙 {total:,}개 중 전반/후반 이동 상태가 바뀌는 것: "
      f"{changed:,}개 ({100 * changed / max(total, 1):.1f}%)")
print("\\n이 비율이 낮으면 트랙 전체 기준 판정이 실용적으로 문제없다는 뜻이고,")
print("높으면 시점별 판정으로 바꿀 이유가 됩니다.")
""")

md("""
## 요약

```
obstacle.offline (3D 박스, rig 좌표, 10 Hz)
egomotion        (자차 자세, 10 Hz)
        │
        ├─ boxes_world()      rig → 거리·방위각,  world → 이동 여부(2 m 임계)
        │
        ├─ visible_at(t)      ±0.15 s 창 → ±60° / 300 m 섹터 → 트랙당 1관측 → 거리순
        │
        ├─ head(8)            상위 8개
        │
        ├─ describe_object()  "automobile 34 m az +13 deg moving"
        │
        └─ (CoT) radar_hits() 박스별 반사점 수·이동 반사점 수·시선속도
```

입력 쪽은 `det_objects`만 다릅니다 — 비전 **1프레임**, 레이더는 그 순간 직전 **20스캔**
(LRR·MRR 기준 약 1초). 자세한 내용은 `notebooks/inspect_data.ipynb` 6절을 보세요.
""")


def main():
    nb = {"cells": CELLS,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "det_object.ipynb")
    with open(out, "w") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print(f"wrote {out}  ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
