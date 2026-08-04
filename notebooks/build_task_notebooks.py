#!/usr/bin/env python3
"""One notebook per task: which raw archive, which code, which numbers.

`det_object.ipynb` walks task 01 by hand. That does not scale to ten tasks, and
a hand-written notebook drifts from the code the moment either changes. These
are generated from one template driven by a table of task facts, so a task whose
definition moves is one edit here rather than one edit per notebook.

Each notebook answers the same four questions:

    where does it come from   the raw archives and the columns actually read
    how is it built           the functions, in the order they run
    what goes in              the video / radar / ego window the loader hands over
    is it right               rebuild the answer from the labels and compare it
                              against the file, item by item

`source` is written as a single string. A list of lines whose entries lack their
trailing newline gets concatenated into one unparseable line the next time
anything writes the file, which is how the first attempt at these was lost.

    python notebooks/build_task_notebooks.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tasks")

# (folder number, task keys, title, raw sources, pipeline, window, notes)
TASKS = [
    dict(
        no="1", key="det_objects", variants=["det_objects_azdeg", "det_objects_3dbbox"],
        title="det_objects — 지금 앞에 무엇이 있는가",
        problem="전방 ±60° 안의 도로 사용자를 클래스와 위치로 열거합니다. 한 순간을 "
                "묻기 때문에 미래 지평선 제약이 없고, 그래서 앵커를 3/6/9/12/15/18초 "
                "여섯 곳에 둘 수 있습니다.",
        raw=[("obstacle.offline", "10 Hz", "center_x/y/z, size_*, orientation_*, "
              "label_class, track_id, reference_frame='rig'"),
             ("egomotion", "10 Hz", "x, y, z, qx..qw — rig 좌표를 월드로 올리는 데 필요"),
             ("radar (lrr1/mrr2/srr0)", "20 또는 12.7 Hz", "azimuth, elevation, "
              "distance, radial_velocity, rcs, doppler_ambiguity"),
             ("camera intrinsics", "클립당 1", "f-theta 다항식 — CoT 의 카메라 방위각용")],
        pipeline=[("boxes_world(obstacle, ego)",
                   "rig 좌표에서 거리·방위각, 월드 좌표에서 이동 여부"),
                  ("visible_at(boxes, t_s)",
                   "±0.15 s 창 → ±60°/300 m 섹터 → 트랙당 1관측 → 거리순"),
                  ("head(MAX_LISTED=8)", "가까운 순 8개"),
                  ("describe_object / describe_object_xyz", "문장화"),
                  ("object_evidence(scan, listed)",
                   "CoT 용: 박스 안 반사점의 수·거리·방위각"),
                  ("camera_azimuth(box centre)",
                   "CoT 용: 이미지 박스에서 역산한 방위각")],
        window="비전 1장 (질문한 순간) · 레이더 20스캔 (직전 약 1초) · ego 1샘플",
        notes=[("이동 표기를 뺐다",
                "라벨의 moved 는 트랙 전체 20초 변위인데 입력은 1프레임 + 1초입니다. "
                "레이더가 보는 물체만 놓고도 그 순간의 도플러와 85.8% 만 일치했고, "
                "나머지는 입력으로 도달할 수 없는 정답이었습니다. 이동 판정은 06 이 "
                "맡습니다."),
               ("상위 8개 절단",
                "앞에 물체가 12개 있어도 8개만 적히므로, 9번째를 맞혀도 오탐이 됩니다."),
               ("오토라벨",
                "source='scene:obstacles:autolabels:v2'. 사람이 검증한 텍스트는 QA 뿐입니다.")],
    ),
    dict(
        no="2", key="track_step", variants=["track_step_azdeg", "track_step_bbox"],
        title="track_step — 이력을 보고 지금을 잇는다",
        problem="t-4..t-1 의 검출 결과를 주고 t 의 물체를 답하게 합니다. 물체는 이미 "
                "가진 id 를 그대로 써야 하고, 처음 보는 것에만 새 id 를 줍니다. "
                "표준적인 tracking 정식화입니다.",
        raw=[("obstacle.offline", "10 Hz", "track_id 가 정체성의 근거"),
             ("egomotion", "10 Hz", "월드 좌표 변환"),
             ("radar", "20 / 12.7 Hz", "5초 창을 4 Hz 로 20스캔"),
             ("camera intrinsics", "클립당 1", "bbox 형식과 CoT 의 가려짐 계산")],
        pipeline=[("local_ids(boxes)",
                   "track_id 를 첫 등장 순으로 1,2,3... 로 재번호. 원본 id 는 임의의 "
                   "큰 수이고, 이 태스크가 묻는 것은 번호가 아니라 일관성입니다"),
                  ("step_frame(boxes, t, ids, camera, scan)",
                   "한 순간의 물체 + 카메라 방위각 + 가려짐 비율 + 레이더 측정"),
                  ("occluded_fraction(box, nearer)",
                   "앞선 물체들이 이 박스를 덮는 면적 비율"),
                  ("step_text(objects, form)", "이력과 답을 같은 형식으로"),
                  ("step_reason(now, previous, form)", "CoT 근거")],
        window="비전 5장 (1 fps) · 레이더 20스캔 (5초, 4 Hz) · ego 20샘플 · 이력 4프레임",
        notes=[("앵커가 1초 간격인 이유",
                "롤아웃에서 이력은 자기 출력이므로 앵커 간격이 곧 이력 간격이 됩니다. "
                "3초 간격 앵커에 1초 간격 이력을 쓰면 학습과 평가의 프롬프트가 달라집니다."),
               ("id 는 값이 아니라 일관성으로 채점",
                "첫 차를 #2 라 불러도 계속 #2 이면 맞습니다. 항목 단위로는 프롬프트의 "
                "이력이 정한 id 를 이어받았는지 보고, 시퀀스 단위로는 IDF1 로 대응을 "
                "한 번 풀어 셉니다."),
               ("가려짐이 76% 항목에 등장",
                "물체의 25.6% 가 이미지에서 절반 이상 가려지고, 그중 48.2% 는 레이더가 "
                "여전히 봅니다. 융합이 값을 하는 지점입니다.")],
    ),
    dict(
        no="3.1", key="plan_ego", variants=["plan_ego_xy", "plan_ego_control"],
        title="plan_ego — 자차는 어디로 가는가",
        problem="앞으로 3초의 자차 경로를 예측합니다. 정답이 egomotion 센서 측정값에서 "
                "직접 나오므로 이 데이터셋에서 가장 신뢰도 높은 타깃입니다.",
        raw=[("egomotion", "10 Hz", "x, y, qx..qw — 이것 하나로 정답이 만들어집니다"),
             ("radar", "20 / 12.7 Hz", "2초 창 20스캔. 앞차의 감속을 읽는 데 쓰입니다"),
             ("camera", "1 Hz", "2장. 신호등·정지선·커브가 운전자 의도를 정합니다")],
        pipeline=[("ego_frame(ego)", "속도, 가속도, 요, 요레이트 파생"),
                  ("ego_waypoints(derived, t_s)",
                   "t+1/2/3초 위치를 현재 자차 좌표계로 회전 변환"),
                  ("ego_controls(derived, t_s)", "같은 궤적을 속도·요레이트로"),
                  ("ego_action(derived, t_s)", "CoT 용 기동 분류")],
        window="비전 2장 (1 fps) · 레이더 20스캔 (2초, 10 Hz) · ego 20샘플",
        notes=[("등속 외삽으로는 부족",
                "현재 속도 × 시간으로 외삽하면 오차 중앙값 1.09 m 이고 보상의 반점 "
                "기준(1 m) 안에 드는 것이 47.9% 뿐입니다. 감속과 조향을 예측해야 하고, "
                "그건 장면에 달려 있습니다."),
               ("앵커 2초 간격",
                "1초 간격이면 인접 정답 차이가 중앙값 0.72 m 로 반점 기준보다 작아 "
                "같은 문항의 반복이 됩니다. 2초면 1.42 m 로 넘어섭니다."),
               ("t=17 이 마지막",
                "18+3=21초는 클립 밖입니다."),
               ("근거가 레이더가 아닌 유일한 태스크",
                "내가 어디로 갈지는 내 속도와 조향이 정합니다. 레이더 개입이 여기까지 "
                "영향을 준다면 그것은 레이더 능력이 아니라는 신호입니다.")],
    ),
    dict(
        no="3.2", key="agent_traj", variants=["agent_traj_azdeg", "agent_traj_bbox"],
        title="agent_traj — 저 물체는 어디로 가는가",
        problem="지목한 물체 하나의 3초 뒤를 예측합니다. 지평선 안에 섹터를 벗어나면 "
                "그렇다고 명시적으로 답합니다.",
        raw=[("obstacle.offline", "10 Hz", "대상의 현재와 미래 위치"),
             ("egomotion", "10 Hz", "월드 변환"),
             ("radar", "20 / 12.7 Hz", "시선속도 — 이 태스크의 핵심 근거"),
             ("camera intrinsics", "클립당 1", "bbox 형식")],
        pipeline=[("visible_at(boxes, t_s)", "현재 물체들"),
                  ("movers[0]", "가장 가까운 이동 물체를 대상으로"),
                  ("visible_at(boxes, t_s + h)", "각 지평선에서 같은 track_id 를 추적"),
                  ("image_bbox(...)", "bbox 형식: 미래 시점의 박스"),
                  ("object_evidence", "CoT 용 반사점·시선속도")],
        window="비전 2장 · 레이더 20스캔 (2초, 10 Hz) · ego 20샘플",
        notes=[("절반이 사라진다",
                "지목한 물체가 +3s 까지 남는 비율이 51.4% 입니다. 답을 잘라내면 짧은 "
                "답이 항상 안전하다고 학습되므로, 'leaves the forward sector' 를 "
                "명시적 답으로 만들었습니다. 항목의 51% 에 등장합니다."),
               ("보상을 조였다",
                "시선속도 등속 외삽만으로 5 m 이내가 91.7% 였습니다. 그 기준으로는 "
                "레이더를 읽는 모델과 외삽하는 모델이 구분되지 않아 거리 반점을 "
                "5 m → 2 m, 방위각을 10° → 5° 로 낮췄습니다."),
               ("물리적으로 가장 강한 사슬",
                "시선속도 × 시간 = 거리 변화. 단안 카메라로는 얻을 수 없는 양입니다.")],
    ),
    dict(
        no="6", key="motion_seg", variants=["motion_seg_azdeg", "motion_seg_bbox"],
        title="motion_seg — 움직이는가, 서 있는가",
        problem="두 개의 수직인 잔차로 판정합니다. 레이더 도플러는 시선 방향을, 카메라 "
                "방위각은 횡방향을 봅니다. 서로가 못 보는 축이 정확히 반대입니다.",
        raw=[("obstacle.offline", "10 Hz", "물체의 월드 궤적 → 순간 속도"),
             ("egomotion", "10 Hz", "두 예상값의 출발점: 속도와 요레이트"),
             ("radar", "20 / 12.7 Hz", "시선속도 측정"),
             ("camera intrinsics", "클립당 1", "방위각과 bbox")],
        pipeline=[("world_speeds(boxes)",
                   "트랙별 월드 속도를 중심차분으로. 순간 라벨의 근거"),
                  ("motion_evidence(row, previous, derived, t_s, v_rig, hit)",
                   "정지 물체라면 보일 시선속도와 방위각을 자차 상태에서 계산"),
                  ("object_evidence(scan, listed)", "실제 측정값")],
        window="비전 2장 · 레이더 20스캔 (2초, 10 Hz) · ego 20샘플",
        notes=[("라벨을 순간 속도로 바꿨다",
                "트랙 전체 변위는 2초 창으로 확립할 수 없는 사실입니다. 순간 월드속도로 "
                "바꾸니 도플러와의 일치율이 85.8% → 88.6% 가 됐습니다."),
               ("두 센서가 서로의 사각을 메운다",
                "카메라 각도 잔차는 정지 물체 중앙값 0.24°, 이동 2.31° 이고, 도플러가 "
                "놓친 이동 물체의 61.4% 를 잡습니다. 반대로 정면으로 다가오는 물체는 "
                "화면에서 안 움직이지만 도플러가 확실히 봅니다."),
               ("앵커 2초 간격",
                "1초 간격에서 이동/정지가 바뀌는 물체는 3.7% 뿐입니다.")],
    ),
    dict(
        no="10", key="qa", variants=["qa"],
        title="qa — 5지선다와 그 근거",
        problem="장면에 대한 객관식 질문입니다. 원본이 답과 함께 풀이(rationale)를 "
                "제공하고, 그 풀이의 수치를 라벨로 재계산해 검증했습니다.",
        raw=[("10_radar_vision_qa/qa_train/*.json", "클립당 약 20문항",
              "question, options(A~E), answer, rationale, verification"),
             ("10_radar_vision_qa/qa_gt/*.json", "139 클립 2,019문항",
              "사람 검수본. 평가 전용이고 자동 수정을 적용하지 않았습니다"),
             ("egomotion / obstacle.offline", "10 Hz",
              "rationale 의 수치를 재계산해 대조하는 데 쓰입니다")],
        pipeline=[("qa_claims.extract(rationale)",
                   "문장 단위로 수치 주장을 뽑음: ego_pos, ego_speed, agent_pos, "
                   "agent_speed, distance, future_pos"),
                  ("verify_qa", "각 주장을 라벨에서 재계산해 비교"),
                  ("correct_qa_numbers", "safe 판정만 라벨값으로 교체"),
                  ("drop_bad_qa_items", "고칠 수 없는 항목 제거"),
                  ("flag_qa_verification", "verification 필드 부착")],
        window="비전 20장 · 레이더 20스캔 (20초, 1 Hz) · ego 질문 프레임까지",
        notes=[("두 세트를 합쳤다",
                "기존 1,999 클립과 새로 받은 8,000 클립은 클립이 하나도 겹치지 않아 "
                "중복 제거 없이 병합했습니다. 9,999 클립 195,874 문항입니다."),
               ("자체 검증과 수정",
                "불일치 주장 중 서술 맥락(safe)만 라벨값으로 바꾸고, 계산에 쓰이거나 "
                "보기와 연동된 것은 항목째 제거했습니다. rationale 은 사실 목록이 "
                "아니라 논증이라, 한 숫자를 바꾸면 다음 줄이 틀려집니다."),
               ("검증기가 틀렸던 지점",
                "사람 검수본의 거리 주장이 처음 58.6% 로 나왔는데, 추출기가 "
                "'X-position is 349.69m' 를 거리로 오인한 탓이었습니다. 고친 뒤 "
                "82.1% 입니다."),
               ("CoT 는 모순이 확인된 것만 제외",
                "'agrees' 만 쓰면 62% 를 버리는데, 그 대부분은 틀린 것이 아니라 "
                "대조할 숫자가 없는 정성 추론입니다. 숫자가 아예 없는 것은 13.7% 뿐입니다.")],
    ),
]

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "id": f"md{len(CELLS):02d}",
                  "metadata": {}, "source": text.strip("\n")})


def code(text):
    CELLS.append({"cell_type": "code", "id": f"code{len(CELLS):02d}",
                  "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.strip("\n")})


def notebook(spec):
    global CELLS
    CELLS = []
    variants = spec["variants"]
    md(f"# {spec['no']} · {spec['title']}\n\n{spec['problem']}\n\n"
       f"이 노트북은 네 가지를 확인합니다 — **어떤 원시 데이터에서**, "
       f"**어떤 코드를 거쳐**, **무엇이 입력으로 들어가고**, "
       f"**빌드된 파일이 그 코드와 일치하는지**. GPU 는 필요 없습니다.")

    code("""
import os, sys, json, textwrap
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

import numpy as np
import pandas as pd

from datatools import paths

ITEMS = os.path.join(paths.COMMON_DIR, "instruct_items_tasks01_06.parquet")
pd.set_option("display.width", 170)
pd.set_option("display.max_colwidth", 80)
wrap = lambda s, i="   ": textwrap.fill(str(s), 94, initial_indent=i,
                                        subsequent_indent=i)
VARIANTS = """ + repr(variants) + """
print("variants:", VARIANTS)
""")

    md("## 1. 어떤 원시 데이터에서 오는가\n\n" +
       "| 아카이브 | 주기 | 읽는 것 |\n|---|---|---|\n" +
       "\n".join(f"| `{a}` | {b} | {c} |" for a, b, c in spec["raw"]))

    md("## 2. 어떤 코드를 거치는가\n\n실행 순서입니다.\n\n" +
       "| 함수 | 하는 일 |\n|---|---|\n" +
       "\n".join(f"| `{a}` | {b} |" for a, b in spec["pipeline"]) +
       "\n\n전부 `datatools/frame_objects.py` 와 `datatools/geometry.py` 에 있습니다."
       if spec["key"] != "qa" else
       "## 2. 어떤 코드를 거치는가\n\n실행 순서입니다.\n\n" +
       "| 단계 | 하는 일 |\n|---|---|\n" +
       "\n".join(f"| `{a}` | {b} |" for a, b in spec["pipeline"]))

    code("""
import inspect
""" + ("from datatools import frame_objects as F\n"
       "for name in " + repr([a.split("(")[0] for a, _ in spec["pipeline"]]) + ":\n"
       "    fn = getattr(F, name, None)\n"
       "    if fn is None:\n"
       "        from datatools import geometry as G\n"
       "        fn = getattr(G, name, None)\n"
       "    if fn is None or not callable(fn):\n"
       "        print(f'{name}: (모듈 함수 아님)'); continue\n"
       "    doc = (inspect.getdoc(fn) or '').split(chr(10))[0]\n"
       "    print(f'{name:34s} {doc[:80]}')"
       if spec["key"] != "qa" else
       "for mod in ('qa_claims', 'verify_qa', 'correct_qa_numbers',\n"
       "            'drop_bad_qa_items', 'flag_qa_verification'):\n"
       "    m = __import__('datatools.' + mod, fromlist=[mod])\n"
       "    print(f'{mod:22s} {(m.__doc__ or chr(10)).strip().splitlines()[0][:78]}')"))

    md(f"## 3. 입력으로 무엇이 들어가는가\n\n**{spec['window']}**\n\n"
       f"입력 창은 태스크마다 다릅니다. 로더의 `WINDOWS` 표가 그것을 정하고, "
       f"레이더는 항상 20스캔이라 창의 길이가 바뀌면 샘플링 속도가 따라 바뀝니다 "
       f"— 인코더 입력 모양은 변하지 않습니다.")

    code("""
from training.instruct_data import WINDOWS, INSTANT_TASKS, WINDOW_TASKS
for v in VARIANTS:
    for name in (v, v + "_cot"):
        if name in WINDOWS:
            secs, hz, frames = WINDOWS[name]
            print(f"{name:26s} 창 {secs}초 · 레이더 {hz} Hz × 20스캔 · 비전 {frames}장")
        elif name in INSTANT_TASKS:
            print(f"{name:26s} 순간 — 비전 1장 · 레이더 20스캔/1초 · ego 1")
        else:
            print(f"{name:26s} 클립 전체 — 비전 20장 · 레이더 20스캔/20초")
""")

    # Everything below this point needs the 3.7 GB parquet and the raw archives.
    # This section needs neither: `notebooks/example_data` carries ten real
    # items per task with the frames and returns that produced them, so the
    # notebook opens on a machine that has only the repository.
    md("## 4. 예제 10건 — 원시 입력째로\n\n"
       "`notebooks/example_data/` 에 테스크마다 10건이 들어 있습니다. "
       "**parquet 도 원본 아카이브도 필요 없습니다.**\n\n"
       "| 경로 | 내용 |\n|---|---|\n"
       "| `gen_data/<task>.jsonl` | LLM 학습에 쓰이는 아이템 그대로 — "
       "instruction, ego, 정답, 근거 |\n"
       "| `raw/<task>/NN/frames/` | 그 아이템에 실제로 들어가는 프레임 |\n"
       "| `raw/<task>/NN/radar.npz` | 그 창의 레이더 반사점 (패딩 제거) |\n\n"
       "평문 변형과 CoT 변형은 같은 아이템입니다 — CoT 정답의 `answer` 필드가 "
       "평문 정답과 글자 그대로 같아서, 한 건이 둘을 모두 보여줍니다.")
    code("""
import glob
EX = os.path.abspath(os.path.join(os.getcwd(), "..", "example_data"))
examples = {}
for v in VARIANTS:
    path = os.path.join(EX, "gen_data", v + ".jsonl")
    examples[v] = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    print(f"{v:26s} {len(examples[v]):>2}건")
rows = [{"id": r["id"], "clip": r["clip_id"][:8], "프레임": r["n_frames"],
         "레이더 점": r["radar_points"], "정답 길이": len(r["target"]),
         "근거 길이": len(r["rationale"])}
        for v in VARIANTS for r in examples[v]]
pd.DataFrame(rows)
""")

    md("한 건을 통째로 봅니다. instruction 이 출력 형식을 고르고, 근거가 그 "
       "형식의 답으로 이어집니다.")
    code("""
pool = examples[VARIANTS[0]]
# 예제는 전방 레이더가 있는 클립에서만 뽑았다. 데이터 전체로는 177,891 클립 중
# 17,130건(9.6%)이 전방 레이더가 없고, 그 조건에서는 아래 그림이 빈 채로 나온다.
r = next((x for x in pool if x["radar_points"]), pool[0])
if not r["radar_points"]:
    print("!! 이 예제는 전방 레이더가 없는 클립입니다 (반사점 0개)")
print("=" * 96)
print(f"{r['id']}   clip {r['clip_id'][:8]}   {r['sensors']}")
print(f"창: {r['window']}")
print()
print("instruction:"); print(wrap(r["instruction"].replace(chr(10), " | ")))
print("ego:");         print(wrap(r["ego"]))
print("answer (GT):"); print(wrap(r["target"]))
print("rationale:");   print(wrap(r["rationale"]))
""")

    md("그 아이템에 실제로 들어가는 프레임과 레이더 반사점입니다. 레이더는 "
       "자차 기준 좌표(x 전방, y 좌)이고 색이 시선속도 — 정지 물체는 자차 속도의 "
       "음수로 모여 보입니다.")
    code("""
import matplotlib.pyplot as plt
from PIL import Image

d = os.path.join(EX, r["raw"])
paths = sorted(glob.glob(os.path.join(d, "frames", "*.jpg")))
z = np.load(os.path.join(d, "radar.npz"))
pts, ch = z["points"].astype(np.float32), [str(c) for c in z["channels"]]
print(f"프레임 {len(paths)}장 · 반사점 {len(pts)}개 · 채널 {ch}")

fig, axes = plt.subplots(1, len(paths) + 1, figsize=(3.1 * (len(paths) + 1), 2.6))
axes = np.atleast_1d(axes)
for ax, p in zip(axes, paths):
    ax.imshow(Image.open(p)); ax.set_title(os.path.basename(p), fontsize=8)
    ax.axis("off")
ax = axes[-1]
# 그림 안 글자는 ASCII 로 둔다. matplotlib 의 기본 폰트에 한글 글리프가 없어
# 라벨이 네모로 나오고, 폰트 등록을 요구하면 노트북이 기계를 가린다.
if len(pts):
    s = ax.scatter(pts[:, ch.index("y")], pts[:, ch.index("x")], s=2,
                   c=pts[:, ch.index("radial_velocity")], cmap="coolwarm")
    ax.set_title(f"radar: {r['radar_scans']} scans, {len(pts)} pts", fontsize=8)
    ax.invert_xaxis(); fig.colorbar(s, ax=ax, label="radial m/s")
else:
    ax.text(0.5, 0.5, "no returns", ha="center", va="center")
    ax.set_title("clip has no forward radar", fontsize=8)
ax.set_xlabel("y left (m)"); ax.set_ylabel("x forward (m)")
plt.tight_layout(); plt.show()
""")

    md("## 5. 빌드된 파일의 실제 아이템\n\n"
       "여기서부터는 parquet 이 있는 기계에서만 돕니다. 위의 예제가 빌드 전체와 "
       "같은 규칙으로 만들어졌는지 확인하는 절입니다.")
    code("""
built = pd.read_parquet(ITEMS)
for v in VARIANTS:
    sub = built[built.task == v]
    if sub.empty:
        print(f"{v}: 파일에 없음"); continue
    r = sub.iloc[0]
    print("=" * 96)
    print(f"{v}   clip {r.clip_id[:8]}  frame {r.frame} (t={r.frame-1}s)  split {r.split}")
    print("Q:"); print(wrap(r.prompt))
    print("A:"); print(wrap(r.target))
""" if spec["key"] != "qa" else """
from training.instruct_data import load_items
items = load_items(("qa",), "train")[:1] + load_items(("qa_cot",), "train")[:1]
for it in items:
    print("=" * 96)
    print(f"{it['task']}   clip {it['clip_id'][:8]}  frame {it['frame']}")
    print("Q:"); print(wrap(it["prompt"].replace(chr(10), " | ")))
    print("A:"); print(wrap(it["target"]))
""")

    md("## 6. CoT — 근거가 답을 만드는가\n\n"
       "`_cot` 변형은 `{\"rationale\": ..., \"answer\": ...}` 입니다. "
       "**근거를 따라가면 답이 나와야** 합니다. 나오지 않으면 그 사슬은 잘못된 "
       "것이고, 보상을 걸면 모델이 그 잘못된 사슬을 배웁니다.")
    code("""
for v in VARIANTS:
    name = v + "_cot"
    sub = built[built.task == name] if 'built' in dir() else None
    if sub is None or sub.empty:
        continue
    r = sub.iloc[0]
    d = json.loads(r.target)
    print("=" * 96); print(name)
    print("R:"); print(wrap(d["rationale"]))
    print("A:"); print(wrap(d["answer"]))
""" if spec["key"] != "qa" else """
from training.instruct_data import load_items
it = load_items(("qa_cot",), "train")[0]
d = json.loads(it["target"])
print("R:"); print(wrap(d["rationale"]))
print("A:"); print(wrap(d["answer"]))
""")

    md("## 7. 보상\n\n평가 채점기에서 유도했습니다. 정답을 그대로 넣으면 1.0 이 "
       "나와야 하고, 내용을 망가뜨리면 떨어져야 합니다.")
    code("""
import re
from training.task_scorers import reward_for

def wreck(text):
    \"\"\"형식은 두고 숫자만 2배로.\"\"\"
    return re.sub(r"\\d+(?:\\.\\d+)?",
                  lambda m: str(round(float(m.group()) * 2, 1)), text)

rows = []
for v in VARIANTS:
    for name in (v, v + "_cot"):
        fn = reward_for(name)
        if fn is None:
            continue
        sub = built[built.task == name] if 'built' in dir() else None
        if sub is None or sub.empty:
            continue
        t = sub.iloc[0].target
        rows.append({"task": name, "reward": fn.__name__,
                     "정답": round(fn(t, t), 3),
                     "숫자 2배": round(fn(wreck(t), t), 3)})
pd.DataFrame(rows)
""" if spec["key"] != "qa" else """
from training.task_scorers import reward_for
from training.instruct_data import load_items
rows = []
for name in ("qa", "qa_cot"):
    fn = reward_for(name)
    t = load_items((name,), "train")[0]["target"]
    wrong = "A" if "A" not in t[:3] else "B"
    rows.append({"task": name, "reward": fn.__name__,
                 "정답": round(fn(t, t), 3),
                 "다른 글자": round(fn(wrong, t), 3)})
pd.DataFrame(rows)
""")

    md("## 8. 데이터 양\n\n`val` 은 `train` 에 합쳐져 있습니다 — 클립 분할이 "
       "train 86,607 / val 54,163 / test 37,121 인데, 모델 선택은 `test` 에서 "
       "하므로 검증용 3분의 1이 쓰이지 않고 있었습니다.")
    code("""
from collections import Counter
from training.instruct_data import load_items
names = [v for v in VARIANTS] + [v + "_cot" for v in VARIANTS]
rows = []
for split in ("train", "test"):
    c = Counter(i["task"] for i in load_items(tuple(names), split))
    for n in names:
        rows.append({"task": n, "split": split, "items": c.get(n, 0)})
pd.DataFrame(rows).pivot(index="task", columns="split", values="items")
""")

    if spec["notes"]:
        md("## 9. 이 태스크에서 내린 결정과 근거\n\n" +
           "\n\n".join(f"**{a}**\n\n{b}" for a, b in spec["notes"]))

    nb = {"cells": CELLS,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{spec['no']}_{spec['key']}.ipynb")
    with open(path, "w") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    return path, len(CELLS)


def main():
    for spec in TASKS:
        path, n = notebook(spec)
        print(f"wrote {os.path.relpath(path, HERE)}  ({n} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
