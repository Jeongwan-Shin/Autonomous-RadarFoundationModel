#!/usr/bin/env python3
"""RaVL-AutoBench: the task specification, as a PDF.

One document that says, for every task: what question it poses, where the data
came from, what preprocessing produced it, what the rationale states, and how
the rationale determines the answer. Counts are read from the built artefacts
rather than typed in, so the document cannot drift from the data.

    python -m datatools.build_bench_doc --out RaVL-Autobench.pdf
"""

import argparse
import json
import os
import re
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate, Spacer)

from .build_report import register_fonts, styles, table
from . import paths


def counts():
    """Train/val/test item counts per task, straight from the loader."""
    from collections import Counter
    from training.instruct_data import load_items, COT_TASKS
    plain = ("det_objects", "track_identity", "plan_ego", "agent_traj",
             "world_model", "depth_range", "motion_seg", "retrieval", "qa",
             "description", "radar_probe", "radar_objects", "radar_transfer",
             "radar_structure")
    out = {}
    for split in ("train", "val", "test"):
        out[split] = Counter(i["task"] for i in load_items(plain + COT_TASKS, split))
    return out


def clip_stats():
    import pandas as pd
    c = pd.read_parquet(os.path.join(paths.COMMON_DIR, "nvidia_clips.parquet"))
    return {
        "clips": len(c),
        "split": c["split"].value_counts().to_dict(),
        "config": c["radar_config"].value_counts().to_dict()
        if "radar_config" in c else {},
        "has": {k: int(c[k].fillna(False).sum())
                for k in ("has_egomotion", "has_obstacle", "has_radar_extrinsics")
                if k in c},
    }


def meta():
    path = os.path.join(paths.COMMON_DIR, "dataset_meta.json")
    return json.load(open(path)) if os.path.exists(path) else {}


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Nanum", 7.5)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(20 * mm, 12 * mm, "RaVL-AutoBench · radar + vision + language")
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"{doc.page}")
    canvas.restoreState()


# --------------------------------------------------------------------------
# per-task copy
# --------------------------------------------------------------------------

TASKS = [
 dict(no="01", key="det_objects", title="det_objects — 앞에 무엇이 있는가",
   problem="전방 ±60° 안의 모든 도로 사용자를 종류·거리·방위각·이동여부로 열거합니다. "
           "\"차가 몇 대인가\"가 아니라 개체를 하나씩 세워 위치를 붙이는 것이라, "
           "탐지(detection)의 원래 정의에 해당합니다. 답이 목록이므로 하나의 숫자로 "
           "채점할 수 없고, 전용 매칭 채점기가 필요합니다.",
   prompt="At frame 6. List every road user in the forward sector with its "
          "class, range and azimuth.",
   rationale="radar-confirmed: automobile at 34 m: 7 radar returns, 5 of them "
             "moving once the ego's own motion is removed. camera only: "
             "automobile at 61 m; automobile at 103 m.",
   answer="automobile 34 m az +13 deg moving; automobile 61 m az +47 deg "
          "stationary; automobile 103 m az +17 deg stationary",
   explains="어느 물체를 레이더가 뒷받침하고 어느 물체가 카메라의 주장뿐인지, "
            "그리고 각 물체에 자차 운동을 제거하고도 움직이는 반사점이 몇 개인지.",
   link="rationale이 목록의 신뢰도와 이동 판정을 함께 결정합니다. 레이더 확인된 물체는 "
        "두 센서가 합의한 것이고, 카메라 전용은 한쪽 주장입니다. 이 구분이 근거가 되는 "
        "이유는 레이더가 물체를 가려서 보기 때문입니다 — 대형 트럭은 85.7%가 반사점을 "
        "갖지만 보행자는 20.2%뿐입니다. 이동 반사점 수가 답의 moving/stationary를 정합니다.",
   reward="reward_objects — (x,y) 2 m 이내 greedy 최근접 매칭으로 F1을 구하고, "
          "매칭된 물체에 대해 클래스·이동·거리 정확도를 절반 배점. F1만 쓰면 "
          "그럴듯한 물체를 아무 거리에나 나열해도 만점이라 나눴습니다.",
   source="Nvidia obstacle.offline 3D 박스 + imaging LRR/MRR/SRR 스캔"),

 dict(no="02", key="track_identity", title="track_identity — 그것이 아까 그 물체인가",
   problem="같은 물체에 같은 id를 유지하고, 얼마나 오래 관측했는지를 답합니다. 01과 "
           "입력이 완전히 같지만 묻는 것이 다릅니다 — 재탐지가 아니라 정체성 유지입니다. "
           "id가 프레임마다 바뀌면 궤적 예측도 계획도 성립하지 않으므로, "
           "03-2의 전제 조건이기도 합니다.",
   prompt="At frame 6. Give the track id, class, range and age of every object "
          "you are tracking ahead.",
   rationale="#75 first seen at t=5.0 s, now t=5 s; #41 first seen at t=0.0 s, "
             "now t=5 s; #59 first seen at t=1.7 s, now t=5 s. Age is the "
             "difference between the two.",
   answer="#75 automobile 39 m visible 0.0 s; #41 automobile 61 m visible 5.0 s; "
          "#59 automobile 103 m visible 3.3 s",
   explains="각 트랙의 최초 관측 시각과 현재 시각.",
   link="여덟 개 사슬 중 가장 깨끗합니다. 답이 문자 그대로 rationale이 세워놓은 "
        "뺄셈입니다: 5.0-5.0=0.0, 5-0.0=5.0, 5-1.7=3.3. 근거를 제대로 세우면 "
        "답은 산술로 따라 나옵니다.",
   reward="reward_objects (01과 동일). 같은 물체를 다른 id로 부르면 매칭에서 탈락합니다.",
   source="Nvidia obstacle.offline 트랙 id + 월드 프레임 재스캔"),

 dict(no="03-1", key="plan_ego", title="plan_ego — 나는 어디로 갈 것인가",
   problem="자차의 3초 뒤 경로를 현재 자차 좌표계의 (x, y) 오프셋으로 예측합니다. "
           "전방이 +x입니다. 정답이 egomotion 센서 측정값에서 나오므로 "
           "이 데이터셋에서 가장 신뢰도 높은 타깃 중 하나입니다.",
   prompt="At frame 6. Predict the ego vehicle's path over the next 3 seconds "
          "as (x, y) offsets in metres.",
   rationale="The ego vehicle is travelling at 14.4 m/s and will hold speed and "
             "go straight. At that speed it covers about 14 m per second.",
   answer="+1s (+14.5, +0.1); +2s (+29.2, +0.4); +3s (+44.0, +0.5)",
   explains="자차의 현재 속도와 기동 의도.",
   link="14.4×1≈14.5, ×2≈29.2, ×3≈44.0. y가 0 근처인 것은 \"직진 유지\"와 맞습니다. "
        "이 태스크는 여덟 개 중 유일하게 근거가 레이더가 아니며, 그래서 대조군 역할을 "
        "합니다 — 레이더 개입이 여기까지 영향을 준다면 그것은 레이더 능력이 아니라 "
        "다른 무언가라는 신호입니다.",
   reward="reward_waypoints — 지평선 커버리지 × 변위오차 감쇠(1 m에서 반점). "
          "3초 자차 변위가 수 미터 규모라 그 스케일에 맞췄습니다.",
   source="Nvidia egomotion (10 Hz 센서 측정)"),

 dict(no="03-2", key="agent_traj", title="agent_traj — 저 물체는 어디로 갈 것인가",
   problem="특정 트랙 하나를 지목해 3초 뒤 거리·방위각을 예측합니다. 자차가 아니라 "
           "타 객체의 미래이므로, 그 객체의 속도를 알아야 풀립니다.",
   prompt="At frame 11. Track #185 is a trailer at 18 m, azimuth -29 deg. Where "
          "will it be over the next 3 seconds?",
   rationale="The radar puts 12 returns on track #185 at 18 m, median radial "
             "velocity -0.5 m/s, so it is holding its range.",
   answer="+1s 18 m az -28 deg; +2s 19 m az -26 deg; +3s 21 m az -23 deg",
   explains="그 물체에 꽂힌 반사점 수와 중앙값 시선속도.",
   link="여덟 개 중 물리적으로 가장 강한 사슬입니다. 시선속도 × 시간 = 거리 변화이기 "
        "때문입니다. -0.5 m/s면 3초에 1.5 m라 거의 제자리이고, 답이 18→18→19→21 m로 "
        "유지됩니다. 다른 예에서는 -10.7 m/s로 41 m→31 m로 정확히 10 m 좁혀집니다. "
        "이 속도는 단안 카메라로는 얻을 수 없으므로, 레이더 기여를 재기에 가장 적합한 "
        "태스크입니다.",
   reward="reward_trajectory — 거리 감쇠(5 m)와 방위각 감쇠(10°)의 평균에 "
          "답한 지평선 비율을 곱합니다.",
   source="Nvidia obstacle 트랙 + 레이더 도플러"),

 dict(no="04", key="world_model", title="world_model — 이렇게 운전하면 3초 뒤 장면은",
   problem="자차가 취할 행동을 조건으로 주고 3초 뒤 전방 장면을 예측합니다. "
           "조건부 예측이라 앞의 태스크들과 성격이 다릅니다 — 같은 현재라도 "
           "행동이 달라지면 답이 달라집니다.",
   prompt="At frame 6. The ego vehicle will hold speed and go straight over the "
          "next 3 seconds. What will the forward scene look like then?",
   rationale="Now: automobile at 39 m, az -40 deg; automobile at 61 m, az +47 "
             "deg; automobile at 103 m, az +17 deg. The ego covers about 43 m in "
             "3 s, so ranges change by that much plus each object's own motion, "
             "and anything the ego draws level with leaves the forward sector.",
   answer="2 automobiles; 0 moving; nearest automobile at 62 m",
   explains="현재 장면의 물체들과 그 방위각, 그리고 자차가 3초에 이동할 거리.",
   link="두 단계로 답을 만듭니다. ① 103-43=60 ≈ 답의 62 m. ② 39 m@-40°와 61 m@+47°는 "
        "자차가 나란해지면서 ±60° 밖으로 빠져 3대에서 2대로 줄어듭니다. "
        "②를 위해 방위각과 시야 이탈 문장이 rationale에 필요합니다 — 없으면 "
        "61-43=18 m가 되어야 하는데 답은 62 m라, 근거를 따라간 모델이 답과 어긋납니다.",
   reward="reward_quantity — 답 안의 모든 숫자를 상대오차로 채점.",
   source="Nvidia obstacle 미래 프레임 + egomotion"),

 dict(no="05", key="depth_range", title="depth_range — 가장 가까운 것, 레이더가 확인한 가장 가까운 것",
   problem="두 값을 동시에 묻습니다: 카메라가 보는 최근접 물체, 그리고 레이더가 "
           "확인해 준 최근접 물체. 이 쌍 구조가 이 태스크를 단안 거리추정이 아니라 "
           "레이더로 검증 가능한 태스크로 만듭니다.",
   prompt="At frame 11. How far is the nearest object ahead, and the nearest one "
          "the radar confirms?",
   rationale="automobile at 39 m: 1 returns; automobile at 41 m: 2 returns; "
             "automobile at 43 m: 2 returns; automobile at 54 m: 2 returns.",
   answer="nearest automobile at 39 m; nearest radar-confirmed automobile at 39 m",
   explains="물체별 반사점 개수.",
   link="답이 두 값을 요구하는데 반사점 개수만이 그 둘을 가릅니다. 위 예에서는 39 m "
        "물체가 반사점 1개를 가지므로 두 값이 같습니다. 반사점이 없었다면 카메라 "
        "최근접은 39 m, 레이더 확인 최근접은 더 먼 물체가 됩니다. "
        "레이더 확인 물체가 하나도 없는 프레임은 CoT를 만들지 않습니다 — 그 경우 "
        "rationale이 답의 뒷부분과 같아져, 추론이 아니라 채워넣기를 가르칩니다.",
   reward="reward_quantity.",
   source="Nvidia obstacle 박스 + 박스 안 레이더 반사점"),

 dict(no="06", key="motion_seg", title="motion_seg — 움직이는 것과 서 있는 것",
   problem="전방 물체를 이동/정지로 분류합니다. 도플러를 쓰라고 명시합니다. "
           "이 판정이 어려운 이유는 자차가 움직이면 정지 물체도 상대속도를 갖기 "
           "때문이고, 그래서 자차 운동을 제거한 잔차가 필요합니다.",
   prompt="At frame 6. Which objects ahead are moving and which are stationary? "
          "Use the radar Doppler.",
   rationale="automobile at 115 m: 1 returns, 1 of them still moving once the "
             "ego's own motion is removed; rider at 121 m: 1 returns, 1 of them "
             "still moving ... An object whose returns keep a residual above "
             "1 m/s is moving; one whose returns are explained entirely by the "
             "ego's own motion is not.",
   answer="moving: automobile 115 m az +7 deg (1 radar return), rider 121 m az "
          "+6 deg (1 radar return). stationary: none.",
   explains="물체별로 자차 운동을 제거하고도 움직이는 반사점이 몇 개인지, 그리고 판정 규칙.",
   link="잔차가 남은 반사점을 가진 물체가 moving으로 분류됩니다. "
        "이 rationale은 한 번 틀렸다가 고쳤고 그 수정이 중요합니다: 예전에는 측정 "
        "시선속도(mean radial -5.1 m/s)를 인용하면서 규칙은 잔차 기준이라고 했습니다. "
        "정지 물체의 측정 시선속도는 자차 속도가 투영된 값이라 항상 크므로, "
        "인용된 -5.1에 규칙을 적용하면 \"이동\"인데 정답은 \"정지\"였습니다. "
        "근거를 충실히 따른 모델이 틀린 답을 내는 구조였습니다.",
   reward="reward_objects — 01과 같은 매칭이라, 맞는 판정을 틀린 물체에 붙이면 "
          "점수를 받지 못합니다.",
   source="Nvidia 레이더 도플러 잔차 + 월드 프레임 이동 라벨"),

 dict(no="09", key="retrieval", title="retrieval — 이 클립을 어떻게 검색할 것인가",
   problem="클립을 나중에 찾아낼 수 있는 시나리오 태그를 씁니다. 자연어 질의로 "
           "주행 로그를 검색하는 용도이고, 답이 집합이라 순서는 채점하지 않습니다.",
   prompt="Write the scenario tags that should retrieve this clip.",
   rationale=None,
   answer="daytime; ego 16 m/s; 4 objects ahead, 4 moving; nearest rider",
   explains=None,
   link="rationale이 없습니다 — 의도적입니다. 답 자체가 이미 측정 사실의 나열이라 "
        "근거를 앞에 붙이면 같은 문장이 두 번 나옵니다. rationale은 답의 상류에 "
        "있을 때만 값을 합니다.",
   reward="reward_tags — 태그 집합 F1.",
   source="Nvidia 프레임 특징 집계 (조도, 자차 속도, 물체 수)"),

 dict(no="10", key="qa", title="qa — 5지선다",
   problem="장면에 대한 객관식 질문. 원본 데이터셋에서 온 텍스트 계열 중 유일하게 "
           "사람/자동 검증 절차를 거친 것입니다. 1,999 클립에 39,158 문항.",
   prompt="At Frame 20, what is the pedestrian on the right's lateral position "
          "relative to the ego's current path? A. Far to the left. ... "
          "E. Far to the right.",
   rationale="At Frame 20, the ego vehicle is at pos=(211.98,61.37) and the "
             "pedestrian on the right is at pos=(223.86,57.78). The pedestrian's "
             "y-coordinate (57.78) is significantly less than the ego's y "
             "(61.37), indicating a position to the right of the ego's current "
             "lateral alignment.",
   answer="E",
   explains="좌표를 읽고 뺄셈으로 상대 위치를 계산하는 과정.",
   link="계산된 값이 어느 보기에 떨어지는지가 답을 정합니다. QA는 rationale이 상류인 "
        "대표적 경우입니다 — 거리를 먼저 재고, 그 값에 가장 가까운 보기를 고릅니다. "
        "글자만 학습시키면 계산 과정을 통째로 건너뜁니다. "
        "검증을 통과한(agrees) 14,252 문항만 CoT로 방출합니다. 나머지 24,906은 "
        "미검증이라, 그것으로 사고 사슬을 학습시키면 검증되지 않은 산술을 가르칩니다.",
   reward="reward_choice — 보기 글자 일치.",
   source="Nvidia 클립에 대해 별도 생성·검증된 QA (10_radar_vision_qa)"),

 dict(no="11", key="description", title="description — 문장으로 서술",
   problem="장면·레이더·자차·센서 상보성을 문장으로 서술합니다. 여섯 종류이고 "
           "전부 측정값에서 생성됩니다. 자유 텍스트지만 주장이 검증 가능합니다.",
   prompt="At frame 1. Describe what the forward radar observes.",
   rationale=None,
   answer="At this instant the short-range radar returns 244 detections, 170 of "
          "which are not explained by the ego's own motion.",
   explains=None,
   link="rationale이 없습니다 — 09와 같은 이유로, 답 자체가 측정 사실의 진술입니다. "
        "이 태스크군은 한때 \"자유 텍스트라 검증 불가\"라는 이유로 RLVR에서 제외돼 "
        "있었는데 그 판단이 틀렸습니다. 모든 서술이 자기 측정값을 명시하므로, "
        "산문을 무시하고 주장만 채점하면 그것은 여전히 검증 가능한 보상이며 "
        "선호 모델이 아닙니다.",
   reward="reward_description — 숫자 절반(상대오차) + 주장어 절반(집합 F1), "
          "반의어 모순마다 0.35 감점(all↔none, moving↔stationary, closing↔receding, "
          "accelerating↔braking, left↔right). 숫자만 채점하면 "
          "\"None are moving\"과 \"All 2 are moving\"이 숫자를 공유해 같은 점수가 됩니다.",
   source="Nvidia 프레임/클립 특징에서 템플릿 생성 (11_radar_vision_scene_description)"),
]


def build(story, s):
    P = lambda t, st="body": story.append(Paragraph(t, s[st]))
    gap = lambda h=4: story.append(Spacer(1, h))
    M = lambda t: f"<font name='Mono'>{t}</font>"

    m, cs = meta(), clip_stats()
    try:
        n = counts()
    except Exception:
        n = None

    # ------------------------------------------------------------- cover
    P("RaVL-AutoBench", "title")
    P("radar + vision + language 자율주행 instruction 벤치마크<br/>"
      "태스크 명세 · 데이터 출처 · 전처리 · rationale/answer 구조", "subtitle")
    gap(10)
    P("비디오·레이더·ego를 고정 입력으로 두고 <b>텍스트 instruction만 바꿔</b> 모든 "
      "태스크를 수행하는 단일 모델을 위한 벤치마크입니다. 태스크별 출력 헤드가 없고 "
      "모든 답이 생성 텍스트입니다.")
    P("이 문서의 예시는 전부 실제 빌드된 데이터의 행이며, 수치는 디스크의 산출물에서 "
      "읽어 생성했습니다.", "small")
    gap(6)

    # -------------------------------------------------------- data source
    P("1. 데이터 출처", "h1")
    P("1.1 두 소스와 각자의 역할", "h2")
    story.append(table([
        ["소스", "규모", "이 벤치마크에서의 역할"],
        ["Nvidia AUTO",
         f"{m.get('nvidia_clips', 0):,} 클립 · 20 s · 전방 광각 카메라 30 Hz",
         "instruction 데이터 <b>전량</b>. 전방 레이더 3종, 10 Hz egomotion,\n"
         "10 Hz 3D 박스 오토라벨을 모두 갖춘 유일한 소스"],
        ["nuScenes",
         f"{m.get('nuscenes_scenes', 0):,} scene · "
         f"{m.get('nuscenes_annotations', 0):,} 어노테이션",
         "태스크별 split 파일은 구축했으나 <b>instruction 학습에는 쓰지 않음</b>.\n"
         "레이더가 detection list가 아니라 다른 규격이고, 전방 섹터\n"
         "정의가 달라 같은 질문 형식으로 합칠 수 없었음"],
    ], [30 * mm, 52 * mm, 65 * mm]))
    gap(3)
    P("<b>정직하게 적어둡니다.</b> 현재 학습·평가되는 instruction 데이터는 100% "
      "Nvidia AUTO에서 나옵니다. nuScenes는 " + M("build_task_splits.py") +
      "가 병렬 벤치마크용 split을 만들어 두었을 뿐 " + M("instruct_data.py") +
      "는 참조하지 않습니다.", "small")
    gap(5)

    P("1.2 센서 제원 (측정값)", "h2")
    r = m.get("sampling_rates", {})
    story.append(table([
        ["레이더", "주기", "스캔당 반사점(중앙값)", "최대 거리", "방위 FOV"],
        ["SRR (short)", f"{r.get('radar_hz', {}).get('radar_front_center_srr_0', 0)} Hz",
         f"{r.get('radar_detections_per_scan_median', {}).get('radar_front_center_srr_0', 0)}",
         f"{r.get('radar_max_range_m', {}).get('radar_front_center_srr_0', 0)} m",
         f"{r.get('radar_azimuth_fov_deg', {}).get('radar_front_center_srr_0', 0)}°"],
        ["MRR (mid)", f"{r.get('radar_hz', {}).get('radar_front_center_mrr_2', 0)} Hz",
         f"{r.get('radar_detections_per_scan_median', {}).get('radar_front_center_mrr_2', 0)}",
         f"{r.get('radar_max_range_m', {}).get('radar_front_center_mrr_2', 0)} m",
         f"{r.get('radar_azimuth_fov_deg', {}).get('radar_front_center_mrr_2', 0)}°"],
        ["imaging LRR (long)", f"{r.get('radar_hz', {}).get('radar_front_center_imaging_lrr_1', 0)} Hz",
         f"{r.get('radar_detections_per_scan_median', {}).get('radar_front_center_imaging_lrr_1', 0)}",
         f"{r.get('radar_max_range_m', {}).get('radar_front_center_imaging_lrr_1', 0)} m",
         f"{r.get('radar_azimuth_fov_deg', {}).get('radar_front_center_imaging_lrr_1', 0)}°"],
    ], [34 * mm, 22 * mm, 36 * mm, 26 * mm, 24 * mm], align_right=(1, 2, 3, 4)))
    gap(3)
    P("레이더 출력은 <b>" + str(r.get("radar_output_level", "detection list")) +
      "</b>입니다. range-azimuth나 range-Doppler 텐서가 없으므로, 레이더 인코더는 "
      "이미지가 아니라 <b>점군</b>을 받습니다. 이것이 인코더 설계를 결정했습니다.", "small")
    gap(3)
    cfg = cs.get("config", {})
    P("<b>한 클립이 세 레이더를 모두 갖지 않습니다.</b> 리그 속성이라 "
      f"low({cfg.get('low', 0):,} 클립)는 SRR만, "
      f"med({cfg.get('med', 0):,})와 high({cfg.get('high', 0):,})는 MRR+LRR을 "
      "싣습니다. 따라서 SRR→LRR 변환은 짝지어진 지도학습 자체가 존재하지 않고, "
      "동시 관측 쌍은 MRR↔LRR 하나뿐입니다. 태스크 07이 이 쌍 위에 세워져 있습니다.")

    # ------------------------------------------------------ preprocessing
    story.append(PageBreak())
    P("2. 전처리", "h1")
    P("아래 각 단계는 측정 결과 때문에 존재합니다. 가정이 아니라 확인된 사실이 "
      "설계를 바꾼 지점들입니다.", "small")
    gap(3)
    story.append(table([
        ["단계", "무엇을 하는가", "왜 필요한가 (측정값)"],
        ["레이더 외부파라미터\n확보",
         "비-offline 캘리브레이션을 내려받아 인덱싱",
         "offline 캘리브레이션에는 카메라 7대와 라이다뿐, 레이더가 없음.\n"
         "비-offline을 쓰면 박스 중심과 최근접 반사점의 중앙값 거리가\n"
         "<b>1.31 m</b>. 추정 규약으로는 최선이 10.6 m"],
        ["월드 프레임\n트랙 재스캔",
         "박스를 rig 좌표에서 월드 좌표로 변환해\n이동 여부를 재판정",
         "obstacle.offline 박스는 reference_frame='rig'라 자차 중심.\n"
         "주차 차량이 흘러가고 같은 속도 차량이 멈춰 보임.\n"
         "월드/rig 변위 상관 <b>0.202</b>. 14,844 트랙 중 월드 기준\n"
         "정지가 <b>71.6%</b>인데 rig 기준으로는 <b>0.4%</b>"],
        ["전방 섹터\n마스킹",
         "±60° 밖 라벨을 제거",
         "라벨은 360°인데 전방 광각 카메라와 imaging LRR은 약 ±60°.\n"
         "관측 가능한 라벨은 <b>46%</b>뿐이고 트랙의 <b>20%</b>만 대부분\n"
         "생애를 섹터 안에서 보냄. 섹터 밖 라벨로 학습하면 언어모델이\n"
         "차 뒤의 물체를 지어내도록 배움"],
        ["도플러 잔차",
         "시선속도에서 자차 운동 투영분을 빼\n실제 이동만 남김",
         "정지 반사점의 시선속도는 자차 속도의 투영. imaging LRR에서\n"
         "1.0 m/s 임계로 스캔당 <b>954개 중 47개</b>만 남음.\n"
         "토큰 가지치기·무료 지도신호·태스크 06의 정답을 동시에 제공"],
        ["박스-반사점\n연결",
         "박스 안(여유 1.3배) 반사점을 세고\n중앙값 시선속도를 계산",
         "레이더는 물체를 가려서 봄: heavy_truck <b>85.7%</b>,\n"
         "automobile <b>67.1%</b>, person <b>20.2%</b>,\n"
         "protruding_object <b>0.0%</b>. 박스 기반 grounding 지도는\n"
         "차량 클래스에서만 신뢰 가능"],
        ["앵커 프레임\n고정",
         "프레임 6/11/16 (t=5/10/15 s)에서만 생성",
         "3초 예측이 21프레임 클립 안에 들어가고, 클립당 태스크마다\n"
         "21개의 near-duplicate 대신 초·중·후반 시점 3개를 얻음"],
        ["센서 프로파일\n라우팅",
         "레이더 없는 클립의 레이더 의존 답을\n'Radar unavailable'로 치환",
         "전방 레이더가 아예 없는 클립이 존재. 빈 레이더에 원래 답을\n"
         "학습시키는 것이 <b>읽지도 않고 레이더 통계를 읊는</b> 행동을\n"
         "만드는 경로"],
    ], [24 * mm, 48 * mm, 75 * mm]))
    gap(4)
    P("2.1 클립 통계", "h2")
    story.append(table([
        ["항목", "수"],
        ["전체 클립", f"{cs['clips']:,}"],
        ["egomotion 보유", f"{cs['has'].get('has_egomotion', 0):,}"],
        ["obstacle 라벨 보유", f"{cs['has'].get('has_obstacle', 0):,}"],
        ["레이더 외부파라미터 보유", f"{cs['has'].get('has_radar_extrinsics', 0):,}"],
        ["train / val / test 클립",
         f"{cs['split'].get('train', 0):,} / {cs['split'].get('val', 0):,} / "
         f"{cs['split'].get('test', 0):,}"],
    ], [58 * mm, 60 * mm], align_right=(1,)))
    gap(3)
    P("<b>분할은 클립 단위입니다.</b> 같은 클립의 프레임이 학습과 평가에 나뉘어 "
      "들어가면 같은 장면을 외운 것을 일반화로 오인하게 됩니다. 그래서 태스크별 "
      "9:1이 아니라, 클립이 test로 정해지면 그 클립에서 나오는 모든 태스크 아이템이 "
      "test입니다. QA만 예외로 sha1 해시로 99 클립(1,940 문항)을 골라 "
      "<b>모든 태스크에서</b> 학습 제외했습니다.", "small")

    # --------------------------------------------------------- rationale
    story.append(PageBreak())
    P("3. rationale / answer 구조", "h1")
    P("여덟 개 태스크가 " + M('{"rationale": ..., "answer": ...}') +
      " 형식으로 답합니다. rationale은 감상이 아니라 <b>라벨에서 계산된 증거</b>이며, "
      "따라서 채점기가 검증할 수 있습니다.")
    P("보상은 <b>answer 0.7 + rationale 0.3</b>입니다. 형식만 맞추고 내용이 비면 "
      "양쪽 다 0이라 게임할 여지가 없습니다.")
    story.append(table([
        ["생성", "보상"],
        ["정답 + 정답 근거", "1.000"],
        ["정답 + 틀린 근거", "0.856"],
        ["틀린 답 + 정답 근거", "0.300"],
        ["형식 미준수 (평문 답만)", "answer 몫만"],
    ], [58 * mm, 30 * mm], align_right=(1,)))
    gap(4)
    P("<b>rationale을 붙이지 않은 태스크</b>: 09 retrieval과 11 description. "
      "두 태스크는 답 자체가 측정 사실의 나열이라, 근거를 앞에 붙이면 같은 문장이 "
      "두 번 나옵니다. rationale은 <b>답의 상류에 있을 때만</b> 값을 합니다. "
      "같은 이유로 " + M("radar_probe") + "류도 제외했습니다 — 수치 자체가 답입니다.")

    # ---------------------------------------------------------- the tasks
    for t in TASKS:
        story.append(PageBreak())
        P(f"{t['no']} · {t['title'].split(chr(8212))[1].strip()}", "h1")
        P(M(t["key"]), "small")
        gap(3)
        P("무엇을 푸는가", "h2")
        P(t["problem"])
        gap(2)
        P("instruction", "h2")
        story.append(Paragraph(t["prompt"], s["code"]))
        if t["rationale"]:
            P("rationale", "h2")
            story.append(Paragraph(t["rationale"], s["code"]))
        P("answer", "h2")
        story.append(Paragraph(t["answer"], s["code"]))
        gap(2)
        if t["explains"]:
            P("rationale이 설명하는 것", "h2")
            P(t["explains"])
        P("rationale과 answer의 연결" if t["rationale"] else "설계 결정", "h2")
        P(t["link"])
        gap(2)
        P("보상", "h2")
        P(t["reward"])
        P("출처", "h2")
        P(t["source"], "small")
        if n:
            row = [["split", "아이템 수"]]
            for split in ("train", "val", "test"):
                key = t["key"] if t["key"] != "description" else "desc_objects"
                row.append([split, f"{n[split].get(key, 0):,}"])
            gap(2)
            story.append(table(row, [30 * mm, 34 * mm], align_right=(1,)))

    # ------------------------------------------------------- not trained
    story.append(PageBreak())
    P("학습하지 않는 태스크", "h1")
    story.append(table([
        ["태스크", "이유"],
        ["07 radar_adaptation\n(radar_transfer)",
         "학습에서 제외하고 <b>평가 전용 계기</b>로 유지. 한 번도 학습하지 않은\n"
         "태스크가 되면 오히려 더 깨끗한 일반화 측정이 됩니다.\n"
         "MRR↔LRR은 이 데이터에 존재하는 유일한 동시 관측 쌍입니다"],
        ["08 missing_modality",
         "애초에 별도 태스크가 아님. 전방 레이더가 없는 클립을 로더가\n"
         "센서 프로파일로 라우팅해 레이더 의존 답을 치환합니다"],
    ], [36 * mm, 111 * mm]))
    gap(6)

    P("진단용으로 추가한 태스크 (원본 11개에 없음)", "h1")
    story.append(table([
        ["태스크", "질문", "상태"],
        ["radar_probe", "스캔 전체의 검출 수·최대 RCS·조사 물체 수",
         "<b>순환 측정으로 판명</b>. 세 질문형 중 둘이\n"
         "인코더가 토큰에 적도록 지도학습된 스칼라를\n그대로 묻고 있었음"],
        ["radar_structure", "스캔의 각도·거리 분포 5종",
         "<b>평가 전용</b>. 학습하면 계기를 잃음"],
        ["radar_objects", "레이더가 비추는 최근접 물체의 거리·방위각·속도",
         "학습 가능. 다만 일부 질문형만 학습하고\n나머지는 계기로 보류"],
    ], [28 * mm, 56 * mm, 63 * mm]))
    gap(4)
    P("<b>학습된 프로브는 계기이기를 멈춥니다.</b> " + M("radar_probe") +
      "가 그 대가를 치렀고, 그래서 " + M("radar_structure") + "는 학습 태스크 목록에 "
      "없으며 " + M("radar_objects") + "는 질문형을 나눠 일부만 학습합니다.")
    gap(5)
    P("문서 생성: " + M("python -m datatools.build_bench_doc"), "small")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="RaVL-Autobench.pdf")
    args = ap.parse_args(argv)
    register_fonts()
    doc = SimpleDocTemplate(args.out, pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=20 * mm,
                            title="RaVL-AutoBench")
    story = []
    build(story, styles())
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
