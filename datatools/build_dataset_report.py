#!/usr/bin/env python3
"""The redefined dataset, as a PDF: what changed, why, and what was measured.

Every task in this build was rebuilt around one question -- can the answer be
reached from the input? Several could not, and the measurements that showed it
are the reason the definitions changed. This document records those numbers
beside the decisions they drove, so the design can be argued with rather than
taken on trust.

Counts are read from `runs/01_data_prep/dataset_summary.json`, which
`build_dataset_report` regenerates from the built artefacts.

    python -m datatools.build_dataset_report --out RaVL_dataset_v2.pdf
"""

import argparse
import glob
import json
import os
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from .build_report import register_fonts, styles, table
from . import paths

SUMMARY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "runs", "01_data_prep", "dataset_summary.json")


def summary():
    """Counts from the built data; rebuilt if the cache is missing."""
    if os.path.exists(SUMMARY):
        return json.load(open(SUMMARY))
    import pandas as pd
    path = os.path.join(paths.COMMON_DIR, "instruct_items_tasks01_06.parquet")
    d = pd.read_parquet(path, columns=["task", "frame", "split"])
    piv = d.pivot_table(index="task", columns="split", aggfunc="size", fill_value=0)
    out = {"frame_items": {"rows": len(d), "bytes": os.path.getsize(path)},
           "by_task": {t: {k: int(v) for k, v in r.items()} for t, r in piv.iterrows()},
           "anchors": {t: sorted(int(x) - 1 for x in g.frame.unique())
                       for t, g in d.groupby("task")}}
    qa_root = os.path.join(paths.SPLIT_ROOT, "10_radar_vision_qa")
    for name in ("qa_train", "qa_gt"):
        n_q, ids = 0, set()
        for f in glob.glob(os.path.join(qa_root, name, "*.json")):
            doc = json.load(open(f))
            ids.add(doc["clip_id"])
            n_q += len(doc.get("qa", []))
        out[name] = {"clips": len(ids), "questions": n_q}
    return out


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Nanum", 7.5)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(20 * mm, 12 * mm, "RaVL-AutoBench · 데이터셋 재정의 분석")
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"{doc.page}")
    canvas.restoreState()


def build(story, s):
    P = lambda t, st="body": story.append(Paragraph(t, s[st]))
    gap = lambda h=4: story.append(Spacer(1, h))
    M = lambda t: f"<font name='Mono'>{t}</font>"
    d = summary()
    n = d["by_task"]
    total = lambda t: sum(n.get(t, {}).values())

    # ---------------------------------------------------------------- cover
    P("데이터셋 재정의 분석 보고서", "title")
    P("RaVL-AutoBench · 태스크 1.1 ~ 6.2 재설계와 QA 5배 확장<br/>"
      "모든 결정에 측정값을 붙였습니다", "subtitle")
    gap(10)
    P("이번 재정의를 관통하는 질문은 하나입니다 — <b>정답이 입력으로 도달 가능한가.</b> "
      "여러 태스크가 그렇지 않았고, 그것을 드러낸 측정이 정의를 바꾼 이유입니다. "
      "이 문서는 그 수치를 결정 옆에 나란히 적어, 설계를 신뢰가 아니라 근거로 "
      "검토할 수 있게 합니다.", "small")
    gap(6)

    P("한 장 요약", "h1")
    story.append(table([
        ["항목", "이전", "이후"],
        ["태스크 정의", "11개, 단일 출력 형식",
         "10개 평문 + 11개 CoT, instruction으로 출력 형식 선택"],
        ["입력", "모든 태스크가 클립 전체 20초",
         "태스크마다 필요한 창 (1초 / 2초 / 5초 / 클립)"],
        ["rationale", "2개 태스크, 결과만 서술",
         "11개 태스크, 카메라와 레이더의 근거를 나란히 놓고 융합"],
        ["프레임 아이템", "9,076,730 행", f"{d['frame_items']['rows']:,} 행 "
         f"({d['frame_items']['bytes']/1e9:.1f} GB)"],
        ["학습 아이템", "-", f"{d['totals']['train']:,} (train + val 병합)"],
        ["평가 아이템", "-", f"{d['totals']['test']:,}"],
        ["QA 학습", "1,999 클립 / 39,158 문항",
         f"{d['qa_train']['clips']:,} 클립 / {d['qa_train']['questions']:,} 문항"],
        ["QA 평가", "sha1 로 고른 99 클립",
         f"사람 검수 {d['qa_gt']['clips']} 클립 / {d['qa_gt']['questions']:,} 문항"],
    ], [30 * mm, 52 * mm, 65 * mm]))

    # ------------------------------------------------------- design axes
    story.append(PageBreak())
    P("1. 재정의를 이끈 네 가지 축", "h1")
    P("태스크마다 같은 네 가지를 물었고, 답이 정의를 만들었습니다.", "small")
    gap(3)
    story.append(table([
        ["축", "묻는 것", "판단 기준"],
        ["입력 범위", "이 질문에 20초가 필요한가",
         "한 순간을 묻는 질문에 20초를 주면 95%가 무관한 문맥이고, 레이더는 "
         "20 Hz 원본에서 초당 1스캔만 남아 정작 필요한 정보가 버려짐"],
        ["출력 형식", "instruction 으로 나눌 수 있는가",
         "같은 입력·같은 근거에 표현만 다른 쌍이면 나눔. 좌표계마다 오차의 "
         "성질이 달라 하나만 쓰면 그 축이 가려짐"],
        ["앵커 간격", "인접 앵커가 서로 다른 문제인가",
         "정답 차이가 보상의 반점 기준을 넘어야 함. 넘지 않으면 같은 문항의 반복"],
        ["rationale", "근거가 답의 상류에 있는가",
         "근거를 따라가면 답이 나와야 함. 답을 다시 쓰는 것은 근거가 아님"],
    ], [24 * mm, 40 * mm, 83 * mm]))
    gap(5)

    P("1.1 입력을 잘라낸 근거", "h2")
    story.append(table([
        ["측정", "값", "결정"],
        ["레이더 원본 주기", "LRR·MRR 20 Hz, SRR 12.7 Hz",
         "20초 클립에 약 400스캔. 기존 방식은 초당 1개만 남겨 95%를 버림"],
        ["1초 창의 스캔 수", "LRR·MRR 20개, SRR 13개",
         "고정 격자로 20개를 뽑으면 어느 센서든 창이 같고 인코더 입력 모양도 유지"],
        ["비전 토큰", "20프레임 1,560 → 1프레임 78",
         "한 순간을 묻는 질문에 1,482 토큰 절약"],
    ], [34 * mm, 45 * mm, 68 * mm]))
    gap(4)

    P("1.2 앵커 간격의 근거", "h2")
    story.append(table([
        ["태스크", "인접 앵커 정답 차이", "보상 반점", "선택한 간격"],
        ["3.1 plan_ego", "1초 0.72 m / 2초 1.42 m", "1 m", "2초 (8개)"],
        ["6 motion_seg", "1초 간격에 상태가 바뀌는 물체 3.7%", "-", "2초 (9개)"],
        ["1 det_objects", "미래를 묻지 않아 지평선 제약 없음", "-", "3초 (6개)"],
        ["2 track_step", "롤아웃 이력 간격과 같아야 함", "-", "1초 (15개)"],
    ], [30 * mm, 52 * mm, 20 * mm, 25 * mm]))
    gap(3)
    P("2.x 만 1초 간격인 이유는 다릅니다. 축차 추적에서 <b>이력은 자기 출력</b>이므로 "
      "앵커 간격이 곧 이력 간격이 됩니다. 3초 간격 앵커에 1초 간격 이력을 쓰면 "
      "학습과 평가의 프롬프트 분포가 어긋납니다.", "small")

    # ------------------------------------------------------------ tasks
    story.append(PageBreak())
    P("2. 태스크별 정의", "h1")
    rows = [["번호", "태스크", "입력", "앵커", "학습", "평가"]]
    spec = [
        ("1.1", "det_objects_azdeg", "비전 1장 · 레이더 1초 · ego 1"),
        ("1.2", "det_objects_3dbbox", "〃"),
        ("2.1", "track_step_azdeg", "비전 5장 · 레이더 5초(4Hz) · 이력 4"),
        ("2.2", "track_step_bbox", "〃"),
        ("3.1.1", "plan_ego_xy", "비전 2장 · 레이더 2초(10Hz) · ego 20"),
        ("3.1.2", "plan_ego_control", "〃"),
        ("3.2.1", "agent_traj_azdeg", "〃"),
        ("3.2.2", "agent_traj_bbox", "〃"),
        ("6.1", "motion_seg_azdeg", "〃"),
        ("6.2", "motion_seg_bbox", "〃"),
    ]
    for no, task, inp in spec:
        anchors = d["anchors"].get(task, [])
        rows.append([no, task, inp,
                     f"{len(anchors)}개" if anchors else "--",
                     f"{n.get(task, {}).get('train', 0):,}",
                     f"{n.get(task, {}).get('test', 0):,}"])
    story.append(table(rows, [13 * mm, 37 * mm, 48 * mm, 14 * mm, 20 * mm, 18 * mm],
                       align_right=(4, 5)))
    gap(4)
    P("각 태스크에 같은 이름의 " + M("_cot") + " 변형이 있고, 근거를 앞에 붙인 "
      "JSON 형식입니다. 보상은 <b>answer 0.7 + rationale 0.3</b>입니다.", "small")
    gap(4)

    P("2.1 출력 형식을 둘로 나눈 이유", "h2")
    P("같은 물체를 세계 좌표로 가리키느냐 이미지 박스로 가리키느냐에 따라 "
      "<b>오차의 성질이 거리에 따라 반대가 됩니다.</b>")
    story.append(table([
        ["", "세계 좌표 (azdeg / xyz)", "이미지 박스 (bbox)"],
        ["매칭", "거리 2 m", "IoU 0.3"],
        ["먼 물체", "5 m 오차는 71 m 대비 7% — 관대",
         "71 m 사람의 박스는 폭 5픽셀 — 몇 픽셀만 어긋나도 무너짐"],
        ["가까운 물체", "2 m 는 8 m 대비 25% — 엄격",
         "8 m 차의 박스는 화면의 30% — 조금 어긋나도 IoU 유지"],
        ["표현 못 하는 것", "-", "카메라 시야 밖 물체는 답에 넣을 수 없음"],
    ], [22 * mm, 60 * mm, 65 * mm]))

    # -------------------------------------------------------- what changed
    story.append(PageBreak())
    P("3. 정의가 바뀐 지점과 그 근거", "h1")
    story.append(table([
        ["바뀐 것", "측정", "이유"],
        ["1.x 에서 이동 표기 제거",
         "레이더가 보는 물체만 놓고도 그 순간의 도플러와 트랙 단위 moved 라벨의 "
         "일치율 85.8%",
         "입력이 1프레임 + 1초인데 라벨은 20초 트랙 변위. 일곱 중 하나가 "
         "원리적으로 도달 불가능한 정답이었고, 06 과도 중복"],
        ["2.x 에서도 이동 표기 제거",
         "-",
         "이력에 이동 표기가 있으면 자기 첫 추측을 복사해 점수를 받을 수 있음. "
         "롤아웃의 첫 스텝은 빈 이력이라 그 추측 자체가 근거 없음"],
        ["02 를 요약형에서 축차형으로",
         "전방 트랙의 섹터 체류 중앙값 1.2초, 20초 내내 남는 트랙 0개",
         "클립 전체를 한 번에 요약하는 것은 tracking 이 아님. 이력을 주고 "
         "한 순간을 묻는 표준 정식화로 교체"],
        ["03.2 에 섹터 이탈 명시",
         "지목한 물체가 +3s 까지 남는 비율 51.4%",
         "절반이 사라지는데 답을 잘라내면 짧은 답이 항상 안전하다고 학습됨"],
        ["03.2 보상 조임 (5→2 m, 10→5°)",
         "시선속도 등속 외삽만으로 5 m 이내가 91.7%",
         "기존 기준으로는 레이더를 읽는 모델과 외삽하는 모델이 구분되지 않음"],
        ["06 라벨을 순간 속도로",
         "도플러와의 일치율 85.8% → 88.6%",
         "트랙 전체 변위는 2초 창으로 확립할 수 없는 사실"],
        ["06 을 두 잔차 융합으로",
         "카메라 각도 잔차 정지 0.24° / 이동 2.31°, 도플러가 놓친 이동 물체의 "
         "61.4% 를 카메라가 잡음",
         "도플러는 횡방향에, 방위각은 정면 접근에 눈이 멀어 서로의 사각을 메움"],
    ], [30 * mm, 52 * mm, 65 * mm]))

    # ------------------------------------------------------------ fusion
    story.append(PageBreak())
    P("4. rationale — 두 센서의 근거를 융합", "h1")
    P("이전 rationale 은 결과만 말했습니다. 지금은 <b>각 센서가 무엇을 말하고 "
      "무엇을 못 보는지</b>를 물체마다 적고, 답이 그것을 융합합니다.")
    gap(3)
    P("4.1 두 센서의 실측 상보성", "h2")
    story.append(table([
        ["", "커버리지", "거리", "방위각", "횡방향 운동"],
        ["카메라 (박스)", "100%", "없음", "오차 p50 0.94°",
         "잔차 p50 2.31° (이동) / 0.24° (정지)"],
        ["레이더 (반사점)", "55%", "오차 p50 0.70 m", "오차 p50 0.74°",
         "시선에 수직이면 거의 못 봄"],
    ], [26 * mm, 20 * mm, 26 * mm, 25 * mm, 50 * mm]))
    gap(3)
    P("교과서적인 \"카메라=각도, 레이더=거리\"는 이 데이터에 맞지 않습니다. "
      "imaging LRR 이라 각도 해상도가 카메라와 비슷합니다. 실제 상보성은 "
      "<b>커버리지 대 거리</b>(탐지)이고 <b>횡방향 대 시선방향</b>(운동)입니다. "
      "rationale 은 그것을 일반론으로 주장하지 않고 물체마다 수치로 적습니다.", "small")
    gap(4)

    P("4.2 06 의 rationale 이 보여주는 계산", "h2")
    story.append(Paragraph(
        "The ego is travelling at 10.7 m/s, yaw +0.0 deg/s.<br/>"
        "automobile at 17 m az +12 deg: radar expects -10.4 m/s if stationary, "
        "measures -8.0 (residual 2.4), camera expects az +15 deg, sees +12 "
        "(residual 2.2 deg);<br/>"
        "automobile at 46 m az -0 deg: radar residual 7.0, camera residual 0.0;<br/>"
        "automobile at 37 m az -30 deg: no radar return, camera residual 0.6.<br/>"
        "A radial residual above 1 m/s or a bearing residual above 1 deg means "
        "the object is moving.", s["code"]))
    P("자차 속도와 요레이트가 <b>두 예상값의 출발점</b>입니다. 속도를 물체 방향으로 "
      "투영하면 정지 물체가 보일 시선속도가 나오고, 속도와 요레이트가 정지 물체가 "
      "드리프트할 방위각을 줍니다. 측정값과의 차이가 각 센서의 잔차이고, 둘 중 "
      "하나만 커도 이동입니다. 두 번째 물체는 정면(방위각 −0°)이라 카메라 잔차가 "
      "0인데 레이더가 7.0으로 잡고, 세 번째는 반사점이 없어 카메라만 판정합니다.")
    gap(4)
    P("4.3 rationale 을 붙이지 않은 태스크", "h2")
    P("09 retrieval 과 11 description 은 <b>답 자체가 측정 사실의 나열</b>이라 "
      "근거를 앞에 붙이면 같은 문장이 두 번 나옵니다. " + M("radar_probe") + "도 "
      "같은 이유입니다 — 수치 자체가 답입니다.")

    # --------------------------------------------------------------- QA
    story.append(PageBreak())
    P("5. QA 확장과 자체 검증", "h1")
    story.append(table([
        ["", "클립", "문항", "성격"],
        ["qa (기존)", "1,999", "39,158", "수정 완료본"],
        ["qa_8000 (신규)", "8,000", "160,032 → 156,716", "검증·수정 적용"],
        ["agent_speed 추가 후 재검증", "-", "-",
         "검증 가능 34.0% → 38.2%, 주장 일치 99.3%"],
        ["→ qa_train (병합)", f"{d['qa_train']['clips']:,}",
         f"{d['qa_train']['questions']:,}", "학습"],
        ["qa_gt (신규)", f"{d['qa_gt']['clips']}", f"{d['qa_gt']['questions']:,}",
         "사람 검수, 평가 전용"],
    ], [34 * mm, 20 * mm, 34 * mm, 59 * mm]))
    gap(3)
    P("세 세트는 <b>클립이 하나도 겹치지 않습니다.</b> 병합에 중복 제거가 필요 "
      "없었고, 평가 세트가 학습 세트와 장면을 공유하지 않습니다.", "small")
    gap(4)

    P("5.1 자체 검증 절차", "h2")
    P("rationale 이 말하는 수치를 라벨에서 다시 계산해 대조했습니다. "
      "불일치는 성격에 따라 다르게 다뤘습니다 — rationale 은 사실 목록이 아니라 "
      "논증이라, 계산에 쓰이는 숫자를 바꾸면 다음 줄이 틀려집니다.")
    story.append(table([
        ["판정", "건수", "처리"],
        ["safe (서술 맥락)", "1,854 (32.0%)", "라벨값으로 교체"],
        ["arithmetic (계산에 쓰임)", "2,933 (50.6%)", "항목 제거"],
        ["answer_dependent (보기와 연동)", "471 (8.1%)", "항목 제거"],
        ["comparative (비교 방향)", "313 (5.4%)", "항목 제거"],
        ["range (범위 표현)", "223 (3.8%)", "항목 제거"],
    ], [46 * mm, 32 * mm, 40 * mm]))
    gap(3)
    P("결과: 3,334 문항(2.1%) 제거, 1,800 항목의 rationale 수정. 재검증 후 "
      "<b>주장 단위 일치율 100%</b>로 기존 qa 와 같은 기준이 됐습니다.")
    gap(4)
    P("5.3 추출기를 넓혔다 — agent_speed", "h2")
    P("검증되지 않은 문항을 표본으로 조사하니, <b>86% 가 숫자를 갖고 있는데 추출기가 "
      "청구항으로 잡지 못하는</b> 상태였습니다. 숫자가 아예 없는 것은 13.7% 뿐입니다. "
      "가장 큰 단일 형태가 타 객체의 속도로, 청구항이 안 잡힌 문항의 8.1% 를 "
      "차지했습니다 — 기존 패턴이 자차 언급을 요구해 "
      + M("the red automobile has a speed of 6.79 m/s") + " 같은 문장을 통째로 "
      "지나쳤습니다.")
    P("물체별 월드 속도를 트랙마다 중심차분으로 계산해 존재 검사로 대조했습니다. "
      "결과는 <b>1.5 m/s 이내 95.6%, 중앙값 오차 0.01 m/s</b> — QA 가 말하는 타 객체 "
      "속도가 라벨과 거의 정확히 일치합니다. 검증 가능 비율이 34.0% 에서 38.2% 로 "
      "올랐습니다.")
    gap(4)
    P("5.4 CoT 방출 규칙을 바꿨다", "h2")
    P("이전에는 " + M('verification == "agrees"') + " 인 문항만 CoT 로 냈습니다. "
      "그 판정은 \"숫자가 있었고 맞았다\"는 뜻이지 \"옳다\"는 뜻이 아닙니다. "
      "남은 62% 는 " + M('unchecked') + " 인데, 그 대부분은 틀린 것이 아니라 "
      "<b>대조할 숫자가 없는 정성 추론</b>입니다.")
    P("규칙을 " + M('status != "disagrees"') + " 로 바꿨습니다. 수정과 제거를 거친 "
      "지금 " + M("disagrees") + " 는 0건이므로 실질적으로 전부 방출입니다. "
      "수치 주장이 있는 문항만 CoT 로 쓰면 모델이 배우는 추론이 <b>산술에 편향</b>되고, "
      "QA 의 taxonomy 넷 중 Reasoning 계열을 통째로 못 배웁니다. "
      + M("verified") + " 플래그는 남겨 두어 나중에 가중치나 별도 평가에 쓸 수 "
      "있습니다.")
    gap(3)
    story.append(table([
        ["", "이전 (agrees 만)", "이후 (모순 아닌 전부)"],
        ["qa_cot 학습", "56,522", f"{n.get('qa_cot', {}).get('train', 0):,}"],
        ["qa_cot 평가", "0", f"{n.get('qa_cot', {}).get('test', 0):,}"],
    ], [30 * mm, 40 * mm, 45 * mm], align_right=(1, 2)))
    gap(4)

    P("5.2 검증기가 틀렸던 지점", "h2")
    P("사람 검수본의 거리 주장이 처음 <b>58.6%</b>로 나왔습니다. 원인은 데이터가 "
      "아니라 제 추출기였습니다.")
    story.append(Paragraph(
        "\"the ego vehicle's X-position is 349.69m, and the white suv's "
        "X-position is 364.68m. The longitudinal distance is the difference: "
        "364.68 - 349.69 = 14.99m.\"", s["code"]))
    P("추출기가 " + M("X-position is 349.69m") + "를 <b>거리 주장</b>으로 잡아, "
      "절대 좌표를 실제 거리(15 m)와 비교하며 209 m 오차를 만들었습니다. 정작 "
      "결론인 14.99 m 는 추출하지 못했습니다. 신규 세트가 절대 좌표를 명시하는 "
      "서술을 쓰는데 기존 qa 에는 없던 형태라 드러나지 않았습니다.")
    story.append(table([
        ["", "수정 전", "수정 후"],
        ["qa_gt 거리 주장", "58.6%", "82.1%"],
        ["qa_gt 항목 전체", "85.6%", "89.8%"],
        ["qa_8000 거리 주장", "92.4%", "98.8%"],
        ["qa_8000 항목 전체", "89.1%", "91.0%"],
    ], [40 * mm, 26 * mm, 26 * mm], align_right=(1, 2)))
    P("<b>사람 검수본이 자동 검증기보다 옳았습니다.</b> 검증기를 신뢰해 데이터를 "
      "깎아냈다면 멀쩡한 문항을 버렸을 것입니다. 사람 검수본에는 자동 수정을 "
      "적용하지 않았습니다 — 검수 결과를 자동 수정이 덮어쓰면 그 가치가 사라집니다.")

    # ---------------------------------------------------------- excluded
    story.append(PageBreak())
    P("6. 이번 빌드에서 제외한 것", "h1")
    story.append(table([
        ["태스크", "상태", "이유"],
        ["04 world_model", "보류", "재정의 대상. 코드는 남기고 태스크 목록에서만 제외"],
        ["05 depth_range", "보류", "1.x 와 상당 부분 중복. 재정의 필요"],
        ["09 retrieval", "보류", "태그 어휘가 고정 5개. 재정의 필요"],
        ["07 radar_transfer", "평가 전용",
         "학습에서 빼면 한 번도 학습하지 않은 태스크가 되어 더 깨끗한 "
         "일반화 측정이 됨"],
        ["11 description 의 context",
         "학습 제외",
         "\"Spain, at night, summer\" — 카메라로 알 수 없는 메타데이터. "
         "학습하면 화면을 보고 국가를 지어내도록 배움"],
        ["11 description 의 qa_summary", "학습 제외",
         "장면이 아니라 데이터셋 자체를 서술"],
    ], [30 * mm, 22 * mm, 95 * mm]))
    gap(5)

    P("7. 이번 작업에서 잡은 결함", "h1")
    P("정의를 고치는 과정에서 드러난, <b>에러 없이 조용히 잘못 동작하던</b> "
      "코드입니다.", "small")
    story.append(table([
        ["결함", "증상", "영향"],
        ["객체 파서가 세미콜론으로만 분리",
         "motion_seg 는 쉼표로 나열 — 4개 중 1개만 읽음",
         "그 태스크의 F1 이 답의 4분의 1로 계산되고 있었음"],
        ["reward_with_rationale 가 프롬프트를 버림",
         "추적 보상은 프롬프트의 이력에서 쓸 id 를 읽는데 전달되지 않음",
         "CoT 변형에서 id 일관성 채점이 통째로 생략"],
        ["QA 채점기가 A~D 만 인식",
         "보기는 A~E",
         "정답이 E 인 411 문항(21.2%) 전량 오답 처리, 상한 78.8%"],
        ["보기 글자 파싱이 접두어에 걸림",
         "\"Answer: E\" 에서 Answer 의 A 를 집음",
         "접두어를 붙이는 모델은 전부 오답"],
        ["radar_hits 의 track_id 가 문자열",
         "레코드는 int 로 비교",
         "radar_confirmed 가 전부 false"],
        ["옛 태스크 이름이 코드에 잔존",
         "평가 목록·생성 길이 상한·혼합 가중치가 새 이름에 안 붙음",
         "평가는 항목 0개로 넘어가고, 답은 잘리고, 가중치는 무시됨"],
    ], [40 * mm, 52 * mm, 55 * mm]))
    gap(5)

    P("8. 현재 상태와 다음", "h1")
    story.append(table([
        ["산출물", "규모"],
        ["instruct_items_tasks01_06.parquet",
         f"{d['frame_items']['rows']:,} 행 / {d['frame_items']['bytes']/1e9:.1f} GB"],
        ["qa_train / qa_gt",
         f"{d['qa_train']['clips']:,} 클립 {d['qa_train']['questions']:,} 문항 / "
         f"{d['qa_gt']['clips']} 클립 {d['qa_gt']['questions']:,} 문항"],
        ["radar_object_probes / radar_structure_probes", "평가 전용 프로브"],
    ], [66 * mm, 81 * mm]))
    gap(4)
    gap(3)
    P("<b>val 을 train 에 합쳤습니다.</b> 클립 분할이 train 86,607 / val 54,163 / "
      "test 37,121 인데, 모델 선택을 test 에서 하므로 검증용 3분의 1이 쓰이지 않고 "
      "있었습니다. test 는 그대로입니다.")
    gap(3)
    P("<b>다음에 해야 할 일</b>")
    P("① 04·05·09 재정의 — 이번 빌드에서 보류한 세 태스크")
    P("② 축차 추적 평가기 검증 — " + M("eval_tracking.py") + "를 실제 데이터로 확인. "
      "빈 이력으로 시작하는 롤아웃과 teacher forcing 의 차이가 곧 노출 편차의 크기")
    P("③ 이력 잡음 보정 — 잡음 없이 한 번 학습해 실제 오차 분포를 측정한 뒤 "
      "그 분포로 잡음 강도를 정함. 추측으로 정하지 않기 위함")
    P("④ 학습 — 위가 정리된 뒤")
    gap(5)
    P("보고서 생성: " + M("python -m datatools.build_dataset_report"), "small")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="RaVL_dataset_v2.pdf")
    args = ap.parse_args(argv)
    register_fonts()
    doc = SimpleDocTemplate(args.out, pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=20 * mm,
                            title="RaVL 데이터셋 재정의 분석")
    story = []
    build(story, styles())
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
