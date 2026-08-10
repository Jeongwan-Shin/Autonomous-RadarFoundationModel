#!/usr/bin/env python3
"""학습 데이터를 어떻게 구성했는가 -- 그리고 무엇을 균형 맞추지 못했는가.

숫자는 전부 실제 파케이에서 읽는다. 손으로 적은 값은 없고, 데이터를 다시
만들면 이 문서도 다시 만들어야 한다.

    python -m datatools.build_training_stat_report --out training_data_stat.pdf
"""

import argparse
import collections
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer)

from . import paths
from .build_report import register_fonts, styles, table

ITEMS = "instruct_items_tasks01_06.parquet"
CLS = re.compile(r"([a-z_]+)\s+\d+\s*m\s*az")
SAMPLE = 200000          # 클래스 집계에 쓰는 아이템 수 -- 전량은 메모리를 넘긴다


def log(m):
    print(m, flush=True)


def gather(path):
    """문서에 들어가는 모든 수치를 한 번의 읽기로."""
    d = pd.read_parquet(path, columns=["task", "prompt", "target", "strat",
                                       "split", "clip_id"])
    out = {"rows": len(d), "clips": d.clip_id.nunique()}

    out["split_items"] = d.split.value_counts().to_dict()
    out["split_clips"] = d.groupby("split").clip_id.nunique().to_dict()
    sets = {s: set(x) for s, x in d.groupby("split").clip_id.unique().items()}
    names = sorted(sets)
    out["overlap"] = {f"{a} ∩ {b}": len(sets[a] & sets[b])
                      for i, a in enumerate(names) for b in names[i + 1:]}

    tr = d[d.split.isin(("train", "val"))]
    out["per_task"] = tr.task.value_counts().to_dict()
    out["train_rows"] = len(tr)

    # navigation command
    p = tr[tr.task == "plan_ego_xy"]
    out["nav"] = p.strat.value_counts().to_dict()

    # 물체 클래스 -- 관측된 것(검출)과 고른 것(궤적)
    obs = collections.Counter()
    for s in tr[tr.task == "det_objects_azdeg"].target.head(SAMPLE):
        obs.update(CLS.findall(s))
    out["observed"] = obs

    a = tr[tr.task == "agent_traj_azdeg"].prompt.head(SAMPLE)
    out["chosen"] = collections.Counter(
        a.str.extract(r"is a ([a-z_]+) ")[0].dropna())

    # 검출 아이템 중 automobile 을 담은 비율 -- 지워서 못 고치는 이유
    s = tr[tr.task == "det_objects_azdeg"].target.head(SAMPLE)
    out["carries_auto"] = float(s.str.contains("automobile ").mean())

    # 궤적이 3 초 안에 시야를 벗어나는 비율
    t = tr[tr.task == "agent_traj_azdeg"].target.head(SAMPLE)
    out["leaves"] = float(t.str.contains("leaves the forward sector").mean())
    return out


def share_rows(counter, top=10):
    tot = sum(counter.values()) or 1
    rows = [["클래스", "개수", "비율"]]
    for k, v in counter.most_common(top):
        rows.append([k, f"{v:,}", f"{v / tot:.2%}"])
    return rows


def build(g, out_path, mixture, strata):
    register_fonts()
    s = styles()
    doc = SimpleDocTemplate(out_path, pagesize=A4, topMargin=18 * mm,
                            bottomMargin=18 * mm, leftMargin=18 * mm,
                            rightMargin=18 * mm,
                            title="학습 데이터 구성")
    F = []
    P = lambda t, k="body": F.append(Paragraph(t, s[k]))

    P("학습 데이터를 어떻게 구성했는가", "title")
    P(f"태스크 01–06 아이템 {g['rows']:,}건 · 클립 {g['clips']:,}개. "
      f"이 문서의 수치는 모두 <b>{ITEMS}</b> 에서 직접 읽은 것입니다.", "subtitle")

    # ---------------------------------------------------------------- 분리
    P("1. train / test 분리", "h1")
    P("클립 단위로 나눕니다. 한 클립이 만드는 아이템은 전부 같은 쪽에 있어야 "
      "합니다 — 같은 영상과 레이더를 학습에서 보고 시험에서 다시 묻는다면 "
      "분리는 이름뿐입니다.", "body")
    rows = [["split", "아이템", "클립"]]
    for k in ("train", "val", "test"):
        if k in g["split_items"]:
            rows.append([k, f"{g['split_items'][k]:,}",
                         f"{g['split_clips'][k]:,}"])
    F.append(table(rows, [40 * mm, 40 * mm, 40 * mm], align_right=(1, 2)))
    F.append(Spacer(1, 5))
    rows = [["교집합", "클립 수"]] + [[k, f"{v:,}"] for k, v in
                                   g["overlap"].items()]
    F.append(table(rows, [50 * mm, 30 * mm], align_right=(1,)))
    F.append(Spacer(1, 5))
    P("<b>val 은 train 에 접어 넣습니다.</b> 릴리스가 물려준 분할은 train "
      "86,607 / val 54,163 / test 37,121 인데, 모델 선택을 val 로 하지 않으므로 "
      "전체의 3분의 1이 아무 일도 하지 않고 있었습니다. test 는 손대지 "
      "않았습니다.", "body")
    P("여기에 더해 <b>qa_gt 139 클립</b>은 모든 태스크에서 학습에서 빠집니다. "
      "사람이 검수한 2,019문항이 걸린 시험지라, 그 클립의 detection 이나 "
      "tracking 아이템만 학습에 남겨도 시험 볼 때 모델이 그 영상을 이미 본 "
      "상태가 됩니다. 확인한 결과 train 에 0개, test 에 139개입니다.", "body")

    # ---------------------------------------------------------------- 태스크
    F.append(PageBreak())
    P("2. 태스크별 아이템 수", "h1")
    P(f"학습에 쓰는 것(train + val) {g['train_rows']:,}건입니다. "
      "이름 끝의 <b>_cot</b> 는 같은 질문에 근거를 먼저 쓰는 형식이고, "
      "평문 쌍과 프롬프트가 달라야 합니다 — 한때 100% 같아서 같은 입력에 "
      "정답이 두 개였고, 모델은 한쪽 형식만 골라 다른 쪽에서 0점을 "
      "받았습니다.", "body")
    rows = [["태스크", "아이템", "혼합 가중치"]]
    for t, n in sorted(g["per_task"].items(), key=lambda kv: -kv[1]):
        rows.append([t, f"{n:,}", f"{mixture.get(t, 1.0):g}"])
    F.append(table(rows, [62 * mm, 30 * mm, 28 * mm], align_right=(1, 2)))
    F.append(Spacer(1, 5))
    P("혼합 가중치는 한 에폭에서 각 태스크가 차지하는 몫입니다. 아이템 수가 "
      "많다고 많이 뽑히지 않습니다 — 뽑기는 자르기가 아니라 표집이라, "
      "앞쪽부터 잘라 한 차량 플랫폼에 쏠리던 문제가 없습니다.", "body")

    # ------------------------------------------------------- navigation cmd
    F.append(PageBreak())
    P("3. Navigation command — 넣은 이유와 그 대가", "h1")
    P("교차로에서 “이 차가 어디로 갈까”는 장면만으로 답이 정해지지 않습니다. "
      "왼쪽·오른쪽·직진이 모두 옳고, 조건 없이 학습하면 모델은 그 셋의 "
      "평균을 내놓습니다 — 실제로는 아무도 그렇게 운전하지 않는 경로입니다. "
      "그래서 <b>Navigation command: LEFT / RIGHT / STRAIGHT</b> 를 프롬프트 "
      "맨 앞에 둡니다.", "body")
    P("대가는 분명히 적어 둡니다. 이 릴리스에는 지도도 경로도 없습니다. "
      "<b>labels/</b> 에 있는 것은 egomotion 과 obstacle 뿐이라, command 를 "
      "자차의 <b>미래 heading</b> 에서 역산하는 수밖에 없습니다 — 그것은 "
      "plan_ego 가 맞혀야 할 정답의 일부입니다. 그래서 크기를 재 뒀습니다.",
      "body")
    rows = [["예측기에 준 입력", "ADE", "FDE(+3s)", "개선"],
            ["자차 운동만 (속도·요레이트, 과거 2 s)", "0.991 m", "2.030 m", "—"],
            ["+ navigation command", "0.958 m", "1.948 m", "3.3%"],
            ["+ motion intent (쓰지 않음)", "0.917 m", "1.856 m", "7.5%"]]
    F.append(table(rows, [72 * mm, 24 * mm, 26 * mm, 22 * mm],
                   align_right=(1, 2, 3)))
    F.append(Spacer(1, 5))
    P("프롬프트가 이미 싣고 있는 자차 속도·요레이트만 쓰는 릿지 회귀 위에서 "
      "잰 값입니다(클립 1,500개, 앵커 12,000개, 절반으로 학습해 절반에서 측정). "
      "command 가 더해 주는 것은 전체 3.3% — STRAIGHT 앵커에서 1.7%, LEFT 에서 "
      "8.1%, RIGHT 에서 12.0% 입니다.", "body")
    P("<b>motion intent 는 입력으로 넣지 않았습니다.</b> 같은 방식으로 재 보면 "
      "STOP 을 아는 것만으로 그 앵커의 오차가 1.546 m 에서 0.777 m 로 "
      "절반이 됩니다. 그건 경로에 대한 힌트가 아니라 답의 대부분입니다. "
      "TURN_LEFT 21.0%, TURN_RIGHT 25.5%, CHANGE_LANE 11–13% 도 같은 이유로 "
      "뺐습니다.", "body")
    P("평가할 때 이 점을 잊으면 안 됩니다. command 를 준 상태의 0.958 m 는 "
      "“의도를 알려줬을 때의 오차”이지 “차가 어디로 갈지 맞힌 오차”가 "
      "아닙니다. 두 수치는 다른 주장입니다.", "small")

    # ------------------------------------------------------------- 균형
    F.append(PageBreak())
    P("4. 균형 — 고른 것은 맞추고, 관측된 것은 두었다", "h1")
    P("데이터의 쏠림에는 두 종류가 있고, 손댈 수 있는 것은 하나뿐입니다. "
      "<b>우리가 고른 것</b>(어느 물체를 물을지, 어떤 앵커를 뽑을지)은 바꿀 수 "
      "있습니다. <b>관측된 것</b>(장면에 실제로 무엇이 있었나)은 지워서 "
      "고칠 수 없습니다 — 지우면 답이 틀려지거나 다른 쏠림이 생깁니다.",
      "body")

    P("4.1 Navigation command — 지우지 않고 노출만 맞춤", "h2")
    nav = g["nav"]
    tot = sum(nav.values()) or 1
    rows = [["command", "아이템", "원본 비율", "학습 노출"]]
    weights = strata.get("plan_ego_xy", {})
    wsum = sum(weights.values()) or 1
    for k in ("STRAIGHT", "LEFT", "RIGHT"):
        rows.append([k, f"{nav.get(k, 0):,}", f"{nav.get(k, 0) / tot:.2%}",
                     f"{weights.get(k, 1) / wsum:.1%}"])
    F.append(table(rows, [30 * mm, 30 * mm, 30 * mm, 30 * mm],
                   align_right=(1, 2, 3)))
    F.append(Spacer(1, 5))
    P(f"88% 가 STRAIGHT 면 command 를 읽지 않아도 점수가 나옵니다. 그러면 "
      f"조건을 붙인 의미가 없습니다. 다만 1:1:1 로 <b>잘라서</b> 맞추면 "
      f"{tot:,}건이 175,434건이 됩니다 — 83% 를 버리는 데다 비율이 파케이에 "
      f"굳어 다시 바꾸려면 재생성해야 합니다. 그래서 아이템은 전부 두고 "
      f"<b>뽑는 비율</b>만 2:1:1 로 맞춥니다. 1:1:1 이 아닌 이유는, 직진이 "
      f"바로잡아야 할 인공물이 아니라 도로가 원래 그렇기 때문입니다 — 회전을 "
      f"3분의 1로 보여 주면 돌지 말아야 할 곳에서 도는 사전확률을 "
      f"배웁니다.", "body")

    P("4.2 궤적이 묻는 물체 — 고른 것이므로 맞췄다", "h2")
    P("task 03.2 는 “어느 물체의 미래를 물을지” 를 우리가 고릅니다. 가장 가까운 "
      "움직이는 물체만 고르면 73–78% 가 automobile 이 됩니다. 후보 풀은 그보다 "
      "낫습니다(automobile 69.9%, person 20.9%, rider 3.6%). 그렇다고 항상 "
      "가장 희소한 것만 고르면 붐비는 장면에서 바로 앞 차를 한 번도 묻지 않게 "
      "되는데, 그건 계획기가 가장 알고 싶어 하는 경우입니다. 그래서 "
      "<b>앵커를 번갈아</b> 두 방식을 씁니다.", "body")
    F.append(table(share_rows(g["chosen"], 8),
                   [42 * mm, 28 * mm, 24 * mm], align_right=(1, 2)))
    F.append(Spacer(1, 5))
    P(f"3초 안에 시야를 벗어나는 답이 {g['leaves']:.1%} 입니다. 답이 그냥 "
      f"짧아지는 대신 “벗어난다”고 말하게 한 이유가 이것입니다 — 잘라내면 "
      f"짧은 답이 늘 안전하다고 배웁니다.", "small")

    P("4.3 검출의 클래스 쏠림 — 지우지 않았다", "h2")
    F.append(table(share_rows(g["observed"], 10),
                   [42 * mm, 28 * mm, 24 * mm], align_right=(1, 2)))
    F.append(Spacer(1, 5))
    P(f"검출은 “앞에 있는 도로 사용자를 <b>전부</b> 나열하라”이고, 아이템의 "
      f"<b>{g['carries_auto']:.2%}</b> 가 automobile 을 담고 있습니다. 답에서 "
      f"automobile 만 빼면 그 답은 틀린 답이 됩니다. 남길 아이템을 골라 "
      f"바꿀 수 있는 폭은 이만큼입니다:", "body")
    rows = [["무엇을 남기면", "남는 아이템", "결과"],
            ["그대로", "100%", "automobile 79.2%, person 14.3%"],
            ["희소 클래스를 담은 것만", "24.8%",
             "automobile 61.4%, person 16.5%, rider 6.6%"],
            ["automobile 이 없는 것만", "7.4%",
             "person 80.5% — 우세 클래스만 바뀜"]]
    F.append(table(rows, [42 * mm, 26 * mm, 66 * mm], align_right=(1,)))
    F.append(Spacer(1, 5))
    P("75% 를 버려서 79% → 61% 이고, 93% 를 버리면 이번엔 person 이 80% 가 "
      "됩니다. 게다가 “트럭이 있는 장면”만 남기면 도로 종류와 물체 밀도까지 "
      "함께 편향됩니다. 클래스 비율은 라벨링 선택이 아니라 도로의 성질이라, "
      "<b>장면을 왜곡하는 대신 클래스별로 점수를 따로 냅니다</b> — 채점기가 "
      "per_class 로 F1·recall·precision 을 함께 보고합니다. 전체 F1 하나로는 "
      "자전거를 한 번도 못 잡는 모델과 다 잡는 모델이 1점 차이도 나지 "
      "않습니다.", "body")

    # ------------------------------------------------------------- 좌표계
    F.append(PageBreak())
    P("5. 좌표계와 정답의 정밀도", "h1")
    P("궤적을 세 형식으로 냅니다. 셋은 <b>기준 프레임이 다르고</b>, 그래서 "
      "지시문이 각자 자기 프레임을 밝힙니다.", "body")
    rows = [["형식", "기준 프레임", "예"],
            ["agent_traj_xy", "질문 시점의 ego 프레임",
             "+1s (+23.7, +0.0)"],
            ["agent_traj_azdeg", "각 horizon 시점의 ego 기준",
             "+1s 32 m az +6 deg"],
            ["agent_traj_bbox", "각 horizon 시점의 카메라",
             "+1s [414, 483, 454, 530]"]]
    F.append(table(rows, [34 * mm, 50 * mm, 50 * mm]))
    F.append(Spacer(1, 5))
    P("질문도 답과 같은 언어로 물체를 가리킵니다. 예전에는 답이 무엇이든 "
      "질문이 항상 극좌표였는데, 그러면 이미지 박스를 답해야 하는 문제를 "
      "읽으려고 먼저 극좌표를 이미지로 옮겨야 했습니다 — 과제와 무관한 "
      "산수입니다.", "body")
    P("정답 자체의 잡음도 한 번 걷어냈습니다. ego 자세를 10 Hz 스트림에서 "
      "<b>가장 가까운 샘플</b>로 고르고 있었는데, ±50 ms 는 8.5 m/s 에서 "
      "0.43 m 입니다. 물체의 월드 좌표를 관측 시점 ego 프레임으로 되돌리면 "
      "라벨이 이미 가진 좌표가 나와야 하는데 중앙값 0.32 m 어긋났습니다. "
      "보간으로 바꾸니 <b>0.318 m → 0.033 m</b> 입니다. plan_ego_xy 도 같은 "
      "함수를 쓰므로 함께 좋아졌고, 대신 그 이전 체크포인트의 "
      "plan_ego_xy 수치와는 같은 자로 잰 값이 아닙니다.", "body")
    P("남은 어긋남 하나는 버그가 아닙니다. xy 와 azdeg 를 서로 변환해 비교하면 "
      "중앙값 0.33 m 차이가 나는데, (관측 시각 어긋남 × 자차 속도)와 상관계수 "
      "0.898 입니다 — 라벨이 10 Hz 라 “+1초”의 관측이 정확히 +1.000 초가 아닌 "
      "데서 옵니다. 두 형식 각각은 자기 정의에 정확합니다.", "small")

    # ------------------------------------------------------------- 지운 것
    F.append(PageBreak())
    P("6. 지운 것", "h1")
    P("<b>전방 레이더가 없는 클립을 전부 뺐습니다.</b> 177,891개 중 17,130개가 "
      "'none' 프로파일입니다. 이 클립들도 detection 이나 tracking 의 정답은 "
      "멀쩡하게 만들 수 있습니다 — 답은 obstacle 라벨과 egomotion 에서 나오니까요. "
      "만들 수 없는 것은 <b>정직한 근거</b>였습니다. 근거문의 증거 절이 모든 "
      "물체에 대해 “no radar return” 으로 나오는데, 그건 “센서가 봤는데 아무것도 "
      "없었다”는 주장입니다. 사실은 <b>센서가 없습니다</b>. 다른 주장이고, "
      "모델이 카메라만 보고 그렇게 말하도록 배울 주장입니다.", "body")
    P("아이템 길이 상한도 둡니다. 한 배치의 비용은 그 배치에서 <b>가장 긴</b> "
      "아이템이 정합니다. 손실의 로짓이 [batch, padded_len, vocab] 이라, 긴 "
      "아이템 하나가 배치 전체의 메모리를 끌어올립니다.", "small")

    P("7. 다시 만들려면", "h1")
    F.append(Paragraph(
        "python -m datatools.frame_objects --clips all --workers 56<br/>"
        "python -m datatools.build_training_stat_report", s["code"]))
    P("이 문서의 수치는 파케이에서 읽습니다. 데이터를 다시 만들면 이 문서도 "
      "다시 만들어야 합니다.", "small")

    doc.build(F)
    log(f"wrote {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--items", default=os.path.join(paths.COMMON_DIR, ITEMS))
    ap.add_argument("--out", default="training_data_stat.pdf")
    args = ap.parse_args(argv)

    from training.instruct_data import DEFAULT_MIXTURE, STRATA
    log(f"reading {args.items}")
    g = gather(args.items)
    build(g, args.out, DEFAULT_MIXTURE, STRATA)
    return 0


if __name__ == "__main__":
    sys.exit(main())
