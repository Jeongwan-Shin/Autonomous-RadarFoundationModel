#!/usr/bin/env python3
"""`watch_eval` 이 쌓은 추세를 표로 낸다.

    python -m training.show_trend --checkpoint <학습 --out 경로>

각 열에 방향을 붙인다. 지표마다 좋은 쪽이 다른데 표에는 숫자만 남으므로,
나중에 읽는 사람이 F1 이 오른 것과 MAE 가 오른 것을 같은 눈으로 보게 된다.

방위각만 화살표가 다르다. 생성된 |방위각| 은 크다고 좋은 것도 작다고 좋은
것도 아니고 정답에 가까워야 한다 -- 이 모델의 알려진 실패가 정면으로
몰아넣는 것이라 0 에 가까울수록 나쁘고, 지나쳐도 나쁘다. 그래서 정답과의
차이를 따로 낸다.
"""

import argparse
import json
import os

# (키, 표시 이름, 방향, 서식). 방향: +1 클수록 좋음, -1 작을수록 좋음, 0 중립
COLUMNS = [
    ("step",            "step",      0, "{:.0f}"),
    ("det_f1",          "det F1",   +1, "{:.3f}"),
    ("det_map",         "det mAP",  +1, "{:.3f}"),
    ("det_f1_shuffled", "섞은레이더", -1, "{:.3f}"),
    ("_radar_gain",     "레이더배수", +1, "{:.2f}"),
    ("az_gen",          "|az|생성",   0, "{:.1f}"),
    ("_az_gap",         "|az|차이",  -1, "{:.1f}"),
    ("det_range_mae",   "거리MAE",   -1, "{:.2f}"),
    ("det_az_mae",      "방위MAE",   -1, "{:.1f}"),
    ("box_f1",          "3D F1",    +1, "{:.3f}"),
    ("box_map",         "3D mAP",   +1, "{:.3f}"),
    ("box_chamfer",     "3D 어긋남",  -1, "{:.1f}"),
    ("box_within2m",    "2m이내",    +1, "{:.2f}"),
    ("box_size_mae",    "크기MAE",   -1, "{:.2f}"),
    ("box_yaw_mae",     "yawMAE",   -1, "{:.1f}"),
    ("l2",              "L2평균",    -1, "{:.3f}"),
    ("l2_1s",           "L2@1s",    -1, "{:.2f}"),
    ("l2_2s",           "L2@2s",    -1, "{:.2f}"),
    ("l2_3s",           "L2@3s",    -1, "{:.2f}"),
    # 같은 체크포인트를 nuScenes 에서 잰 것. 같은 열 이름이지만 다른 rig, 다른
    # 섹터(±30 도), 다른 클래스 분포이므로 위 값과 나란히 빼면 안 된다 --
    # 각각이 자기 안에서 어떻게 움직이는지를 본다.
    ("nus_det_f1",      "nuS detF1", +1, "{:.3f}"),
    ("nus_range_mae",   "nuS거리MAE", -1, "{:.2f}"),
    ("nus_az_gen",      "nuS|az|",    0, "{:.1f}"),
    ("nus_box_f1",      "nuS 3D F1", +1, "{:.3f}"),
    ("nus_l2",          "nuS L2",    -1, "{:.3f}"),
]
ARROW = {+1: "↑", -1: "↓", 0: ""}


def derived(row):
    """표에만 있는 열 -- 저장된 값에서 바로 나온다."""
    f, s = row.get("det_f1"), row.get("det_f1_shuffled")
    row["_radar_gain"] = (f / s) if f and s else None
    g, t = row.get("az_gen"), row.get("az_truth")
    row["_az_gap"] = abs(g - t) if g is not None and t is not None else None
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args(argv)
    path = os.path.join(args.checkpoint, "trend.jsonl")
    rows = [derived(json.loads(l)) for l in open(path)]
    if not rows:
        print("아직 없음")
        return 0

    head = "  " + " ".join(f"{n + ARROW[d]:>10}" for _, n, d, _ in COLUMNS)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in rows:
        cells = []
        for k, _, _, fmt in COLUMNS:
            v = r.get(k)
            cells.append(f"{'—':>10}" if v is None else f"{fmt.format(v):>10}")
        print("  " + " ".join(cells))

    first, last = rows[0], rows[-1]
    print()
    print(f"  |az| 정답 {last.get('az_truth', float('nan')):.1f} "
          f"· 생성이 여기에 가까워지는 것이 목표")
    moved = []
    for k, name, d, fmt in COLUMNS:
        if d == 0 or first.get(k) is None or last.get(k) is None:
            continue
        change = (last[k] - first[k]) * d
        moved.append((change / (abs(first[k]) or 1), name, first[k], last[k]))
    moved.sort(reverse=True)
    print(f"  step {first['step']:.0f} → {last['step']:.0f} 사이 "
          f"가장 좋아진 것과 나빠진 것")
    for frac, name, a, b in moved[:2] + moved[-2:]:
        print(f"    {name:10s} {a:8.3f} → {b:8.3f}  ({frac:+.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
