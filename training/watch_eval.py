#!/usr/bin/env python3
"""학습이 저장할 때마다 그 체크포인트를 재고, 추세를 한 줄씩 쌓는다.

한 번의 점수는 좋아지고 있는지 말해 주지 않는다. 200 스텝마다 같은 항목을 같은
방식으로 재서 나란히 놓아야, 방위각이 실제로 벌어지고 있는지 아니면 손실만
내려가고 있는지가 보인다.

학습 프로세스 안이 아니라 옆에서 돈다. 학습 루프에 생성을 끼워 넣으면 거기서
난 버그가 학습 전체를 죽이는데, 이쪽은 죽어도 학습이 계속된다. 대신 GPU 를
나눠 써야 하므로 학습의 micro-batch 를 낮춰 자리를 비워 두어야 한다.

    python -m training.watch_eval --checkpoint <학습 --out 경로> --device 4

무엇을 보는가:

  det_objects_azdeg   F1, 그리고 생성된 |방위각| 의 평균. 이 모델의 알려진
                      실패가 방위각을 정면으로 몰아넣는 것이라, 정답의 평균과
                      나란히 찍는다. 둘이 가까워지는 것이 이 학습이 답해야 할
                      질문이다.
  det_objects_3dbbox  F1 과 크기·요각 오차
  plan_ego_xy         변위 오차와, 예측 경로가 라벨된 물체와 겹치는 비율
"""

import argparse
import json
import os
import re
import shutil
import time

import numpy as np

WATCHED = ("det_objects_azdeg", "det_objects_3dbbox", "plan_ego_xy")
AZ = re.compile(r"az\s*([+-]?\d+)\s*deg")
XY = re.compile(r"\+(\d)s\s*\(\s*([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*\)")

# The ego footprint the collision test sweeps along the predicted path. The
# release does not ship vehicle dimensions per clip, so this is the usual
# passenger-car box; the number is a rate under a fixed assumption, not a
# certified safety figure, and it is comparable across checkpoints because the
# assumption never changes.
EGO_LENGTH, EGO_WIDTH = 4.6, 1.9


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def saved_step(path):
    try:
        h = json.load(open(os.path.join(path, "latest", "history.json")))
        return int(h[-1]["step"]) if h else None
    except Exception:
        return None


def mean_abs_azimuth(text):
    v = [abs(float(x)) for x in AZ.findall(text or "")]
    return float(np.mean(v)) if v else None


def waypoints(text):
    return {int(h): (float(x), float(y)) for h, x, y in XY.findall(text or "")}


def collides(path, obstacles):
    """예측 경로 위 자차 사각형이 라벨된 물체 상자와 겹치는가.

    축 정렬 근사다. 자차의 진행 방향으로 회전시키지 않으므로 회전 구간에서는
    조금 넉넉하게 잡힌다 -- 체크포인트끼리 비교하는 데는 같은 근사가 양쪽에
    걸리므로 문제가 되지 않지만, 절대값으로 인용하면 안 된다.
    """
    hit = 0
    for h, (x, y) in path.items():
        for ox, oy, ol, ow in obstacles.get(h, ()):
            if (abs(x - ox) < (EGO_LENGTH + ol) / 2
                    and abs(y - oy) < (EGO_WIDTH + ow) / 2):
                hit += 1
                break
    return hit


def evaluate(snapshot, args):
    from training.eval_all_tasks import load_model, run_task

    class A:
        pass
    a = A()
    a.checkpoint, a.model, a.split = snapshot, args.model, args.split
    a.items, a.workers, a.show = args.items, 2, 0
    a.max_new_floor, a.all_profiles, a.seed = 0, True, 0
    a.radar_dropout, a.out = 0.0, None
    loaded = load_model(a)

    row = {}
    for task in WATCHED:
        r = run_task(task, a, loaded)
        if not r:
            continue
        # `run_task` scores twice, once on the real radar and once on a
        # shuffled one. `full` is the model; `shuffled` is the control.
        s = r.get("full") or {}
        gens = r.get("generations") or []
        if task == "det_objects_azdeg":
            g = [mean_abs_azimuth(x.get("generated")) for x in gens]
            t = [mean_abs_azimuth(x.get("reference") or x.get("target"))
                 for x in gens]
            row["det_f1_shuffled"] = (r.get("shuffled") or {}).get("f1")
            row["az_gen"] = float(np.mean([v for v in g if v is not None] or [0]))
            row["az_truth"] = float(np.mean([v for v in t if v is not None] or [0]))
            row["det_f1"] = s.get("f1")
            row["det_range_mae"] = s.get("range_mae")
            row["det_az_mae"] = s.get("az_mae")
        elif task == "det_objects_3dbbox":
            row["box_f1"] = s.get("f1")
            row["box_size_mae"] = s.get("size_mae")
            row["box_yaw_mae"] = s.get("yaw_mae")
        elif task == "plan_ego_xy":
            row["l2"] = s.get("displacement_mae_m")
            per_h = {1: [], 2: [], 3: []}
            for x in gens:
                p, t = waypoints(x["generated"]), waypoints(x["reference"])
                for h in per_h:
                    if h in p and h in t:
                        per_h[h].append(float(np.hypot(p[h][0] - t[h][0],
                                                       p[h][1] - t[h][1])))
            for h, v in per_h.items():
                row[f"l2_{h}s"] = float(np.mean(v)) if v else None
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True, help="학습의 --out 경로")
    ap.add_argument("--device", default="4", help="이 감시기가 쓸 GPU")
    ap.add_argument("--items", type=int, default=60,
                    help="태스크당 항목 수. 추세를 보는 것이므로 작아도 되지만 "
                         "체크포인트마다 같아야 한다")
    ap.add_argument("--model", default="8B")
    ap.add_argument("--split", default="test")
    ap.add_argument("--every", type=int, default=1,
                    help="저장 몇 번마다 잴 것인가")
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    trend = os.path.join(args.checkpoint, "trend.jsonl")
    done = set()
    if os.path.exists(trend):
        done = {json.loads(l)["step"] for l in open(trend)}
    log(f"감시 시작 {args.checkpoint} · GPU {args.device} · "
        f"태스크당 {args.items}건 · 이미 잰 것 {len(done)}개")

    seen = 0
    while True:
        step = saved_step(args.checkpoint)
        if step is None or step in done:
            time.sleep(args.poll)
            continue
        seen += 1
        if seen % args.every:
            done.add(step)
            continue
        snap = os.path.join(args.checkpoint, f"_eval_{step}")
        try:
            shutil.rmtree(snap, ignore_errors=True)
            shutil.copytree(os.path.join(args.checkpoint, "latest"), snap)
            if saved_step(args.checkpoint) != step:
                log(f"step {step} 복사 중 저장이 끼어들었습니다 -- 건너뜁니다")
                shutil.rmtree(snap, ignore_errors=True)
                done.add(step)
                continue
            started = time.monotonic()
            row = evaluate(snap, args)
            row["step"] = step
            row["seconds"] = round(time.monotonic() - started, 1)
            with open(trend, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            log(f"step {step:>6}  det F1 {row.get('det_f1'):.3f}  "
                f"|az| 생성 {row.get('az_gen'):.1f} / 정답 "
                f"{row.get('az_truth'):.1f}  L2 {row.get('l2'):.3f} m  "
                f"({row['seconds']:.0f}s)")
        except Exception as exc:
            log(f"step {step} 평가 실패: {type(exc).__name__} {exc}")
        finally:
            shutil.rmtree(snap, ignore_errors=True)
            done.add(step)


if __name__ == "__main__":
    raise SystemExit(main())
