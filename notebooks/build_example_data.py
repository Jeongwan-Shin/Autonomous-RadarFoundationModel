#!/usr/bin/env python3
"""테스크마다 예제 10건을 원시 입력째로 떼어내 노트북이 바로 열 수 있게 저장한다.

노트북이 지금까지 확인할 수 있던 것은 3.7 GB parquet 이 있는 기계 위에서뿐이었다.
여기서 떨어지는 `example_data/` 는 그 의존을 없앤다 — 모델이 실제로 받는 프레임과
레이더 반사점, 그리고 그 입력에서 나와야 하는 정답과 근거가 한 폴더에 함께 있다.

    notebooks/example_data/
      raw/<task>/00..09/frames/f00.jpg ...   실제로 입력되는 프레임 (384x216)
      raw/<task>/00..09/radar.npz            마스크된 반사점만, 스캔 인덱스와 함께
      raw/<task>/00..09/meta.json            클립·앵커·센서·창 사양
      gen_data/<task>.jsonl                  LLM 학습에 쓰이는 아이템 그대로

`_cot` 변형에서 뽑는다. CoT 정답의 `answer` 필드가 평문 변형의 정답과 글자 그대로
같으므로 한 아이템이 두 변형을 모두 보여준다. 평문과 CoT 를 따로 뽑으면 서로 다른
클립이 잡혀 근거의 수치와 정답의 수치가 어긋나 보인다.

레이더는 [20, 1024, 8] 텐서 그대로가 아니라 마스크가 참인 점만 저장한다. 패딩이
대부분이라 그대로 두면 예제 하나가 327 kB 이고, 걸러내면 그 몇 분의 일이다.

전방 레이더가 없는 클립은 17,130개, 전체의 9.6% 다. 그 비율은 학습에서 만나는
사실이지만 예제로는 보여줄 것이 없으므로 기본적으로 건너뛴다. 비율 자체를 보고
싶으면 `--any-radar` 로 그대로 뽑는다.

    python -m notebooks.build_example_data --per-task 10
"""

import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from training.instruct_data import InstructDataset, WINDOWS, INSTANT_TASKS
from training.radar_data import CHANNELS
from training.train_vlm import MODEL_DIR

TASKS = ["det_objects_azdeg", "det_objects_3dbbox", "track_step_azdeg",
         "track_step_bbox", "plan_ego_xy", "plan_ego_control",
         "agent_traj_azdeg", "agent_traj_bbox", "motion_seg_azdeg",
         "motion_seg_bbox", "qa"]
PAD = re.compile(r"<\|(radar|vision|video|image)_(start|end|pad)\|>")
OUT = os.path.join(HERE, "example_data")


def window_spec(task):
    if task in WINDOWS:
        seconds, hz, frames = WINDOWS[task]
        return {"kind": "window", "seconds": seconds, "radar_hz": hz,
                "video_frames": frames}
    if task in INSTANT_TASKS:
        return {"kind": "instant", "seconds": 1, "radar_hz": 20,
                "video_frames": 1}
    return {"kind": "clip", "seconds": 20, "radar_hz": 1, "video_frames": 20}


def save(sample, payload, task, slot):
    """One example: the frames and returns that go in, the answer that comes out."""
    d = os.path.join(OUT, "raw", task, f"{slot:02d}")
    os.makedirs(os.path.join(d, "frames"), exist_ok=True)
    names = []
    for k, image in enumerate(sample["frames"]):
        name = f"f{k:02d}.jpg"
        image.save(os.path.join(d, "frames", name), quality=88)
        names.append(name)

    points, mask = sample["points"].numpy(), sample["mask"].numpy().astype(bool)
    scan = np.repeat(np.arange(mask.shape[0], dtype=np.uint8),
                     mask.sum(axis=1))
    np.savez_compressed(os.path.join(d, "radar.npz"),
                        points=points[mask].astype(np.float16),
                        scan=scan, channels=np.array(CHANNELS))

    lines = sample["user"].split("\n")
    grab = lambda p: next((l for l in lines if l.startswith(p)), "")
    instruction = "\n".join(l for l in lines
                            if not l.startswith(("Sensors", "Ego motion")))
    record = {
        "id": f"{task}/{slot:02d}",
        "task": task,
        "clip_id": sample["clip_id"],
        "sensors": grab("Sensors"),
        "ego": grab("Ego motion"),
        "instruction": re.sub(r"\n{3,}", "\n\n", PAD.sub("", instruction)).strip(),
        "target": payload.get("answer", ""),
        "rationale": payload.get("rationale", ""),
        "cot_target": sample["target"],
        "window": window_spec(task),
        "n_frames": len(names),
        "radar_points": int(mask.sum()),
        "radar_scans": int(mask.shape[0]),
        "raw": os.path.relpath(d, OUT),
    }
    json.dump(record, open(os.path.join(d, "meta.json"), "w"),
              indent=1, ensure_ascii=False)
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-task", type=int, default=10)
    ap.add_argument("--split", default="train")
    ap.add_argument("--any-radar", action="store_true",
                    help="전방 레이더가 없는 클립(전체의 9.6%)도 예제로 받는다. "
                         "기본은 반사점이 있는 건만 골라, 예제를 여는 사람이 "
                         "빈 산점도를 먼저 만나지 않게 한다")
    ap.add_argument("--samples", type=int, default=8000,
                    help="이 중에서 태스크마다 앞에서부터 --per-task 개를 고른다")
    args = ap.parse_args(argv)

    from transformers import AutoProcessor, AutoTokenizer
    names = tuple(t + "_cot" for t in TASKS)
    md = MODEL_DIR["8B"]
    tok = AutoTokenizer.from_pretrained(md)
    proc = AutoProcessor.from_pretrained(md)
    proc.tokenizer = tok
    ds = InstructDataset(tasks=names, split=args.split, processor=proc,
                         tokenizer=tok, samples=args.samples, all_profiles=True)
    print(f"dataset {len(ds):,}", flush=True)

    filled = {t: 0 for t in TASKS}
    records = {t: [] for t in TASKS}
    for i in range(len(ds)):
        task = ds.items[i]["task"].removesuffix("_cot")
        if task not in filled or filled[task] >= args.per_task:
            continue
        sample = ds[i]
        if not args.any_radar and not sample["mask"].any():
            continue          # 리그에 전방 레이더가 없는 클립 — 반사점이 0개다
        try:
            payload = json.loads(sample["target"])
        except Exception:
            continue          # 잘린 JSON 은 예제로 쓸 수 없다
        records[task].append(save(sample, payload, task, filled[task]))
        filled[task] += 1
        if filled[task] == args.per_task:
            print(f"  {task} {args.per_task}건", flush=True)
        if all(v >= args.per_task for v in filled.values()):
            break

    gen = os.path.join(OUT, "gen_data")
    os.makedirs(gen, exist_ok=True)
    for task, rows in records.items():
        with open(os.path.join(gen, f"{task}.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    short = {t: filled[t] for t in TASKS if filled[t] < args.per_task}
    if short:
        print(f"!! 모자란 태스크: {short}", flush=True)
    total = sum(os.path.getsize(os.path.join(p, f))
                for p, _, fs in os.walk(OUT) for f in fs)
    print(f"wrote {OUT}  ({sum(filled.values())}건, {total/1e6:.1f} MB)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
