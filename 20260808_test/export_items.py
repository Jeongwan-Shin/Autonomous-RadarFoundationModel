#!/usr/bin/env python3
"""테스트 아이템을 입력째로 떼어내 다른 서버에서도 그대로 돌아가게 만든다.

여기서 나오는 폴더에는 원본 아카이브도 3.6 GB 파케이도 필요 없다. 아이템마다
모델이 실제로 받는 것 -- 프레임, 레이더 반사점, 프롬프트 -- 과 정답이 함께 들어
있어서, 옮긴 서버에는 체크포인트와 베이스 모델만 있으면 된다.

이 스크립트는 원본 서버에서 한 번만 돌린다. 저장소의 코드나 데이터는 읽기만 한다.

    python 20260808_test/export_items.py --items 200
"""

import argparse
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    args = ap.parse_args(argv)

    from transformers import AutoProcessor, AutoTokenizer
    from training.eval_all_tasks import ALL
    from training.instruct_data import InstructDataset, WINDOWS, INSTANT_TASKS
    from training.radar_data import CHANNELS
    from training.train_vlm import MODEL_DIR

    md = MODEL_DIR["8B"]
    tok = AutoTokenizer.from_pretrained(md)
    proc = AutoProcessor.from_pretrained(md)
    proc.tokenizer = tok

    os.makedirs(args.out, exist_ok=True)
    manifest = {"split": args.split, "items_per_task": args.items,
                "channels": list(CHANNELS), "tasks": {}}

    for task in ALL:
        # The description kinds are keyed off the group name, exactly as the
        # evaluator does it; asking for "desc_radar" directly matches no branch.
        group = "description" if task.startswith("desc_") else task
        ds = InstructDataset(tasks=(group,), split=args.split, processor=proc,
                             tokenizer=tok, samples=0 if group != task else args.items,
                             all_profiles=True, radar_dropout=0.0)
        if group != task:
            ds.items = [i for i in ds.items if i["task"] == task][: args.items]
        if not len(ds):
            print(f"  {task}: 아이템 없음", flush=True)
            continue

        rows = []
        for k in range(min(len(ds), args.items)):
            s = ds[k]
            d = os.path.join(args.out, task, f"{k:03d}")
            os.makedirs(os.path.join(d, "frames"), exist_ok=True)
            names = []
            for j, image in enumerate(s["frames"]):
                name = f"f{j:02d}.jpg"
                image.save(os.path.join(d, "frames", name), quality=88)
                names.append(name)
            pts = s["points"].numpy()
            mask = s["mask"].numpy().astype(bool)
            # Only the real returns. The padded tensor is [20, 1024, 8] and
            # mostly zeros; keeping it whole would multiply the bundle by ten.
            np.savez_compressed(
                os.path.join(d, "radar.npz"),
                points=pts[mask].astype(np.float16),
                scan=np.repeat(np.arange(mask.shape[0], dtype=np.uint8),
                               mask.sum(axis=1)),
                shape=np.array(pts.shape, dtype=np.int32))
            rows.append({
                "id": f"{task}/{k:03d}", "task": task,
                "clip_id": s["clip_id"], "dir": os.path.join(task, f"{k:03d}"),
                "user": s["user"],          # 프롬프트 전체, 자리표시자 포함
                "target": s["target"],
                "sensor": int(s["sensor"]),
                "frames": names,
                "radar_points": int(mask.sum()),
            })
        with open(os.path.join(args.out, f"{task}.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["tasks"][task] = len(rows)
        print(f"  {task:26s} {len(rows):>4}건", flush=True)

    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"),
              indent=1)
    total = sum(os.path.getsize(os.path.join(p, f))
                for p, _, fs in os.walk(args.out) for f in fs)
    print(f"\n{sum(manifest['tasks'].values()):,}건 · {total/1e9:.2f} GB "
          f"→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
