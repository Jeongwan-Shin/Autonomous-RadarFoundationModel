#!/usr/bin/env python3
"""테스트 번들을 클립 중심으로 떼어낸다.

전에는 태스크마다 200건씩 뽑았고, 그래서 태스크끼리 클립이 거의 겹치지 않았다
-- `det_objects` 의 198개 클립과 `plan_ego` 의 199개 클립 중 공통은 1개였다.
같은 장면을 놓고 "이 태스크는 무엇을 묻고 저 태스크는 무엇을 묻는가" 를 볼
수가 없고, 아이템마다 자기가 쓰는 프레임만 갖고 있어서 (`det_objects` 는 1장,
`track_step` 은 5장) 클립을 통째로 볼 수도 없다.

여기서는 클립을 먼저 고른다. 클립 하나에 20 프레임 전부와 레이더와 ego 를 한
번씩 두고, 그 클립이 만들어 내는 <b>모든</b> 태스크의 프롬프트와 정답을 그 옆에
모은다. 아이템은 프레임을 번호로 가리키므로 같은 프레임이 스무 번 복사되지
않고, 레이더 창은 (프레임, 표본 속도) 가 같으면 한 파일을 공유한다.

    python 20260808_test/export_items.py --clips 10

`--clips` 를 올리면 채점용 표본도 늘어난다. 열 개는 들여다보기에는 충분하지만
점수를 말하기에는 작은 수라, 이 번들로 낸 값은 지표가 아니라 참고다.
"""

import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)


def log(m):
    print(m, flush=True)


# 번들이 저장소를 참조하지 않으려면 이 모듈들의 사본이 필요하다. 손으로
# 복사해 두었더니 낡았다 -- `agent_traj_xy` 가 생긴 뒤 번들의 채점기는 그
# 태스크를 몰라 일반 텍스트로 떨어뜨렸고, 그건 점수 0.00 과 구별되지 않는다.
# 그래서 번들을 만들 때마다 다시 복사한다.
COPIES = {"connector.py": "training/connector.py",
          "radar_encoder.py": "training/radar_encoder.py",
          "task_scorers.py": "training/task_scorers.py",
          "number_tokens.py": "training/number_tokens.py",
          "number_vocab.txt": "training/number_vocab.txt"}

# 사본이 원본과 달라야 하는 유일한 곳. 저장소에서는 vocab 이 training/ 아래에
# 있고 번들에서는 모듈 옆에 있다.
VOCAB_FIX = (
    'os.path.dirname(os.path.dirname(os.path.abspath(__file__))),\n'
    '                          "training", "number_vocab.txt")',
    'os.path.dirname(os.path.abspath(__file__)),\n'
    '                          "number_vocab.txt")')


def sync_modules():
    out = os.path.join(HERE, "ravl")
    for name, rel in COPIES.items():
        text = open(os.path.join(REPO, rel)).read()
        if name == "number_tokens.py":
            if VOCAB_FIX[0] not in text:
                raise SystemExit("number_tokens.py 의 VOCAB_PATH 가 바뀌었습니다 "
                                 "-- export_items.py 의 VOCAB_FIX 를 고치세요.")
            text = text.replace(*VOCAB_FIX)
        path = os.path.join(out, name)
        if not os.path.exists(path) or open(path).read() != text:
            open(path, "w").write(text)
            log(f"  사본 갱신 ravl/{name}")


def radar_key(task, frame, windows, instant_tasks, window_tasks):
    """(프레임, 표본 속도) 가 같은 아이템은 같은 레이더 창을 쓴다."""
    if task in instant_tasks:
        return f"f{int(frame):02d}_instant"
    if task in window_tasks:
        return f"f{int(frame):02d}_hz{windows[task][1]}"
    return "clip"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clips", type=int, default=10)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-per-task", type=int, default=0,
                    help="한 클립에서 한 태스크가 내놓는 아이템 상한 (0 = 전부)")
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    args = ap.parse_args(argv)

    from transformers import AutoProcessor, AutoTokenizer
    from training.eval_all_tasks import ALL
    from training.instruct_data import (INSTANT_TASKS, WINDOWS, WINDOW_TASKS,
                                        InstructDataset, ego_text)
    from training.radar_data import CHANNELS
    from training.train_vlm import MODEL_DIR

    sync_modules()

    md = MODEL_DIR["8B"]
    tok = AutoTokenizer.from_pretrained(md)
    proc = AutoProcessor.from_pretrained(md)
    proc.tokenizer = tok

    # 한 번만 읽는다. 태스크마다 데이터셋을 새로 지으면 3.7 GB 파케이를
    # 스물아홉 번 읽게 된다.
    groups = sorted({"description" if t.startswith("desc_") else t for t in ALL})
    log(f"데이터셋 구성 -- {len(groups)}개 그룹, split={args.split}")
    ds = InstructDataset(tasks=tuple(groups), split=args.split, processor=proc,
                         tokenizer=tok, samples=0, all_profiles=True,
                         radar_dropout=0.0)
    log(f"  아이템 {len(ds.items):,}건")

    wanted_tasks = set(ALL)
    by_clip = collections.defaultdict(list)
    for k, item in enumerate(ds.items):
        if item["task"] in wanted_tasks:
            by_clip[item["clip_id"]].append(k)

    # 태스크를 가장 많이 덮는 클립부터. 스무 개 태스크만 나오는 클립을 고르면
    # 나머지 아홉은 그 장면에서 무엇을 묻는지 볼 수 없다.
    def coverage(idxs):
        return len({ds.items[i]["task"] for i in idxs})
    ranked = sorted(by_clip.items(), key=lambda kv: (-coverage(kv[1]), kv[0]))
    chosen = ranked[: args.clips]
    log(f"  클립 {len(by_clip):,}개 중 {len(chosen)}개 선택 "
        f"(태스크 커버리지 {coverage(chosen[0][1])}"
        f"~{coverage(chosen[-1][1])} / {len(wanted_tasks)})")

    os.makedirs(args.out, exist_ok=True)
    clips_dir = os.path.join(args.out, "clips")
    by_task = collections.defaultdict(list)
    clip_rows, n_items = [], 0

    for clip_id, idxs in chosen:
        d = os.path.join(clips_dir, clip_id)
        os.makedirs(os.path.join(d, "frames"), exist_ok=True)
        os.makedirs(os.path.join(d, "radar"), exist_ok=True)

        # 프레임은 클립당 한 벌. 아이템은 번호로 가리킨다.
        whole = next((i for i in idxs
                      if ds.items[i]["task"] not in INSTANT_TASKS
                      and ds.items[i]["task"] not in WINDOW_TASKS), idxs[0])
        full = ds[whole]
        frame_names = []
        for j, image in enumerate(full["frames"]):
            name = f"f{j:02d}.jpg"
            image.save(os.path.join(d, "frames", name), quality=88)
            frame_names.append(name)

        # ego 는 프롬프트 안에 이미 문장으로 들어가 있다. 그 문장이 어떤
        # 수치에서 나왔는지 함께 두어야 답을 검산할 수 있으므로, 클립 전체의
        # ego 상태를 원본 그대로 한 벌 저장한다.
        sensor_name, position = ds.radar_pos[clip_id]
        state = np.asarray(ds.radar_ds[sensor_name][position]["ego_state"])
        np.save(os.path.join(d, "ego.npy"), state)
        with open(os.path.join(d, "ego.txt"), "w") as fh:
            fh.write(f"# {sensor_name}\n")
            fh.write("# speed / accel / yaw-rate, binned exactly as the "
                     "prompts quote them\n")
            fh.write(ego_text(state, len(state) - 1) + "\n")

        seen_radar, rows = {}, []
        per_task = collections.Counter()
        for i in sorted(idxs, key=lambda i: (ds.items[i]["task"],
                                             ds.items[i]["frame"])):
            item = ds.items[i]
            if args.max_per_task and per_task[item["task"]] >= args.max_per_task:
                continue
            per_task[item["task"]] += 1
            s = ds[i]
            key = radar_key(item["task"], item["frame"], WINDOWS,
                            INSTANT_TASKS, WINDOW_TASKS)
            if key not in seen_radar:
                pts = s["points"].numpy()
                mask = s["mask"].numpy().astype(bool)
                # 실제 반사점만. 패딩된 텐서는 [n, 1024, 8] 이고 대부분 0 이라
                # 통째로 두면 번들이 열 배가 된다.
                np.savez_compressed(
                    os.path.join(d, "radar", f"{key}.npz"),
                    points=pts[mask].astype(np.float16),
                    scan=np.repeat(np.arange(mask.shape[0], dtype=np.uint8),
                                   mask.sum(axis=1)),
                    shape=np.array(pts.shape, dtype=np.int32))
                seen_radar[key] = int(mask.sum())

            # 이 아이템이 실제로 보는 프레임 번호. 같은 클립이라도 태스크마다
            # 다르다 -- det_objects 는 한 장, track_step 은 다섯 장.
            last = len(frame_names) - 1
            if item["task"] in INSTANT_TASKS:
                uses = [min(int(item["frame"]), last)]
            elif item["task"] in WINDOW_TASKS:
                n = WINDOWS[item["task"]][2]
                end = min(int(item["frame"]), last)
                uses = list(range(max(0, end - n + 1), end + 1))
            else:
                uses = list(range(len(frame_names)))

            row = {"id": f"{clip_id}/{item['task']}/{int(item['frame']):02d}",
                   "task": item["task"], "clip_id": clip_id,
                   "frame": int(item["frame"]),
                   "user": s["user"], "target": s["target"],
                   "sensor": int(s["sensor"]), "radar": key,
                   "frames": uses, "radar_points": seen_radar[key]}
            rows.append(row)
            by_task[item["task"]].append(row)
            n_items += 1

        with open(os.path.join(d, "tasks.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        clip_rows.append({"clip_id": clip_id, "n_frames": len(frame_names),
                          "n_items": len(rows),
                          "tasks": sorted({r["task"] for r in rows}),
                          "radar_windows": len(seen_radar)})
        log(f"  {clip_id}  아이템 {len(rows):>4}  "
            f"태스크 {len(clip_rows[-1]['tasks']):>2}  "
            f"레이더 창 {len(seen_radar):>3}")

    os.makedirs(os.path.join(args.out, "by_task"), exist_ok=True)
    for task, rows in sorted(by_task.items()):
        with open(os.path.join(args.out, "by_task", f"{task}.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(args.out, "clips.jsonl"), "w") as fh:
        for r in clip_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"split": args.split, "n_clips": len(clip_rows),
               "n_items": n_items, "channels": list(CHANNELS),
               "tasks": {t: len(r) for t, r in sorted(by_task.items())},
               "layout": "clip-centric"},
              open(os.path.join(args.out, "manifest.json"), "w"),
              indent=1, ensure_ascii=False)

    total = sum(os.path.getsize(os.path.join(p, f))
                for p, _, fs in os.walk(args.out) for f in fs)
    log(f"\n클립 {len(clip_rows)}개 · 아이템 {n_items:,}건 · "
        f"태스크 {len(by_task)}종 · {total/1e9:.2f} GB → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
