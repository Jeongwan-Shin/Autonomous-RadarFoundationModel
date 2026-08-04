#!/usr/bin/env python3
"""빌드된 데이터의 규모를 세어 보고서가 읽을 파일 하나로 남긴다.

수를 손으로 옮겨 적으면 데이터가 바뀐 다음에도 문서는 옛 수를 말한다. 세는 곳을
여기 하나로 두고 `build_dataset_report` 가 그 결과만 읽는다.

어느 클립을 쓸 수 있는지는 `training.instruct_data.usable_clips` 에 물어본다.
전방 레이더가 있어야 한다는 조건, QA 홀드아웃 제외, `val` 을 `train` 에 병합 —
전부 거기 한 곳에 있고, 여기서 다시 구현하면 언젠가 두 벌이 어긋나 보고서가
학습에 쓰이지 않은 수를 말하게 된다.

아이템 자체는 로더를 통째로 세우지 않고 원본에서 센다. `samples=0` 으로 세우면
2,700만 건이 파이썬 딕셔너리로 메모리에 올라간다 — 그저 세기 위해서.

    python -m datatools.dataset_summary
"""

import glob
import json
import os
import sys

import pandas as pd

from . import paths

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "runs", "01_data_prep", "dataset_summary.json")
DESC_DIR = "11_radar_vision_scene_description"


def frame_counts(keep, out, anchors):
    """01~06 — 하나의 파케이. 열 세 개만 읽어 클립과 분할로 거른다."""
    path = os.path.join(paths.COMMON_DIR, "instruct_items_tasks01_06.parquet")
    d = pd.read_parquet(path, columns=["task", "clip_id", "split", "frame"])
    for split, allowed in (("train", ("train", "val")), ("test", ("test",))):
        sub = d[d["split"].isin(allowed) & d["clip_id"].isin(keep[split])]
        for task, n in sub.groupby("task").size().items():
            out.setdefault(task, {})[split] = int(n)
    for task, g in d.groupby("task"):
        anchors[task] = sorted(int(x) - 1 for x in g.frame.unique())
    return {"rows": len(d), "bytes": os.path.getsize(path)}


def description_counts(keep, out):
    """11 — 프레임 단위 넷과 클립 요약 하나. context 와 qa_summary 는 제외."""
    root = os.path.join(paths.SPLIT_ROOT, DESC_DIR)
    frame_path = os.path.join(root, "descriptions_frame.parquet")
    if os.path.exists(frame_path):
        d = pd.read_parquet(frame_path, columns=["clip_id", "split", "kind"])
        for split, allowed in (("train", ("train", "val")), ("test", ("test",))):
            sub = d[d["split"].isin(allowed) & d["clip_id"].isin(keep[split])]
            for kind, n in sub.groupby("kind").size().items():
                out.setdefault(f"desc_{kind}", {})[split] = int(n)
    clip_path = os.path.join(root, "descriptions_clip.parquet")
    if os.path.exists(clip_path):
        d = pd.read_parquet(clip_path, columns=["clip_id", "split", "kind"])
        d = d[d["kind"] == "clip_summary"]
        for split, allowed in (("train", ("train", "val")), ("test", ("test",))):
            sub = d[d["split"].isin(allowed) & d["clip_id"].isin(keep[split])]
            out.setdefault("desc_clip_summary", {})[split] = int(len(sub))


def probe_counts(keep, out):
    """07 radar_probe — 장면 특징 파케이의 (클립, 프레임) 행마다 한 문항.

    세 가지 질문 형태가 있지만 행마다 `frame % 3` 으로 하나만 고르므로 행 수가
    곧 아이템 수다. 이 파케이에는 분할 열이 없어 클립 목록이 유일한 경계다.
    """
    path = os.path.join(paths.COMMON_DIR, "scene_features_all_clips.parquet")
    if not os.path.exists(path):
        return
    d = pd.read_parquet(path, columns=["clip_id", "lrr1_n_points"])
    d = d[d["lrr1_n_points"].notna()]
    for split in ("train", "test"):
        out.setdefault("radar_probe", {})[split] = int(
            d["clip_id"].isin(keep[split]).sum())


def qa_counts(keep, out):
    """10 — 방출 규칙이 여기서만 정해지므로 로더를 그대로 부른다. 문항 20만이라
    메모리에 다 올려도 된다."""
    from training.instruct_data import load_items
    for split in ("train", "test"):
        for task in ("qa", "qa_cot"):
            items = load_items((task,), split, usable_clips=keep[split])
            out.setdefault(task, {})[split] = len(items)

    root = os.path.join(paths.SPLIT_ROOT, "10_radar_vision_qa")
    sets = {}
    for name in ("qa_train", "qa_gt"):
        n_q, ids = 0, set()
        for f in glob.glob(os.path.join(root, name, "*.json")):
            doc = json.load(open(f))
            ids.add(doc["clip_id"])
            n_q += len(doc.get("qa", []))
        sets[name] = {"clips": len(ids), "questions": n_q}
    return sets


def main(argv=None):
    from training.instruct_data import usable_clips
    clips = pd.read_parquet(os.path.join(paths.COMMON_DIR,
                                         "nvidia_clips.parquet"))
    keep = {}
    for split in ("train", "test"):
        usable, _ = usable_clips(split, "lrr1", True, clips=clips)
        keep[split] = set(usable)
        print(f"{split:6s} 클립 {len(usable):>8,}", flush=True)

    by_task, anchors = {}, {}
    frame_items = frame_counts(keep, by_task, anchors)
    print("  01~06 완료", flush=True)
    description_counts(keep, by_task)
    probe_counts(keep, by_task)
    print("  07/11 완료", flush=True)
    qa_sets = qa_counts(keep, by_task)
    print("  10 완료", flush=True)

    out = {"frame_items": frame_items, "anchors": anchors, "by_task": by_task,
           "totals": {s: sum(v.get(s, 0) for v in by_task.values())
                      for s in ("train", "test")},
           **qa_sets}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    print(f"  프레임 아이템 {frame_items['rows']:,} "
          f"({frame_items['bytes']/1e9:.2f} GB)")
    print(f"  학습 {out['totals']['train']:,} · 평가 {out['totals']['test']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
