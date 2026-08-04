#!/usr/bin/env python3
"""태스크마다 실제 아이템 하나에서 프롬프트·정답·근거를 함께 뽑는다.

`_cot` 변형에서 뽑는다. CoT 정답의 `answer` 필드는 평문 변형의 정답 문자열과
글자 그대로 같으므로(frame_objects.py 의 emit 참고), 한 아이템만으로 두 변형을
모두 보여줄 수 있다. 평문과 CoT 를 따로 뽑으면 서로 다른 클립이 잡혀 근거의
속도와 정답의 이동거리가 어긋나 보인다.

`build_dataset_report` 가 이 결과를 읽어 태스크별 페이지를 채운다.

    python -m datatools.task_detail
"""
import json
import os
import re

from transformers import AutoProcessor, AutoTokenizer
from training.instruct_data import InstructDataset, WINDOWS, INSTANT_TASKS
from training.train_vlm import MODEL_DIR

TASKS = ["det_objects_azdeg", "det_objects_3dbbox", "track_step_azdeg",
         "track_step_bbox", "plan_ego_xy", "plan_ego_control",
         "agent_traj_azdeg", "agent_traj_bbox", "motion_seg_azdeg",
         "motion_seg_bbox", "qa"]
names = tuple(t + "_cot" for t in TASKS)
PAD = re.compile(r"<\|(radar|vision|video|image)_(start|end|pad)\|>")
md = MODEL_DIR["8B"]
tok = AutoTokenizer.from_pretrained(md)
proc = AutoProcessor.from_pretrained(md); proc.tokenizer = tok
ds = InstructDataset(tasks=names, split="train", processor=proc, tokenizer=tok,
                     samples=3000, all_profiles=True)
print(f"dataset {len(ds):,}", flush=True)
got = {}
for i in range(len(ds)):
    task = ds.items[i]["task"]
    if task in got or task not in names:
        continue
    s = ds[i]
    try:
        payload = json.loads(s["target"])
    except Exception:
        continue          # 잘린 JSON 은 예시로 쓸 수 없다
    got[task] = (s, payload)
    print(f"  {task}", flush=True)
    if len(got) == len(names):
        break
out = {}
for task in TASKS:
    hit = got.get(task + "_cot")
    if hit is None:
        print(f"  !! {task} 표본 없음", flush=True)
        continue
    s, payload = hit
    lines = s["user"].split("\n")
    instruction = "\n".join(l for l in lines
                            if not l.startswith(("Sensors", "Ego motion")))
    out[task] = {
        "clip_id": s["clip_id"],
        "sensors": next((l for l in lines if l.startswith("Sensors")), ""),
        "ego": next((l for l in lines if l.startswith("Ego motion")), ""),
        "instruction": re.sub(r"\n{3,}", "\n\n", PAD.sub("", instruction)).strip(),
        "target": payload.get("answer", ""),
        "cot": payload,
        "frames": len(s["frames"]),
        "window": WINDOWS.get(task) or ("instant" if task in INSTANT_TASKS else "clip"),
        "user_tokens": len(tok(s["user"])["input_ids"]),
        "radar_tokens": 240,
    }
out_path = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "runs", "01_data_prep",
    "task_detail.json")
json.dump(out, open(out_path, "w"),
          indent=1, ensure_ascii=False)
print(f"wrote {out_path}  ({len(out)} tasks)", flush=True)

