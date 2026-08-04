#!/usr/bin/env python3
"""Generate and score every task on its own terms, with a radar control.

`eval_vlm` reports teacher-forced loss for all eleven tasks, which ranks them
against each other and describes none of them. This generates the answer the
model would actually produce and hands it to the scorer that fits the task --
detection matching for object lists, displacement error for waypoints,
correlation for quantities, exact match for the multiple choice.

Every task is also run with another clip's radar spliced in. The distance
between the two columns is what the radar is contributing to that task, which is
the question this project exists to answer and which no single-column table can
show.

    python -m training.eval_all_tasks --checkpoint checkpoints/vlm_8B_long_base \\
        --tasks det_objects,plan_ego --items 200
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.task_scorers import scorer_for, summarise

ALL = ("det_objects_azdeg", "det_objects_3dbbox", "track_step_azdeg",
       "track_step_bbox", "plan_ego_xy", "plan_ego_control",
       "agent_traj_azdeg", "agent_traj_bbox",
       "motion_seg_azdeg", "motion_seg_bbox", "radar_transfer", "qa",
       "desc_radar", "desc_complementarity", "desc_objects",
       "desc_ego_maneuver", "desc_clip_summary", "radar_probe")
# How long an answer each task needs. Object lists run to eight entries; the
# tracking answers carry an id and, in the bbox form, four coordinates each, so
# they need more room than the detection ones. A task missing from this table
# silently gets the short default and has its answer truncated, which is why the
# names here have to track the task list rather than drift behind it.
MAX_NEW = {"det_objects_azdeg": 200, "det_objects_3dbbox": 240,
           "track_step_azdeg": 260, "track_step_bbox": 300,
           "motion_seg_azdeg": 280, "motion_seg_bbox": 320,
           "agent_traj_azdeg": 100, "agent_traj_bbox": 120,
           "plan_ego_xy": 80, "plan_ego_control": 110,
           "desc_radar": 120, "desc_complementarity": 120, "desc_objects": 120,
           "desc_ego_maneuver": 80, "desc_clip_summary": 160, "retrieval": 60,
           "qa": 8}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Phrases unique to one question form, most specific first. Several tasks
# alternate between questions whose answers differ in magnitude by a factor of a
# hundred; pooled, the correlation just detects which question was asked and
# reads near 1.0 while the model tracks nothing. Grouping by form scores each
# against its own scale, and for `radar_probe` it is what separates the two
# forms the encoder was supervised on from the one it was not.
QUESTION_FORMS = (
    ("dBsm", "rcs"),
    ("camera-only", "illuminated"),
    ("azimuth is the nearest radar return", "bearing"),
    ("left of centre", "lateral"),
    ("fastest approaching", "closing"),
    ("within 20 m", "near_far"),
    ("leftmost", "spread"),
    ("nearest object the radar illuminates, in metres", "obj_range"),
    ("azimuth is the nearest object the radar", "obj_azimuth"),
    ("radial velocity of the nearest object", "obj_closing"),
    ("second nearest radar-illuminated", "obj_gap"),
    ("how many have moving returns", "obj_moving"),
    # Before the catch-all: `radar_transfer` asks this one as "...at the 90th
    # percentile of its detections", so matching on "detections" first pooled a
    # range in metres with a count in the hundreds and inflated the correlation.
    ("How far out does", "reach"),
    ("detections", "detections"),
)


def question_form(asked):
    return next((f for phrase, f in QUESTION_FORMS if phrase in asked), "main")


def load_model(args):
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    from training.connector import RadarConnector, add_radar_tokens, llm_hidden_size
    from training.radar_encoder import (RadarEncoder, encoder_kwargs,
                                        load_encoder_state)
    from training.train_vlm import MODEL_DIR

    device = torch.device("cuda", 0)
    model_dir = MODEL_DIR[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    weights = os.path.join(args.checkpoint, "model")
    source = weights if os.path.isdir(weights) else model_dir
    llm = AutoModelForImageTextToText.from_pretrained(
        source, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    pad_id = add_radar_tokens(tokenizer, llm)
    processor.tokenizer = tokenizer

    state = torch.load(os.path.join(args.checkpoint, "adapters.pt"),
                       map_location="cpu")
    trained = state["args"]
    encoder = RadarEncoder(**{"dim": trained["radar_dim"],
                              "n_frames": trained["frames"],
                              **{k: v for k, v in encoder_kwargs(trained).items()
                                 if k not in ("dim", "n_frames")}})
    load_encoder_state(encoder, state["encoder"])
    encoder = encoder.to(device).to(torch.bfloat16).eval()
    connector = RadarConnector(trained["radar_dim"], llm_hidden_size(model_dir))
    connector.load_state_dict(state["connector"])
    connector = connector.to(device).to(torch.bfloat16).eval()
    lora = os.path.join(args.checkpoint, "lora")
    if os.path.isdir(lora):
        from peft import PeftModel
        llm = PeftModel.from_pretrained(llm, lora)
    llm.eval()
    log(f"weights from {source}")
    return tokenizer, processor, llm, encoder, connector, pad_id, trained, device


def run_task(task, args, loaded):
    from training.instruct_data import InstructDataset, build_collate
    from training.train_vlm import RadarInjector
    tokenizer, processor, llm, encoder, connector, pad_id, trained, device = loaded

    # `load_items` keys the description kinds off the group name "description"
    # and only then splits them into desc_radar, desc_objects and the rest.
    # Asking for "desc_radar" directly matches no branch and returns nothing.
    group = "description" if task.startswith("desc_") else task
    dataset = InstructDataset(
        tasks=(group,), split=args.split, processor=processor, tokenizer=tokenizer,
        n_frames=trained["frames"], radar_tokens=encoder.n_tokens,
        samples=0 if group != task else args.items,
        all_profiles=True, radar_dropout=0.0)
    if group != task:
        dataset.items = [i for i in dataset.items if i["task"] == task][: args.items]
    if not len(dataset):
        log(f"{task}: no items in split={args.split}")
        return None
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.workers,
                        collate_fn=build_collate(processor, tokenizer,
                                                 trained["max_length"]))
    injector = RadarInjector(llm.get_input_embeddings(), pad_id)
    header = tokenizer("<|im_start|>assistant\n",
                       add_special_tokens=False)["input_ids"]
    scorer = scorer_for(task)
    records = {"full": [], "shuffled": []}
    # Every generation, kept. Scoring has been wrong four times so far -- pooled
    # question forms, a shuffle that shuffled nothing, a match threshold on an
    # axis the task does not have, an untuned ridge penalty -- and each fix cost
    # a full re-run of inference because only the aggregate survived. The text
    # is what the GPU actually produced; the metric is an opinion about it, and
    # opinions get revised.
    generations = []
    pairs = {"full": [], "shuffled": []}
    previous, shown = None, []

    for batch in loader:
        points = batch.pop("points").to(device, torch.bfloat16)
        radar_mask = batch.pop("radar_mask").to(device)
        sensor = batch.pop("sensor", None)
        if sensor is not None:
            sensor = sensor.to(device)
        batch.pop("task", None)
        clip_ids = batch.pop("clip_id", None)
        clip_id = clip_ids[0] if clip_ids else None
        labels = batch.pop("labels")
        tensors = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}

        ids = tensors["input_ids"][0].tolist()
        cut = None
        for start in range(len(ids) - len(header), -1, -1):
            if ids[start:start + len(header)] == header:
                cut = start + len(header)
                break
        if cut is None:
            continue
        prompt = {k: (v[:, :cut] if k in ("input_ids", "attention_mask",
                                          "mm_token_type_ids") else v)
                  for k, v in tensors.items()}
        reference = tokenizer.decode(labels[0][labels[0] != -100],
                                     skip_special_tokens=True)
        # radar_probe alternates between three questions whose answers differ in
        # magnitude by a factor of a hundred. Pooling them, the correlation just
        # detects which question was asked and reads 1.0 while the model tracks
        # nothing. Group by question so each is scored against its own scale.
        asked = tokenizer.decode(tensors["input_ids"][0][:cut],
                                 skip_special_tokens=True)
        form = question_form(asked)

        with torch.no_grad():
            tokens = connector(encoder(points, radar_mask, sensor)["tokens"])
            for mode in ("full", "shuffled"):
                if mode == "full":
                    injector.pending = tokens
                elif previous is not None and previous.shape == tokens.shape:
                    # Rolling within the batch does nothing at batch 1 -- a
                    # single row rolled onto itself is the same row -- so the
                    # previous item's radar is carried forward instead.
                    injector.pending = previous
                else:
                    continue
                out = llm.generate(**prompt,
                                   max_new_tokens=MAX_NEW.get(task, 48),
                                   do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id
                                   or tokenizer.eos_token_id)
                text = tokenizer.decode(out[0, cut:], skip_special_tokens=True)
                result = scorer(text, reference)
                records[mode].append(result)
                generations.append({
                    "task": task, "mode": mode, "form": form,
                    "clip_id": clip_id, "prompt": asked[-220:],
                    "generated": text.strip(), "reference": reference.strip(),
                })
                if "pred" in result and result["pred"] is not None:
                    pairs[mode].append((result["pred"], result["truth"], form))
                if mode == "full" and len(shown) < args.show:
                    shown.append((text.strip().replace("\n", " ")[:100],
                                  reference[:100]))
            previous = tokens.clone()

    injector.remove()
    out = {"task": task, "n": len(records["full"]), "examples": shown,
           "generations": generations}
    def correlate(rows):
        if len(rows) < 5:
            return None
        a = np.array([p for p, _, _ in rows], float)
        b = np.array([t for _, t, _ in rows], float)
        return (float(np.corrcoef(a, b)[0, 1])
                if a.std() > 1e-9 and b.std() > 1e-9 else 0.0)

    for mode in ("full", "shuffled"):
        rows = pairs[mode]
        forms = sorted({f for _, _, f in rows})
        by_form = {f: correlate([r for r in rows if r[2] == f]) for f in forms}
        # The headline is the mean of the per-question correlations, never the
        # pooled one.
        valid = [v for v in by_form.values() if v is not None]
        corr = sum(valid) / len(valid) if valid else None
        out[mode] = summarise(task, records[mode], correlation=corr)
        if out[mode]:
            out[mode]["by_form"] = by_form
            out[mode]["pooled_corr"] = correlate(rows)
    return out


HEADLINE = {"detection": ("f1", "F1", 100, "%"),
            "waypoints": ("displacement_mae_m", "위치오차", 1, " m"),
            "trajectory": ("range_mae_m", "거리오차", 1, " m"),
            "quantity": ("corr", "상관", 1, ""),
            "tags": ("f1", "F1", 100, "%"),
            "choice": ("accuracy", "정확도", 100, "%")}


def report(results):
    print("\n" + "=" * 96)
    print(f"  {'task':22s}{'지표':>12s}{'n':>6s}{'full':>12s}{'shuffled':>12s}"
          f"{'레이더 기여':>14s}")
    print("  " + "-" * 92)
    for r in results:
        if not r or not r.get("full"):
            continue
        kind = r["full"].get("metric", "")
        spec = HEADLINE.get(kind)
        if not spec:
            print(f"  {r['task']:22s}{kind:>12s}{r['n']:>6d}"
                  f"{'loss only':>12s}{'--':>12s}{'--':>14s}")
            continue
        key, label, scale, unit = spec
        a = r["full"].get(key)
        b = r.get("shuffled", {}).get(key)
        fa = "--" if a is None else f"{a*scale:.1f}{unit}"
        fb = "--" if b is None else f"{b*scale:.1f}{unit}"
        if a is None or b is None:
            delta = "--"
        else:
            # For errors, lower is better, so the radar helps when shuffling
            # makes it worse; for scores it is the other way round.
            lower_better = key in ("displacement_mae_m", "range_mae_m")
            diff = (b - a) if lower_better else (a - b)
            delta = f"{diff*scale:+.1f}{unit}"
        print(f"  {r['task']:22s}{label:>12s}{r['n']:>6d}{fa:>12s}{fb:>12s}"
              f"{delta:>14s}")
    print("=" * 96)
    print("  레이더 기여: full과 shuffled(다른 클립의 레이더)의 차이. 0에 가까우면")
    print("  그 태스크는 레이더 없이 풀리고 있다는 뜻입니다.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="8B")
    ap.add_argument("--tasks", default=",".join(ALL))
    ap.add_argument("--split", default="val")
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--show", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    torch.cuda.set_device(0)
    loaded = load_model(args)
    results = []
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        log(f"--- {task}")
        r = run_task(task, args, loaded)
        if r:
            results.append(r)
            for got, want in r["examples"]:
                print(f"      got  : {got}")
                print(f"      truth: {want}")
    report(results)
    if args.out:
        # Generations go beside the summary, not inside it: they are ~200x
        # larger and a summary should stay readable. One JSON object per line,
        # so `rescore_generations.py` can stream them without loading the file.
        stream = os.path.splitext(args.out)[0] + ".generations.jsonl"
        written = 0
        with open(stream, "w") as fh:
            for r in results:
                for g in r.pop("generations", []):
                    fh.write(json.dumps(g, ensure_ascii=False) + "\n")
                    written += 1
        log(f"wrote {stream}  ({written:,} generations)")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2, default=float)
        log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
