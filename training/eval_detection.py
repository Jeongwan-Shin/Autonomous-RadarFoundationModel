#!/usr/bin/env python3
"""Score tasks 01 and 02 by matching objects, not by comparing strings.

Detection and tracking are currently reported as teacher-forced loss, which
cannot say whether the model found the objects. Two answers listing the same
three cars in a different order are identical as detections and score far apart
as text, and a model that emits a plausible list of the right length scores well
on loss while finding nothing.

So the answer is parsed back into objects and matched against the reference by
position, the way a detector is normally scored:

  precision / recall / F1   at a centre-distance threshold, in metres
  class accuracy            over matched pairs only, so a miss is not counted
                            twice
  range / azimuth error     mean absolute, over matched pairs
  motion accuracy           moving vs stationary, over matched pairs
  id accuracy               task 02 only: does the matched object carry the
                            right track id

Matching is greedy nearest-first on the (x, y) position implied by range and
azimuth. Hungarian assignment would be marginally better and needs scipy on a
handful of points; greedy differs only when two objects are within the threshold
of each other, which the forward-sector geometry makes rare.

    python -m training.eval_detection --checkpoint checkpoints/vlm_8B_long_base
"""

import argparse
import json
import math
import os
import re
import sys
import time

import torch
from torch.utils.data import DataLoader

# "automobile 34 m az +13 deg moving" and, for task 02,
# "#20 automobile 34 m visible 3.1 s"
DETECTION = re.compile(
    r"(?:#(?P<tid>\d+)\s+)?(?P<cls>[a-z_]+)\s+(?P<rng>-?\d+(?:\.\d+)?)\s*m"
    r"(?:\s*az\s*(?P<az>[+-]?\d+(?:\.\d+)?)\s*deg)?"
    r"(?P<motion>\s+moving|\s+stationary)?")
THRESHOLDS = (2.0, 4.0)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_objects(text):
    """Objects as dicts. Azimuth is optional -- task 02 omits it."""
    out = []
    for part in (text or "").split(";"):
        m = DETECTION.search(part)
        if not m:
            continue
        azimuth = m.group("az")
        out.append({
            "tid": int(m.group("tid")) if m.group("tid") else None,
            "cls": m.group("cls"),
            "rng": float(m.group("rng")),
            "az": float(azimuth) if azimuth is not None else 0.0,
            "moving": (m.group("motion") or "").strip() == "moving",
            "has_az": azimuth is not None,
        })
    return out


def position(obj):
    a = math.radians(obj["az"])
    return obj["rng"] * math.cos(a), obj["rng"] * math.sin(a)


def match(predicted, truth, threshold):
    """Greedy nearest-first pairing. Returns [(prediction, truth)] and the counts."""
    pairs, used = [], set()
    candidates = []
    for i, p in enumerate(predicted):
        px, py = position(p)
        for j, t in enumerate(truth):
            tx, ty = position(t)
            d = math.hypot(px - tx, py - ty)
            if d <= threshold:
                candidates.append((d, i, j))
    for d, i, j in sorted(candidates):
        if i in used or ("t", j) in used:
            continue
        used.add(i)
        used.add(("t", j))
        pairs.append((predicted[i], truth[j]))
    return pairs


def score(records, threshold):
    tp = fp = fn = 0
    cls_ok = motion_ok = id_ok = id_total = matched = 0
    range_err = az_err = 0.0
    for predicted, truth in records:
        pairs = match(predicted, truth, threshold)
        tp += len(pairs)
        fp += len(predicted) - len(pairs)
        fn += len(truth) - len(pairs)
        for p, t in pairs:
            matched += 1
            cls_ok += p["cls"] == t["cls"]
            motion_ok += p["moving"] == t["moving"]
            range_err += abs(p["rng"] - t["rng"])
            if p["has_az"] and t["has_az"]:
                az_err += abs(p["az"] - t["az"])
            if t["tid"] is not None:
                id_total += 1
                id_ok += p["tid"] == t["tid"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "threshold_m": threshold, "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "class_acc": cls_ok / max(matched, 1),
        "motion_acc": motion_ok / max(matched, 1),
        "range_mae": range_err / max(matched, 1),
        "azimuth_mae": az_err / max(matched, 1),
        "id_acc": id_ok / max(id_total, 1) if id_total else None,
    }


def self_test():
    """The matcher has to be right before any model is judged by it."""
    truth = parse_objects("automobile 34 m az +13 deg moving; "
                          "automobile 68 m az +9 deg stationary")
    assert len(truth) == 2 and truth[0]["cls"] == "automobile"
    assert truth[0]["moving"] and not truth[1]["moving"]

    perfect = score([(truth, truth)], 2.0)
    assert perfect["f1"] == 1.0 and perfect["class_acc"] == 1.0, perfect

    # Order must not matter; these are sets of objects, not sequences.
    reversed_ = score([(truth[::-1], truth)], 2.0)
    assert reversed_["f1"] == 1.0, reversed_

    # A miss and a false positive, not a lucky match.
    wrong = parse_objects("person 200 m az -40 deg moving")
    missed = score([(wrong, truth)], 2.0)
    assert missed["tp"] == 0 and missed["fp"] == 1 and missed["fn"] == 2, missed

    # Just inside and just outside the threshold.
    near = parse_objects("automobile 35 m az +13 deg moving")
    assert score([(near, truth)], 2.0)["tp"] == 1
    far = parse_objects("automobile 40 m az +13 deg moving")
    assert score([(far, truth)], 2.0)["tp"] == 0

    # Track ids are read, and a wrong id is still a correct detection.
    tracks = parse_objects("#20 automobile 34 m visible 3.1 s")
    assert tracks[0]["tid"] == 20 and not tracks[0]["has_az"]
    other = parse_objects("#77 automobile 34 m visible 1.0 s")
    s = score([(other, tracks)], 2.0)
    assert s["tp"] == 1 and s["id_acc"] == 0.0, s
    print("matcher self-test passed")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint")
    ap.add_argument("--model", default="8B")
    ap.add_argument("--task", default="det_objects",
                    choices=("det_objects", "track_identity"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--items", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--show", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        self_test()
        return 0
    if not args.checkpoint:
        ap.error("--checkpoint is required unless --self-test")
    self_test()

    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    from training.connector import RadarConnector, add_radar_tokens, llm_hidden_size
    from training.instruct_data import InstructDataset, build_collate
    from training.radar_encoder import (RadarEncoder, encoder_kwargs,
                                        load_encoder_state)
    from training.train_vlm import MODEL_DIR, RadarInjector

    torch.cuda.set_device(0)
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
    lora_dir = os.path.join(args.checkpoint, "lora")
    if os.path.isdir(lora_dir):
        from peft import PeftModel
        llm = PeftModel.from_pretrained(llm, lora_dir)
    llm.eval()
    log(f"weights from {source}")

    dataset = InstructDataset(
        tasks=(args.task,), split=args.split, processor=processor,
        tokenizer=tokenizer, n_frames=trained["frames"],
        radar_tokens=encoder.n_tokens, samples=args.items,
        all_profiles=True, radar_dropout=0.0)
    log(f"{len(dataset):,} items of {args.task} ({args.split})")
    collate = build_collate(processor, tokenizer, trained["max_length"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.workers, collate_fn=collate)

    injector = RadarInjector(llm.get_input_embeddings(), pad_id)
    header = tokenizer("<|im_start|>assistant\n",
                       add_special_tokens=False)["input_ids"]
    records, shown = [], []

    for batch in loader:
        points = batch.pop("points").to(device, torch.bfloat16)
        radar_mask = batch.pop("radar_mask").to(device)
        sensor = batch.pop("sensor", None)
        if sensor is not None:
            sensor = sensor.to(device)
        batch.pop("task", None)
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

        with torch.no_grad():
            injector.pending = connector(
                encoder(points, radar_mask, sensor)["tokens"])
            out = llm.generate(**prompt, max_new_tokens=args.max_new_tokens,
                               do_sample=False,
                               pad_token_id=tokenizer.pad_token_id
                               or tokenizer.eos_token_id)
        text = tokenizer.decode(out[0, cut:], skip_special_tokens=True)
        records.append((parse_objects(text), parse_objects(reference)))
        if len(shown) < args.show:
            shown.append((text.strip()[:110], reference[:110]))

    injector.remove()
    if not records:
        log("nothing generated")
        return 1

    if shown:
        print("\n  generated vs reference")
        for got, want in shown:
            print(f"    got  : {got}")
            print(f"    truth: {want}")

    report = {"task": args.task, "split": args.split,
              "checkpoint": args.checkpoint, "n": len(records), "by_threshold": {}}
    print(f"\n  {len(records)} items")
    print(f"  {'thresh':>7s}{'precision':>11s}{'recall':>9s}{'F1':>8s}"
          f"{'class':>8s}{'motion':>8s}{'range MAE':>11s}{'az MAE':>9s}{'id acc':>9s}")
    for threshold in THRESHOLDS:
        s = score(records, threshold)
        report["by_threshold"][str(threshold)] = s
        idacc = "      --" if s["id_acc"] is None else f"{s['id_acc']*100:8.1f}%"
        print(f"  {threshold:6.0f}m{s['precision']*100:10.1f}%{s['recall']*100:8.1f}%"
              f"{s['f1']*100:7.1f}%{s['class_acc']*100:7.1f}%"
              f"{s['motion_acc']*100:7.1f}%{s['range_mae']:10.2f}m"
              f"{s['azimuth_mae']:8.1f}°{idacc}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
