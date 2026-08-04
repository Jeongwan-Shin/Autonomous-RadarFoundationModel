#!/usr/bin/env python3
"""Autoregressive tracking evaluation: the model's own output becomes its history.

`eval_all_tasks.py` scores items independently, which is the wrong shape for
this task. Tracking is a rollout -- what the model says at t becomes the history
it reads at t+3 -- and a model that is graded with the truth handed back to it
every step is graded on a situation it will never be in.

The history is seeded **empty**, not with the labels. Deployed, there is no
ground-truth history to start from, and seeding with one measures a system that
does not exist. It also means every id in the rollout is the model's own
invention, which is exactly what `idf1` is built to handle: predicted ids are
mapped onto label ids by the assignment explaining the most matched detections,
so a model that calls the first car #2 where the label calls it #1 scores full
marks as long as it keeps calling it #2.

    python -m training.eval_tracking --checkpoint <ckpt> --task track_step_azdeg
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch

from training.task_scorers import idf1, reward_tracking, score_tracking


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


HISTORY_HEADER = "Previous detections:"
CONTINUE = "Continue tracking:"


def rewrite_history(prompt, history, step_seconds=1):
    """Replace the prompt's history block with what the model actually said.

    The instruction line after the block is kept verbatim, because it is what
    selects the output format -- range/azimuth or image box -- and rewriting it
    would silently change the task.
    """
    tail = prompt[prompt.index(CONTINUE):] if CONTINUE in prompt else prompt
    if not history:
        # Nothing seen yet. Saying so is better than omitting the block: the
        # model is trained on prompts that always carry one, and an absent
        # section is a different distribution from an empty one.
        lines = [f"t-{step_seconds}s: none"]
    else:
        # Oldest first, as the built data writes it: t-4s, t-3s, ... t-1s.
        lines = [f"t-{back}s: {text}"
                 for back, text in sorted(history, reverse=True)]
    return f"{HISTORY_HEADER}\n" + "\n".join(lines) + "\n" + tail


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="8B")
    ap.add_argument("--task", default="track_step_azdeg",
                    choices=("track_step_azdeg", "track_step_bbox"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--clips", type=int, default=60)
    ap.add_argument("--history", type=int, default=4,
                    help="how many past steps to carry, matching the data")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--teacher-forcing", action="store_true",
                    help="feed the labels back instead of the model's output, "
                         "to measure how much the rollout costs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    from training.connector import RadarConnector, add_radar_tokens, llm_hidden_size
    from training.instruct_data import InstructDataset, build_collate, load_items
    from training.radar_encoder import RadarEncoder, encoder_kwargs, load_encoder_state
    from training.train_vlm import MODEL_DIR, RadarInjector

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    model_dir = MODEL_DIR[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    weights = os.path.join(args.checkpoint, "model")
    llm = AutoModelForImageTextToText.from_pretrained(
        weights if os.path.isdir(weights) else model_dir,
        dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    pad_id = add_radar_tokens(tokenizer, llm)
    processor.tokenizer = tokenizer
    log(f"weights from {weights}")

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

    # Group the items by clip and order them in time, which is the one thing a
    # shuffled per-item loader cannot give and this evaluation cannot do without.
    items = load_items((args.task,), args.split)
    by_clip = defaultdict(list)
    for item in items:
        by_clip[item["clip_id"]].append(item)
    clips = sorted(by_clip)[: args.clips]
    log(f"{len(clips)} clips, {sum(len(by_clip[c]) for c in clips)} steps")

    dataset = InstructDataset(tasks=(args.task,), split=args.split,
                              processor=processor, tokenizer=tokenizer,
                              n_frames=trained["frames"],
                              radar_tokens=encoder.n_tokens, all_profiles=True)
    collate = build_collate(processor, tokenizer, trained["max_length"])
    injector = RadarInjector(llm.get_input_embeddings(), pad_id)
    header = tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]

    index = {(i["clip_id"], i["frame"]): n
             for n, i in enumerate(dataset.items)}
    sequences, per_step, started = [], [], time.monotonic()

    for done, clip_id in enumerate(clips, 1):
        history, steps = [], sorted(by_clip[clip_id], key=lambda i: i["frame"])
        for item in steps:
            position = index.get((clip_id, item["frame"]))
            if position is None:
                continue
            sample = dataset[position]
            sample = dict(sample)
            sample["user"] = rewrite_history(sample["user"], history)
            batch = collate([sample])

            points = batch.pop("points").to(device, torch.bfloat16)
            radar_mask = batch.pop("radar_mask").to(device)
            sensor = batch.pop("sensor", None)
            sensor = sensor.to(device) if sensor is not None else None
            batch.pop("task", None), batch.pop("clip_id", None)
            labels = batch.pop("labels")
            tensors = {k: v.to(device) for k, v in batch.items()
                       if torch.is_tensor(v)}

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
                                         skip_special_tokens=True).strip()

            with torch.no_grad():
                injector.pending = connector(
                    encoder(points, radar_mask, sensor)["tokens"])
                out = llm.generate(**prompt, max_new_tokens=args.max_new_tokens,
                                   do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id
                                   or tokenizer.eos_token_id)
            generated = tokenizer.decode(out[0, cut:],
                                         skip_special_tokens=True).strip()

            per_step.append(score_tracking(generated, reference, sample["user"]))
            per_step[-1]["reward"] = reward_tracking(generated, reference,
                                                     sample["user"])
            sequences.append((clip_id, generated, reference))

            carried = reference if args.teacher_forcing else generated
            # Anchors are one second apart, so every carried entry ages by one
            # and the oldest falls off once the window is full.
            history = [(back + 1, text) for back, text in history]
            history.append((1, carried))
            history = sorted(history)[: args.history]
        if done % 10 == 0 or done == len(clips):
            rate = done / (time.monotonic() - started) * 60
            log(f"  {done}/{len(clips)} clips  {rate:.1f} clip/min")

    injector.remove()

    by_seq = defaultdict(list)
    for clip_id, generated, reference in sequences:
        by_seq[clip_id].append((generated, reference))
    scores = [idf1(v) for v in by_seq.values()]
    total = {k: sum(s[k] for s in scores) for k in ("id_tp", "id_fp", "id_fn")}
    denom = 2 * total["id_tp"] + total["id_fp"] + total["id_fn"]

    steps = len(per_step)
    fold = lambda k: sum(s.get(k, 0) for s in per_step)
    tp, fp, fn = fold("tp"), fold("fp"), fold("fn")
    checkable, carried = fold("id_checkable"), fold("id_carried")

    result = {
        "task": args.task, "clips": len(by_seq), "steps": steps,
        "teacher_forcing": args.teacher_forcing,
        "detection_f1": (2.0 * tp / (2 * tp + fp + fn)) if (tp or fp or fn) else 0.0,
        "id_carried": (carried / checkable) if checkable else None,
        "idf1_pooled": (2.0 * total["id_tp"] / denom) if denom else 0.0,
        "idf1_per_clip": sum(s["idf1"] for s in scores) / max(len(scores), 1),
        "reward": fold("reward") / max(steps, 1),
    }
    log("")
    log(f"  탐지 F1        {result['detection_f1']:.3f}")
    log(f"  id 유지        " +
        (f"{result['id_carried']:.3f}" if result["id_carried"] is not None else "--"))
    log(f"  IDF1 (합산)    {result['idf1_pooled']:.3f}")
    log(f"  IDF1 (클립평균) {result['idf1_per_clip']:.3f}")
    log(f"  보상 평균      {result['reward']:.3f}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"summary": result,
                       "generations": [{"clip_id": c, "generated": g,
                                        "reference": r}
                                       for c, g, r in sequences]},
                      fh, indent=1, ensure_ascii=False)
        log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
