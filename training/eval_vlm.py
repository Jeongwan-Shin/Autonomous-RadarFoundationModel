#!/usr/bin/env python3
"""Evaluate a trained checkpoint, and check whether the radar pathway is used.

Held-out loss on its own cannot answer the question this project exists to
answer. The vision tower sees the same scene the radar does, and most of the
text -- all of `qa`, `objects` and `ego_maneuver` -- is derivable from the video
alone. A model that ignores its radar tokens entirely would still score well, and
the loss curve would look fine.

So every metric is reported under three conditions:

  full        as trained
  zeroed      radar embeddings replaced with zeros. Cheap, but out of
              distribution: the drop conflates "radar mattered" with "the model
              met an input it never saw during training".
  shuffled    radar tokens taken from a different clip in the batch. Input
              statistics are unchanged and only the correspondence is broken, so
              a drop here is evidence that the model was using the radar to say
              something about *this* scene.

`shuffled` is the one to read. If it matches `full`, the radar pathway is
decorative regardless of what the training loss did.

Per task:
  qa                    exact-match accuracy on the answer letter
  desc_*, ood_reasoning teacher-forced loss and perplexity

    python -m training.eval_vlm --model 8B --checkpoint checkpoints/vlm_8B_align
"""

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

ABLATIONS = ("full", "zeroed", "shuffled")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def apply_ablation(radar_embeddings, mode):
    if mode == "full":
        return radar_embeddings
    if mode == "zeroed":
        return torch.zeros_like(radar_embeddings)
    if mode == "shuffled":
        batch = radar_embeddings.shape[0]
        if batch < 2:
            # Rolling a single-row batch returns it unchanged, which would report
            # a fake "no effect". Better to skip than to mislead.
            return None
        return radar_embeddings.roll(shifts=1, dims=0)
    raise ValueError(mode)


@torch.no_grad()
def evaluate(llm, encoder, connector, injector, loader, device, tokenizer,
             max_batches=0, ablations=ABLATIONS):
    llm.eval()
    stats = {a: {} for a in ablations}
    # Digit tokens, so the numeric part of an answer can be scored on its own.
    # Whole-sentence loss is dominated by the template -- "N detections, M
    # moving." is mostly fixed text -- and a model that memorised the skeleton
    # while ignoring the radar still scores well on it. The digits are where the
    # clip-specific information actually lives.
    digit_ids = set()
    for token, index in tokenizer.get_vocab().items():
        stripped = token.lstrip("\u0120 ")
        if stripped and all(c.isdigit() for c in stripped):
            digit_ids.add(index)
    digit_tensor = torch.tensor(sorted(digit_ids), device=next(llm.parameters()).device)

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        points = batch.pop("points").to(device, torch.bfloat16)
        radar_mask = batch.pop("radar_mask").to(device)
        sensor = batch.pop("sensor", None)
        if sensor is not None:
            sensor = sensor.to(device)
        tasks = batch.pop("task")
        tensors = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}

        radar_tokens = connector(encoder(points, radar_mask, sensor)["tokens"])

        for mode in ablations:
            ablated = apply_ablation(radar_tokens, mode)
            if ablated is None:
                continue
            injector.pending = ablated
            out = llm(**tensors)

            labels = tensors["labels"]
            shift_logits = out.logits[:, :-1]
            shift_labels = labels[:, 1:]
            per_token = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]).float(),
                shift_labels.reshape(-1), reduction="none",
                ignore_index=-100).view(shift_labels.shape)
            valid = shift_labels != -100

            is_digit = torch.isin(shift_labels, digit_tensor)
            for b, task in enumerate(tasks):
                bucket = stats[mode].setdefault(
                    task, {"loss": 0.0, "tokens": 0, "correct": 0, "n": 0,
                           "digit_loss": 0.0, "digit_tokens": 0,
                           "digit_correct": 0})
                n_tokens = int(valid[b].sum())
                if n_tokens:
                    bucket["loss"] += float(per_token[b][valid[b]].sum())
                    bucket["tokens"] += n_tokens
                numeric = valid[b] & is_digit[b]
                if numeric.any():
                    bucket["digit_loss"] += float(per_token[b][numeric].sum())
                    bucket["digit_tokens"] += int(numeric.sum())
                    predicted = shift_logits[b][numeric].argmax(-1)
                    bucket["digit_correct"] += int(
                        (predicted == shift_labels[b][numeric]).sum())
                bucket["n"] += 1

                if task == "qa" and n_tokens:
                    # The answer is a single letter, so the first supervised
                    # position carries the whole decision.
                    first = valid[b].nonzero()[0, 0]
                    predicted = shift_logits[b, first].argmax().item()
                    target = shift_labels[b, first].item()
                    bucket["correct"] += int(predicted == target)

    llm.train()
    summary = {}
    for mode, per_task in stats.items():
        summary[mode] = {}
        for task, bucket in per_task.items():
            mean_loss = bucket["loss"] / max(bucket["tokens"], 1)
            entry = {"loss": mean_loss,
                     "ppl": float(torch.exp(torch.tensor(mean_loss))),
                     "n": bucket["n"]}
            if bucket["digit_tokens"]:
                entry["digit_loss"] = bucket["digit_loss"] / bucket["digit_tokens"]
                entry["digit_acc"] = bucket["digit_correct"] / bucket["digit_tokens"]
            if task == "qa":
                entry["accuracy"] = bucket["correct"] / max(bucket["n"], 1)
            summary[mode][task] = entry
    return summary


def report(summary):
    tasks = sorted({t for m in summary.values() for t in m})
    print("\n" + "=" * 78)
    header = f"{'task':24s}" + "".join(f"{m:>17s}" for m in summary)
    print(header)
    print("-" * 78)
    for task in tasks:
        row = f"{task:24s}"
        for mode in summary:
            entry = summary[mode].get(task)
            if entry is None:
                row += f"{'-':>17s}"
            elif "accuracy" in entry:
                row += f"{entry['accuracy']*100:>12.1f}% acc"
            else:
                row += f"{entry['loss']:>13.4f} nll"
        print(row)

    if "full" in summary and "shuffled" in summary:
        print("\nradar dependence (shuffled minus full; positive means the model "
              "was using it)")
        for task in tasks:
            a = summary["full"].get(task)
            b = summary["shuffled"].get(task)
            if not a or not b:
                continue
            extra = ""
            if "digit_acc" in a and "digit_acc" in b:
                extra = (f"   digits {(a['digit_acc']-b['digit_acc'])*100:+6.2f} %p"
                         f"  ({a['digit_acc']*100:.1f}% -> {b['digit_acc']*100:.1f}%)")
            if "accuracy" in a:
                delta = (a["accuracy"] - b["accuracy"]) * 100
                print(f"  {task:24s} {delta:+7.2f} accuracy points{extra}")
            else:
                print(f"  {task:24s} {b['loss'] - a['loss']:+7.4f} nll{extra}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="8B")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tasks", default="qa,description,ood_reasoning")
    ap.add_argument("--all-profiles", action="store_true")
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    from training.connector import RadarConnector, add_radar_tokens, llm_hidden_size
    from training.instruct_data import (InstructDataset, build_collate,
                                        expand_tasks)
    from training.radar_encoder import (RadarEncoder, encoder_kwargs,
                                        load_encoder_state)
    from training.train_vlm import MODEL_DIR, RadarInjector

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    model_dir = MODEL_DIR[args.model]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    # A `full` checkpoint carries every weight; `align`/`joint` carry only the
    # adapters and read the base model. Loading the base model for a full run
    # would silently evaluate the untrained network.
    trained_weights = os.path.join(args.checkpoint, "model")
    weight_source = trained_weights if os.path.isdir(trained_weights) else model_dir
    if weight_source != model_dir:
        log(f"loading fully fine-tuned weights from {weight_source}")
    llm = AutoModelForImageTextToText.from_pretrained(
        weight_source, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    # The saved model already has the radar rows, so the resize below is a no-op
    # for it and a real resize for the base model.
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
    trained["radar_tokens"] = encoder.n_tokens
    connector = RadarConnector(trained["radar_dim"], llm_hidden_size(model_dir))
    connector.load_state_dict(state["connector"])
    connector = connector.to(device).to(torch.bfloat16).eval()

    lora_dir = os.path.join(args.checkpoint, "lora")
    if os.path.isdir(lora_dir):
        from peft import PeftModel
        llm = PeftModel.from_pretrained(llm, lora_dir)
        log(f"LoRA adapters loaded from {lora_dir}")

    # Evaluation never drops the radar: the zeroed/shuffled ablations below are
    # how sensor reliance is measured, and a second source of blanking would mix
    # into them.
    dataset = InstructDataset(tasks=expand_tasks(args.tasks), split="val",
                              processor=processor, tokenizer=tokenizer,
                              n_frames=trained["frames"],
                              radar_tokens=trained["radar_tokens"],
                              samples=args.samples,
                              all_profiles=args.all_profiles,
                              radar_dropout=0.0)
    log(f"val set {len(dataset):,} items: {dataset.task_counts()}")
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers,
                        collate_fn=build_collate(processor, tokenizer,
                                                 trained["max_length"]))

    injector = RadarInjector(llm.get_input_embeddings(), pad_id)
    summary = evaluate(llm, encoder, connector, injector, loader, device,
                       tokenizer, args.max_batches)
    injector.remove()
    report(summary)

    out = args.out or os.path.join(args.checkpoint, "eval.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
