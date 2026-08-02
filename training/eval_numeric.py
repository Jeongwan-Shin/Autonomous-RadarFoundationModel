#!/usr/bin/env python3
"""Does the number the model writes track the number the radar measured?

Every metric so far is teacher-forced: the model is shown the correct prefix and
scored on the next token. That cannot distinguish a model that reads the radar
from one that has memorised what a plausible answer looks like, because the
prefix carries most of the answer. This generates instead, parses the number out
of what comes back, and correlates it with the truth.

The correlation is the point. Digit accuracy above a prior-only floor says the
model beats a constant; a correlation says its answer moves with the scene. And
running the same generation with another clip's radar spliced in says whether
the movement comes from the radar or from the camera.

Three references are reported alongside, because a bare correlation is easy to
over-read:

  constant   always emit the training median. Correlation 0 by construction; its
             error is what any model has to beat
  shuffled   the same model, another clip's radar. The distance between this and
             `full` is the radar's actual contribution
  oracle     the truth itself, for the error floor imposed by rounding the
             answer into bands

    python -m training.eval_numeric --checkpoint checkpoints/vlm_8B_sw_cmp_coarse \\
        --task radar_probe_coarse --items 400
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def numbers(text):
    return [float(x) for x in NUMBER.findall(text or "")]


def summarise(name, predicted, truth):
    """Error and correlation of one prediction series against the truth."""
    predicted, truth = np.asarray(predicted, float), np.asarray(truth, float)
    ok = np.isfinite(predicted) & np.isfinite(truth)
    predicted, truth = predicted[ok], truth[ok]
    if len(predicted) < 3:
        return None
    # Guarded: a model that emits one constant has zero variance, and numpy
    # would report nan rather than the 0 correlation that actually describes it.
    corr = (float(np.corrcoef(predicted, truth)[0, 1])
            if predicted.std() > 1e-9 and truth.std() > 1e-9 else 0.0)
    denominator = np.maximum(np.abs(truth), 1.0)
    return {"n": int(len(predicted)), "corr": corr,
            "mae": float(np.abs(predicted - truth).mean()),
            "rel": float((np.abs(predicted - truth) / denominator).mean())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="8B")
    ap.add_argument("--task", default="radar_probe")
    ap.add_argument("--items", type=int, default=400)
    # Generation runs one at a time on purpose. Batching needs left padding for
    # a decoder-only model, and left padding moves the video and radar
    # placeholders, which the injector locates by position. One at a time is
    # slower and unambiguous.
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--show", type=int, default=6,
                    help="print this many generated answers next to the truth")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

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
    trained_weights = os.path.join(args.checkpoint, "model")
    source = trained_weights if os.path.isdir(trained_weights) else model_dir
    llm = AutoModelForImageTextToText.from_pretrained(
        source, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    pad_id = add_radar_tokens(tokenizer, llm)
    processor.tokenizer = tokenizer
    log(f"weights from {source}")

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

    dataset = InstructDataset(
        tasks=(args.task,), split="val", processor=processor, tokenizer=tokenizer,
        n_frames=trained["frames"], radar_tokens=encoder.n_tokens,
        samples=args.items, all_profiles=True, radar_dropout=0.0)
    log(f"{len(dataset):,} items of {args.task}")
    collate = build_collate(processor, tokenizer, trained["max_length"])
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate)

    injector = RadarInjector(llm.get_input_embeddings(), pad_id)
    header = tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    results = {mode: {"pred": [], "truth": [], "form": []}
               for mode in ("full", "shuffled")}
    shown = []
    previous = None

    for batch in loader:
        points = batch.pop("points").to(device, torch.bfloat16)
        radar_mask = batch.pop("radar_mask").to(device)
        sensor = batch.pop("sensor", None)
        if sensor is not None:
            sensor = sensor.to(device)
        batch.pop("task", None)
        labels = batch.pop("labels")
        tensors = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}

        # The prompt has to stop where the answer starts, or generation would be
        # handed the answer it is meant to produce.
        ids = tensors["input_ids"]
        cut = None
        row = ids[0].tolist()
        for start in range(len(row) - len(header), -1, -1):
            if row[start:start + len(header)] == header:
                cut = start + len(header)
                break
        if cut is None:
            continue
        prompt = {k: (v[:, :cut] if k in ("input_ids", "attention_mask",
                                          "mm_token_type_ids") else v)
                  for k, v in tensors.items()}
        truth = [numbers(tokenizer.decode(r[r != -100], skip_special_tokens=True))
                 for r in labels]
        # The three probe forms ask for different physical quantities and only
        # one of them -- RCS -- is something a camera cannot measure at all.
        # Pooling them hides which is carrying the correlation.
        asked = tokenizer.decode(ids[0][:cut], skip_special_tokens=True)
        form = ("rcs" if "dBsm" in asked else
                "illuminated" if "illuminate" in asked else "detections")

        with torch.no_grad():
            tokens = connector(encoder(points, radar_mask, sensor)["tokens"])
            # Rolling within the batch is what eval_vlm does, and it silently
            # does nothing at batch 1 -- a single row rolled onto itself is the
            # same row, so `shuffled` would report identical numbers to `full`
            # and read as "the radar changes nothing". Carrying the previous
            # item's radar forward gives a genuine mismatch at any batch size.
            for mode in results:
                if mode == "full":
                    injector.pending = tokens
                elif previous is not None and previous.shape == tokens.shape:
                    injector.pending = previous
                else:
                    continue
                out = llm.generate(**prompt, max_new_tokens=args.max_new_tokens,
                                   do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id
                                   or tokenizer.eos_token_id)
                for b in range(out.shape[0]):
                    text = tokenizer.decode(out[b, cut:], skip_special_tokens=True)
                    got = numbers(text)
                    if mode == "full" and len(shown) < args.show:
                        shown.append((text.strip().replace("\n", " ")[:80],
                                      tokenizer.decode(
                                          labels[b][labels[b] != -100],
                                          skip_special_tokens=True)[:80]))
                    # First number only: the question forms differ in how many
                    # they ask for, and the first is the one every form has.
                    results[mode]["pred"].append(got[0] if got else np.nan)
                    results[mode]["truth"].append(
                        truth[b][0] if truth[b] else np.nan)
                    results[mode]["form"].append(form)
            previous = tokens.clone()

    injector.remove()
    truth = results["full"]["truth"]
    finite = [t for t in truth if np.isfinite(t)]
    median = float(np.median(finite)) if finite else 0.0

    report = {"task": args.task, "checkpoint": args.checkpoint,
              "constant_median": median}
    for mode in results:
        # Each condition keeps its own truth list: `shuffled` has no predecessor
        # for the first item and so is one shorter, and pairing it against the
        # full-length truth would silently misalign every row by one.
        report[mode] = summarise(mode, results[mode]["pred"],
                                 results[mode]["truth"])
    report["constant"] = summarise("constant", [median] * len(truth), truth)

    if shown:
        print("\n  generated vs truth")
        for got, want in shown:
            print(f"    got  : {got}")
            print(f"    truth: {want}")
    print()
    print(f"  {'condition':12s}{'n':>6s}{'corr':>9s}{'MAE':>10s}{'rel err':>10s}")
    for name in ("full", "shuffled", "constant"):
        s = report.get(name)
        if s:
            print(f"  {name:12s}{s['n']:6d}{s['corr']:+9.3f}{s['mae']:10.2f}"
                  f"{s['rel']*100:9.1f}%")

    print(f"\n  by question form ({'full':>16s} {'shuffled':>16s})")
    report["by_form"] = {}
    for form in ("detections", "illuminated", "rcs"):
        cells, entry = "", {}
        for mode in ("full", "shuffled"):
            sel = [i for i, f in enumerate(results[mode]["form"]) if f == form]
            s = summarise(mode, [results[mode]["pred"][i] for i in sel],
                          [results[mode]["truth"][i] for i in sel])
            entry[mode] = s
            cells += (f"  corr {s['corr']:+.3f} MAE {s['mae']:7.2f}" if s
                      else f"{'  (too few)':>28s}")
        report["by_form"][form] = entry
        n = entry["full"]["n"] if entry["full"] else 0
        print(f"    {form:12s} n={n:4d}{cells}")
    if report.get("full") and report.get("shuffled"):
        print(f"\n  radar contribution: correlation "
              f"{report['full']['corr']:+.3f} -> {report['shuffled']['corr']:+.3f} "
              f"when the radar comes from another clip")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
