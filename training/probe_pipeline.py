#!/usr/bin/env python3
"""Where does the radar's RCS go between the encoder and the answer?

Two measurements bracket the problem and disagree by a lot. A linear probe of
the encoder's output tokens recovers a frame's strongest return at R^2 0.88.
The same model, asked in words, produces a number that correlates with the truth
at 0.06-0.14. Somewhere between those two points the quantity is discarded, and
nothing measured so far says where.

Four stages, one probe each, on the same items:

  encoder     the 240 tokens the encoder emits
  connector   the same tokens projected into the language model's embedding
              space. A drop here means the projection loses it
  hidden      the language model's own hidden state at the last prompt position,
              which is what the first answer token is generated from. A drop here
              means the language model receives it and does not carry it forward
  generated   the number the model actually writes, already measured

The stage where R^2 falls is the one to fix. If `hidden` holds the quantity and
`generated` does not, the fault is the digit-by-digit output format and nothing
upstream needs touching.

    python -m training.probe_pipeline --checkpoint checkpoints/vlm_8B_full_resume
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

# The three probe questions alternate by frame; this one asks for the strongest
# return in dBsm, the only quantity of the three a camera cannot measure.
RCS_MARKER = "dBsm"
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


ALPHAS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)


def ridge_r2(features, target, train_frac=0.5, tune_frac=0.25):
    """Held-out R^2 of a linear read of `features`, solved in the dual.

    Dual because the blocks run to 92,160 dimensions against a few hundred rows;
    the primal would be singular and the Gram matrix is small.

    The ridge penalty is chosen on a third split rather than fixed. With 166
    training rows against 92,160 features a fixed alpha of 1.0 overfits so hard
    that held-out R^2 came out at -0.5 for a quantity a better-powered probe
    recovers at +0.88 -- the number said more about the regularisation than about
    the representation.
    """
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    ok = np.isfinite(target)
    features, target = features[ok], target[ok]
    n = len(target)
    fit_end, tune_end = int(n * train_frac), int(n * (train_frac + tune_frac))
    if fit_end < 20 or n - tune_end < 20:
        return float("nan")

    x_fit, y_fit = features[:fit_end], target[:fit_end]
    x_tune, y_tune = features[fit_end:tune_end], target[fit_end:tune_end]
    x_test, y_test = features[tune_end:], target[tune_end:]

    mu, sigma = x_fit.mean(0), x_fit.std(0) + 1e-6
    x_fit, x_tune, x_test = ((x_fit - mu) / sigma, (x_tune - mu) / sigma,
                             (x_test - mu) / sigma)
    offset = y_fit.mean()
    gram = x_fit @ x_fit.T
    eye = np.eye(len(gram))

    def score(x_eval, y_eval, alpha):
        dual = np.linalg.solve(gram + alpha * eye, y_fit - offset)
        prediction = (x_eval @ x_fit.T) @ dual + offset
        residual = ((y_eval - prediction) ** 2).sum()
        total = ((y_eval - y_fit.mean()) ** 2).sum()
        return 1.0 - residual / max(total, 1e-9)

    best = max(ALPHAS, key=lambda a: score(x_tune, y_tune, a))
    return float(score(x_test, y_test, best))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="8B")
    ap.add_argument("--task", default="radar_probe")
    ap.add_argument("--items", type=int, default=1200,
                    help="drawn before the RCS filter, which keeps about a third")
    ap.add_argument("--workers", type=int, default=4)
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
        tasks=(args.task,), split="val", processor=processor, tokenizer=tokenizer,
        n_frames=trained["frames"], radar_tokens=encoder.n_tokens,
        samples=args.items, all_profiles=True, radar_dropout=0.0)
    collate = build_collate(processor, tokenizer, trained["max_length"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.workers, collate_fn=collate)

    injector = RadarInjector(llm.get_input_embeddings(), pad_id)
    header = tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    # `hidden_shuffled` is the control that decides everything downstream. If
    # the hidden state carries RCS just as well when the radar comes from another
    # clip, the quantity is being inferred from the camera and no amount of work
    # on the output format will make the radar matter.
    blocks = {"encoder": [], "connector": [], "hidden": [], "hidden_shuffled": []}
    truth = []
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

        ids = tensors["input_ids"][0].tolist()
        cut = None
        for start in range(len(ids) - len(header), -1, -1):
            if ids[start:start + len(header)] == header:
                cut = start + len(header)
                break
        if cut is None:
            continue
        asked = tokenizer.decode(tensors["input_ids"][0][:cut],
                                 skip_special_tokens=True)
        if RCS_MARKER not in asked:
            continue
        target = tokenizer.decode(labels[0][labels[0] != -100],
                                  skip_special_tokens=True)
        found = NUMBER.findall(target)
        if not found:
            continue

        with torch.no_grad():
            radar = encoder(points, radar_mask, sensor)["tokens"]
            projected = connector(radar)
            prompt = {k: (v[:, :cut] if k in ("input_ids", "attention_mask",
                                              "mm_token_type_ids") else v)
                      for k, v in tensors.items()}
            injector.pending = projected
            # The state the first answer token is sampled from.
            last = llm(**prompt, output_hidden_states=True).hidden_states[-1][0, -1]
            mismatched = None
            if previous is not None and previous.shape == projected.shape:
                injector.pending = previous
                mismatched = llm(**prompt,
                                 output_hidden_states=True).hidden_states[-1][0, -1]

        blocks["encoder"].append(radar[0].float().flatten().cpu().numpy())
        blocks["connector"].append(projected[0].float().mean(0).cpu().numpy())
        blocks["hidden"].append(last.float().cpu().numpy())
        blocks["hidden_shuffled"].append(
            mismatched.float().cpu().numpy() if mismatched is not None
            else np.full(last.shape[0], np.nan, dtype=np.float32))
        truth.append(float(found[0]))
        previous = projected.clone()

    injector.remove()
    log(f"{len(truth)} RCS items")
    if len(truth) < 40:
        log("too few to probe")
        return 1

    report = {"checkpoint": args.checkpoint, "n": len(truth)}
    print()
    print(f"  {'stage':16s}{'R^2':>9s}   what a drop here would mean")
    meaning = {
        "encoder": "the encoder never had it",
        "connector": "the projection into embedding space loses it",
        "hidden": "the language model is given it and drops it",
        "hidden_shuffled": "control: this high means the camera, not the radar",
    }
    for name in ("encoder", "connector", "hidden", "hidden_shuffled"):
        rows = np.asarray(blocks[name], dtype=np.float64)
        keep = np.isfinite(rows).all(axis=1)
        score = ridge_r2(rows[keep], np.asarray(truth)[keep])
        report[name] = score
        print(f"  {name:16s}{score:9.3f}   {meaning[name]}")
    print()
    print("  generated: measured separately by training.eval_numeric; the last")
    print("  reported correlation for this question form was 0.06-0.14")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
