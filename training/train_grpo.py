#!/usr/bin/env python3
"""GRPO on the tasks whose answer a program can check.

Why reinforcement learning is the right tool here, and only here. Probing the
stack found the radar's strongest return recoverable at R^2 0.65 from the hidden
state that generates the first answer token, collapsing to 0.005 when another
clip's radar is spliced in -- so the quantity is present, radar-borne, and
carried all the way to the output layer. The number the model then writes
correlates with the truth far below that. The representation is not the problem;
the objective is. Cross-entropy over digit tokens scores 639 against a true 638
as harshly as 100, because it only ever asks whether the argmax matched.

A verifiable reward asks the question the loss cannot: how far off was the
number. Nothing here needs a preference model or a human -- the answers were
computed from the labels, so a checker can score them exactly.

Group Relative Policy Optimisation, rather than PPO: sampling G completions per
prompt and normalising the reward within that group replaces the value network,
which would otherwise be a second 8 B model in memory. The behaviour policy is
the policy at generation time, and the clipped ratio against it is what keeps an
update from running away -- so no frozen reference copy is needed either.

    torchrun --nproc_per_node=1 -m training.train_grpo \\
        --init checkpoints/vlm_8B_long_base --reward relative --steps 200
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


# --------------------------------------------------------------------------
# rewards
# --------------------------------------------------------------------------

def reward_relative(generated, reference):
    """1 at exact, falling linearly with relative error. The graded signal the
    token loss cannot express."""
    got, want = NUMBER.findall(generated), NUMBER.findall(reference)
    if not got or not want:
        return 0.0
    p, t = float(got[0]), float(want[0])
    return float(max(0.0, 1.0 - abs(p - t) / max(abs(t), 1.0)))


def reward_exact(generated, reference):
    """1 only on an exact match. The control that says whether grading matters."""
    got, want = NUMBER.findall(generated), NUMBER.findall(reference)
    return float(bool(got) and bool(want) and float(got[0]) == float(want[0]))


def reward_tolerance(generated, reference, band=0.10):
    """1 inside a 10% band, 0 outside. Between the other two: forgiving about
    precision, still binary."""
    got, want = NUMBER.findall(generated), NUMBER.findall(reference)
    if not got or not want:
        return 0.0
    p, t = float(got[0]), float(want[0])
    return float(abs(p - t) <= band * max(abs(t), 1.0))


def reward_all_numbers(generated, reference):
    """Every number in the answer, not just the first. `radar_probe` asks two
    quantities at once and rewarding only the first leaves the second free."""
    got, want = NUMBER.findall(generated), NUMBER.findall(reference)
    if not got or not want:
        return 0.0
    scores = []
    for i, w in enumerate(want):
        if i >= len(got):
            scores.append(0.0)
            continue
        t, p = float(w), float(got[i])
        scores.append(max(0.0, 1.0 - abs(p - t) / max(abs(t), 1.0)))
    return float(sum(scores) / len(scores))


REWARDS = {"relative": reward_relative, "exact": reward_exact,
           "tolerance": reward_tolerance, "all_numbers": reward_all_numbers}


# --------------------------------------------------------------------------
# log probabilities
# --------------------------------------------------------------------------

def extend_token_types(extra, length):
    """Mark the generated tokens as text.

    `mm_token_type_ids` is built for the prompt and marks which positions carry
    video or radar. Scoring the prompt plus its continuation makes the sequence
    longer, and Qwen3-VL indexes that tensor with the attention mask when it
    builds 3-D rope positions -- a length mismatch raises there rather than
    anywhere obvious. Continuations are plain text, so the padding value is 0.
    """
    types = extra.get("mm_token_type_ids")
    if types is None or types.shape[1] >= length:
        return extra
    pad = torch.zeros(types.shape[0], length - types.shape[1],
                      dtype=types.dtype, device=types.device)
    return {**extra, "mm_token_type_ids": torch.cat([types, pad], dim=1)}


def sequence_logprobs(llm, input_ids, attention_mask, prompt_len, extra):
    """Per-token log probability of the generated continuation."""
    extra = extend_token_types(extra, input_ids.shape[1])
    out = llm(input_ids=input_ids, attention_mask=attention_mask, **extra)
    logits = out.logits[:, :-1]
    targets = input_ids[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    picked = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # Only the continuation is the model's own choice; the prompt is given.
    mask = torch.zeros_like(picked, dtype=torch.bool)
    mask[:, prompt_len - 1:] = True
    mask &= attention_mask[:, 1:].bool()
    return picked, mask


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--init", required=True, help="SFT checkpoint to start from")
    ap.add_argument("--model", default="8B")
    ap.add_argument("--task", default="radar_probe",
                    help="comma-separated. Training several verifiable tasks at "
                         "once is where GRPO earns its keep: measured, "
                         "continuing SFT on radar_probe alone collapsed the "
                         "untrained radar_transfer from 0.838 to 0.166 while "
                         "GRPO held it at 0.851, so the tasks can share a "
                         "policy without trampling each other")
    ap.add_argument("--reward", default="relative", choices=sorted(REWARDS))
    ap.add_argument("--group", type=int, default=8, help="samples per prompt")
    ap.add_argument("--prompts", type=int, default=2, help="prompts per step")
    ap.add_argument("--micro", type=int, default=2,
                    help="sequences per backward pass")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    from training.connector import RadarConnector, add_radar_tokens, llm_hidden_size
    from training.instruct_data import InstructDataset, build_collate
    from training.radar_encoder import (RadarEncoder, encoder_kwargs,
                                        load_encoder_state)
    from training.train_vlm import MODEL_DIR, RadarInjector

    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    model_dir = MODEL_DIR[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    weights = os.path.join(args.init, "model")
    source = weights if os.path.isdir(weights) else model_dir
    llm = AutoModelForImageTextToText.from_pretrained(
        source, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    pad_id = add_radar_tokens(tokenizer, llm)
    processor.tokenizer = tokenizer
    llm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    llm.enable_input_require_grads()
    llm.config.use_cache = False

    state = torch.load(os.path.join(args.init, "adapters.pt"), map_location="cpu")
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
    # The radar path is frozen: this stage is about the output layer learning to
    # read a representation that probing already showed is present and correct.
    for p in list(encoder.parameters()) + list(connector.parameters()):
        p.requires_grad = False
    log(f"policy from {source}, reward '{args.reward}', group {args.group}")

    tasks = tuple(x.strip() for x in args.task.split(",") if x.strip())
    dataset = InstructDataset(
        tasks=tasks, split="train", processor=processor,
        tokenizer=tokenizer, n_frames=trained["frames"],
        radar_tokens=encoder.n_tokens,
        samples=args.steps * args.prompts * 4, all_profiles=True,
        radar_dropout=0.0, seed=args.seed)
    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        num_workers=args.workers,
                        collate_fn=build_collate(processor, tokenizer,
                                                 trained["max_length"]))
    counts = dataset.task_counts()
    log(f"{len(dataset):,} prompts: " +
        ", ".join(f"{k} {v:,}" for k, v in sorted(counts.items())))

    injector = RadarInjector(llm.get_input_embeddings(), pad_id)
    header = tokenizer("<|im_start|>assistant\n",
                       add_special_tokens=False)["input_ids"]
    reward_fn = REWARDS[args.reward]
    trainable = [p for p in llm.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.0)
    os.makedirs(args.out, exist_ok=True)
    history, step, pending = [], 0, []
    started = time.monotonic()

    def flush(batch_groups):
        """One optimiser step over the collected groups."""
        nonlocal step
        rewards = np.array([g["reward"] for g in batch_groups], dtype=np.float64)
        # Group-relative advantage: the mean of the group is the baseline, which
        # is what removes the need for a value network.
        advantages = []
        for g in batch_groups:
            group = np.array(g["group_rewards"], dtype=np.float64)
            spread = group.std()
            advantages.append(0.0 if spread < 1e-6
                              else (g["reward"] - group.mean()) / spread)

        optimiser.zero_grad(set_to_none=True)
        total = 0.0
        for start in range(0, len(batch_groups), args.micro):
            chunk = batch_groups[start:start + args.micro]
            chunk_adv = advantages[start:start + args.micro]
            loss = 0.0
            for g, adv in zip(chunk, chunk_adv):
                if adv == 0.0:
                    continue
                injector.pending = g["radar"]
                picked, mask = sequence_logprobs(
                    llm, g["ids"], g["mask"], g["prompt_len"], g["extra"])
                new = (picked * mask).sum() / mask.sum().clamp(min=1)
                ratio = torch.exp(new - g["old_logp"])
                clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
                loss = loss - torch.min(ratio * adv, clipped * adv)
            if isinstance(loss, torch.Tensor):
                (loss / len(batch_groups)).backward()
                total += float(loss)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimiser.step()
        step += 1
        if step % 20 == 0 or step == 1:
            rate = step / (time.monotonic() - started) * 60
            spread = float(np.mean([np.std(g["group_rewards"])
                                    for g in batch_groups]))
            # The reported loss is ~0 by construction -- advantages are zero-mean
            # within a group, so the terms cancel. The gradient is not zero, and
            # `spread` is what decides whether there is anything to learn from:
            # a group whose eight samples all score the same teaches nothing.
            log(f"step {step}/{args.steps}  reward {rewards.mean():.4f}  "
                f"best {np.mean([max(g['group_rewards']) for g in batch_groups]):.4f}  "
                f"spread {spread:.3f}  grad {grad_norm:.4f}  {rate:.1f} step/min")
            history.append({"step": step, "reward": float(rewards.mean()),
                            "spread": spread, "grad_norm": grad_norm})
        return []

    for batch in loader:
        if step >= args.steps:
            break
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
        for begin in range(len(ids) - len(header), -1, -1):
            if ids[begin:begin + len(header)] == header:
                cut = begin + len(header)
                break
        if cut is None:
            continue
        reference = tokenizer.decode(labels[0][labels[0] != -100],
                                     skip_special_tokens=True)
        prompt = {k: (v[:, :cut] if k in ("input_ids", "attention_mask",
                                          "mm_token_type_ids") else v)
                  for k, v in tensors.items()}
        extra = {k: v for k, v in prompt.items()
                 if k not in ("input_ids", "attention_mask")}
        # Generation gets the prompt-length copy; scoring extends it per call.

        with torch.no_grad():
            radar = connector(encoder(points, radar_mask, sensor)["tokens"])
            llm.config.use_cache = True
            injector.pending = radar
            sampled = llm.generate(
                **prompt, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, top_p=0.95,
                num_return_sequences=args.group,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            llm.config.use_cache = False

        texts = [tokenizer.decode(s[cut:], skip_special_tokens=True)
                 for s in sampled]
        group_rewards = [reward_fn(t, reference) for t in texts]
        if step < 1 and not history:
            log(f"reference: {reference[:70]!r}")
            for text, r in zip(texts[:4], group_rewards[:4]):
                log(f"  sample r={r:.3f}: {text.strip()[:70]!r}")

        for i, sequence in enumerate(sampled):
            row = sequence.unsqueeze(0)
            attention = torch.ones_like(row)
            with torch.no_grad():
                injector.pending = radar
                picked, mask = sequence_logprobs(llm, row, attention, cut, extra)
                old_logp = (picked * mask).sum() / mask.sum().clamp(min=1)
            pending.append({"ids": row, "mask": attention, "prompt_len": cut,
                            "extra": extra, "radar": radar,
                            "old_logp": old_logp, "reward": group_rewards[i],
                            "group_rewards": group_rewards})

        if len(pending) >= args.prompts * args.group:
            pending = flush(pending)
            if args.save_every and step % args.save_every == 0:
                torch.save({"args": vars(args), "history": history},
                           os.path.join(args.out, "history.pt"))

    injector.remove()
    llm.save_pretrained(os.path.join(args.out, "model"), safe_serialization=True)
    torch.save({"connector": connector.state_dict(),
                "encoder": encoder.state_dict(),
                "args": {**trained, **vars(args)}, "history": history},
               os.path.join(args.out, "adapters.pt"))
    with open(os.path.join(args.out, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    log(f"saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
