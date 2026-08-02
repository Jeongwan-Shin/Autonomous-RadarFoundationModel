#!/usr/bin/env python3
"""Work out what fits on 5 x B200 for each LLM size, and pick a parallelism plan.

Measured facts this is built on, not vendor numbers:

  per GPU HBM        183,359 MiB reported, 178.4 GiB usable, ~179 GiB
  aggregate          895 GiB across 5 devices
  dense bf16 matmul  ~1,400 TFLOP/s per GPU, 7.05 PFLOP/s aggregate
  interconnect       NV18 full mesh -- every pair is NVLink, so sharded
                     parallelism pays little communication penalty

Five devices is an awkward count. Tensor parallelism wants the head count and
hidden dim to divide by the degree, and 32 or 64 heads do not divide by 5, so TP
is off the table here. FSDP (ZeRO-3) shards by parameter count rather than by any
structural dimension, so it takes 5 devices without complaint. That is the whole
reason the plans below use FSDP and never TP.

    python training/memory_plan.py
    python training/memory_plan.py --ctx 4639 --batch 4
"""

import argparse

GIB = 1024 ** 3
PER_GPU_GIB = 179.0
N_GPU = 5
AGGREGATE_GIB = PER_GPU_GIB * N_GPU

# Read from each checkpoint's config.json, not assumed. There is no 70B dense
# Qwen3-VL; the tiers that exist are 8B and 32B dense, then two MoE models.
#
# For MoE, memory follows the total parameter count -- every expert has to be
# resident -- while compute follows the active count. That is what makes the
# 235B model interesting here: it needs the memory of a 235B model but costs
# roughly what a 22B dense model costs per token.
#
# (name, total_params, active_params, layers, hidden, weights_gb)
MODELS = [
    ("Qwen3-VL-8B", 8.0e9, 8.0e9, 36, 4096, 17.5),
    ("Qwen3-VL-32B", 32.0e9, 32.0e9, 64, 5120, 66.7),
    ("Qwen3-VL-30B-A3B", 30.0e9, 3.0e9, 48, 2048, 62.1),
    ("Qwen3-VL-235B-A22B", 235.0e9, 22.0e9, 94, 4096, 471.3),
]

# Bytes per parameter for each training mode.
MODES = {
    # bf16 weights + bf16 grads + fp32 Adam moment pair + fp32 master copy
    "full_ft": 2 + 2 + 8 + 4,
    # base frozen in bf16; only adapters carry grads and optimiser state, and at
    # ~0.5% of parameters that cost rounds to nothing next to the weights
    "lora": 2 + 0.06,
}


def activation_gib(layers, hidden, ctx, batch, checkpointing):
    """Rough activation footprint in GiB.

    With gradient checkpointing only layer boundaries are kept, so the term is
    linear in layers and independent of the per-layer internals. Without it the
    attention and MLP intermediates dominate; the 34x factor is the usual
    transformer estimate covering several hidden-sized tensors plus the 4x MLP.
    """
    tokens = batch * ctx
    if checkpointing:
        return tokens * hidden * 2 * (layers + 1) / GIB
    return tokens * hidden * 2 * 34 * layers / GIB


def plan(params, layers, hidden, ctx, batch):
    rows = []
    for mode, bytes_per_param in MODES.items():
        state = params * bytes_per_param / GIB
        for shard in ("none", "zero2", "fsdp"):
            # none  : every device holds a full replica of everything
            # zero2 : gradients and optimiser state sharded, weights replicated.
            #         Communicates far less than ZeRO-3 because weights never
            #         need gathering, and for an 8B model it cuts the 119 GiB
            #         replicated footprint to 38 GiB, which is what actually
            #         unlocks a usable micro-batch.
            # fsdp  : ZeRO-3, weights sharded too
            if shard == "none":
                per_gpu_state = state
            elif shard == "zero2":
                weights = params * 2 / GIB
                per_gpu_state = weights + (state - weights) / N_GPU
            else:
                per_gpu_state = state / N_GPU
            for ckpt in (True, False):
                act = activation_gib(layers, hidden, ctx, batch, ckpt)
                total = per_gpu_state + act
                rows.append({
                    "mode": mode, "shard": shard, "ckpt": ckpt,
                    "max_micro_batch": max_micro_batch(
                        per_gpu_state, layers, hidden, ctx, ckpt),
                    "state_gib": per_gpu_state, "act_gib": act,
                    "total_gib": total,
                    # leave 8% for fragmentation, NCCL buffers and the CUDA context
                    "fits": total < PER_GPU_GIB * 0.92,
                })
    return rows


def max_micro_batch(per_gpu_state, layers, hidden, ctx, ckpt, cap=64):
    """Largest micro-batch that still leaves the 8% reserve."""
    budget = PER_GPU_GIB * 0.92 - per_gpu_state
    if budget <= 0:
        return 0
    per_sample = activation_gib(layers, hidden, ctx, 1, ckpt)
    return min(cap, int(budget / per_sample)) if per_sample > 0 else cap


def rank(row):
    """Preference order: full fine-tuning, then no checkpointing, then no sharding.

    Checkpointing is ranked ahead of sharding deliberately. Recomputing the
    forward pass costs roughly 30% of throughput, while FSDP on an NV18 full mesh
    costs little -- every pair of devices has a direct NVLink path. Ranking
    sharding first would have recommended LoRA without sharding for 70B at
    136 GiB, over the strictly better sharded, uncheckpointed 97 GiB plan.
    """
    return (0 if row["mode"] == "full_ft" else 1,
            1 if row["ckpt"] else 0,
            0 if row["shard"] == "none" else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ctx", type=int, default=1699,
                    help="sequence length; 1699 is the recommended budget "
                         "(vision 980 + radar 256 + ego 63 + text 400)")
    ap.add_argument("--batch", type=int, default=1,
                    help="micro-batch per GPU")
    args = ap.parse_args(argv)

    print(f"5 x B200, {PER_GPU_GIB:.0f} GiB each, {AGGREGATE_GIB:.0f} GiB total")
    print(f"context {args.ctx}, micro-batch {args.batch} per GPU")
    print("tensor parallelism is unavailable at 5 devices (head counts do not "
          "divide), so sharding means FSDP\n")

    for name, params, active, layers, hidden, weights_gb in MODELS:
        tag = (f"{params/1e9:.0f}B total / {active/1e9:.0f}B active"
               if active != params else f"{params/1e9:.0f}B")
        print(f"=== {name} ({tag}, {layers} layers, hidden {hidden}, "
              f"{weights_gb:.0f} GB on disk) ===")
        best = None
        for row in plan(params, layers, hidden, args.ctx, args.batch):
            mark = "OK " if row["fits"] else "   "
            print(f"  {mark}{row['mode']:8s} shard={row['shard']:5s} "
                  f"ckpt={str(row['ckpt']):5s}  state {row['state_gib']:7.1f} "
                  f"+ act {row['act_gib']:6.1f} = {row['total_gib']:7.1f} GiB"
                  f"   max mb {row['max_micro_batch']:3d}")
            if row["fits"] and (best is None or rank(row) < rank(best)):
                best = row
        if best is None:
            print("  -> nothing fits; add gradient accumulation or CPU offload\n")
        else:
            gb = best["max_micro_batch"] * N_GPU
            print(f"  -> {best['mode']}, shard={best['shard']}, "
                  f"ckpt={best['ckpt']}, micro-batch {best['max_micro_batch']} "
                  f"per GPU -> {gb} per step before accumulation\n")


if __name__ == "__main__":
    main()
