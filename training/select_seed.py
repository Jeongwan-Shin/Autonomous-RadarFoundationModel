#!/usr/bin/env python3
"""Pick the best seed on validation, then report that one checkpoint on test.

Seeds do land in different places -- two GRPO runs on identical settings went
0.378 -> 0.725 -> 0.391 and 0.330 -> 0.366 -> 0.549 -- and for a model you
intend to ship, taking the best of them is the right call, not a statistical
sin. What is a sin is taking the best of them *on the test set* and quoting that
number as the model's performance. Selecting the maximum of N draws shifts the
estimate up by roughly the spread between seeds, which here is 0.028: exactly
the size of the effects being argued about.

So the two steps are separated. Selection reads `val` and is allowed to be
greedy. Reporting reads `test` for the single winner and nothing else, so the
figure quoted is an honest out-of-sample number for the checkpoint chosen.

    python -m training.select_seed --checkpoints ckpt_s0 ckpt_s1 ckpt_s2 \\
        --tasks radar_objects --metric contribution
"""

import argparse
import json
import os
import subprocess
import sys


def log(msg):
    print(msg, flush=True)


def evaluate(checkpoint, tasks, split, items, out, gpu, workers=2):
    """One eval_all_tasks run. Returns its parsed output."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [sys.executable, "-u", "-m", "training.eval_all_tasks",
           "--checkpoint", checkpoint, "--tasks", tasks, "--split", split,
           "--items", str(items), "--workers", str(workers), "--out", out]
    subprocess.run(cmd, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return {e["task"]: e for e in json.load(open(out))}


def contribution(entry):
    """Radar-attributable correlation. `full` alone is not a score to select on:
    the camera answers much of every one of these questions by itself, so a
    checkpoint can top the table by getting better at guessing."""
    a, b = entry.get("full", {}), entry.get("shuffled", {})
    if "corr" not in a or "corr" not in b:
        return None
    return a["corr"] - b["corr"]


def score_of(tasks_result, tasks, metric):
    values = []
    for task in tasks:
        e = tasks_result.get(task)
        if not e:
            continue
        v = contribution(e) if metric == "contribution" else e["full"].get("corr")
        if v is not None:
            values.append(v)
    return sum(values) / len(values) if values else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--tasks", default="radar_objects")
    ap.add_argument("--metric", default="contribution",
                    choices=("contribution", "full"))
    ap.add_argument("--val-items", type=int, default=300)
    ap.add_argument("--test-items", type=int, default=500)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--outdir", default="runs/12_selection")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    log(f"선택 단계 — val {args.val_items}문항, 지표 {args.metric}")
    scores = {}
    for ckpt in args.checkpoints:
        name = os.path.basename(ckpt.rstrip("/"))
        out = os.path.join(args.outdir, f"{name}_val.json")
        result = evaluate(ckpt, args.tasks, "val", args.val_items, out, args.gpu)
        scores[ckpt] = score_of(result, tasks, args.metric)
        log(f"  {name:28s} val {scores[ckpt]:+.3f}")

    ranked = sorted((v, k) for k, v in scores.items() if v is not None)
    if not ranked:
        log("val에서 점수를 얻지 못했습니다")
        return 1
    best = ranked[-1][1]
    log(f"\n선택: {os.path.basename(best.rstrip('/'))} "
        f"(val {scores[best]:+.3f}, {len(ranked)}개 중)")

    name = os.path.basename(best.rstrip("/"))
    out = os.path.join(args.outdir, f"{name}_test.json")
    result = evaluate(best, args.tasks, "test", args.test_items, out, args.gpu)
    log(f"\n보고 단계 — test {args.test_items}문항, 선택된 체크포인트 하나만")
    for task in tasks:
        e = result.get(task)
        if not e:
            continue
        c = contribution(e)
        log(f"  {task:18s} full {e['full'].get('corr', float('nan')):+.3f}  "
            f"shuffled {e['shuffled'].get('corr', float('nan')):+.3f}  "
            f"기여 {c:+.3f}" if c is not None else f"  {task}")
    log(f"\n선택은 val에서, 보고는 test에서. 최댓값을 test에서 고르면 그 수치는 "
        f"시드 편차만큼 부풀려집니다.")
    summary = os.path.join(args.outdir, "selection.json")
    json.dump({"val_scores": {os.path.basename(k.rstrip('/')): v
                              for k, v in scores.items()},
               "selected": name, "metric": args.metric,
               "test": {t: contribution(result[t]) for t in tasks
                        if t in result}},
              open(summary, "w"), indent=1, default=float)
    log(f"wrote {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
