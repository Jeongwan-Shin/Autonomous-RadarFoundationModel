#!/usr/bin/env python3
"""Re-score saved generations. No GPU, no model, no re-running inference.

Scoring in this project has been wrong four times: three question forms pooled
into one correlation that only detected which question was asked; a shuffled
control that at batch size 1 rolled a row onto itself and so was not shuffled;
a 2 m match threshold applied to a task with no azimuth; a ridge penalty left
fixed when tuning it moved an R^2 from -0.5 to +0.67. Each fix meant generating
everything again, hours of GPU time, to answer a question about arithmetic.

The generations are what the model actually produced. The metric is an opinion
about them, and opinions get revised. `eval_all_tasks.py` now writes every one
to `<out>.generations.jsonl`, and this re-derives the numbers from that file, so
the next scoring fix costs seconds.

    python -m training.rescore_generations runs/10_big_eval/multi_s0.generations.jsonl
    python -m training.rescore_generations runs/*/*.generations.jsonl --by-form
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

from training.task_scorers import scorer_for, summarise


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def correlate(pairs):
    if len(pairs) < 5:
        return None
    a = np.array([p for p, _ in pairs], float)
    b = np.array([t for _, t in pairs], float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def score(rows, by_form=False):
    """Re-run the current scorers over saved text, grouped as the report needs."""
    out = {}
    for task in sorted({r["task"] for r in rows}):
        scorer = scorer_for(task)
        entry = {}
        for mode in ("full", "shuffled"):
            subset = [r for r in rows if r["task"] == task and r["mode"] == mode]
            if not subset:
                continue
            records, pairs = [], {}
            for r in subset:
                result = scorer(r["generated"], r["reference"])
                records.append(result)
                if result.get("pred") is not None:
                    pairs.setdefault(r.get("form", "main"), []).append(
                        (result["pred"], result["truth"]))
            forms = {f: correlate(v) for f, v in pairs.items()}
            valid = [v for v in forms.values() if v is not None]
            summary = summarise(task, records)
            # The mean over forms, not a correlation over the pooled pairs:
            # pooling questions whose answers differ in magnitude by 100x makes
            # the coefficient detect the question rather than the answer.
            if valid:
                summary["corr"] = float(np.mean(valid))
                summary["by_form"] = forms
            entry[mode] = summary
        if entry:
            out[task] = entry
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", help="*.generations.jsonl (globs fine)")
    ap.add_argument("--by-form", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    files = sorted({p for pattern in args.paths for p in glob.glob(pattern)})
    if not files:
        print("no generation files matched", file=sys.stderr)
        return 1

    everything = {}
    for path in files:
        rows = load(path)
        name = os.path.basename(path).replace(".generations.jsonl", "")
        scored = score(rows, args.by_form)
        everything[name] = scored
        print(f"\n=== {name}  ({len(rows):,} generations) ===")
        for task, entry in scored.items():
            full = entry.get("full", {})
            shuf = entry.get("shuffled", {})
            if "corr" in full and "corr" in shuf:
                print(f"  {task:18s} full {full['corr']:+.3f}  "
                      f"shuffled {shuf['corr']:+.3f}  "
                      f"기여 {full['corr'] - shuf['corr']:+.3f}")
                if args.by_form:
                    for form in sorted(full.get("by_form", {})):
                        a = full["by_form"][form]
                        b = shuf.get("by_form", {}).get(form)
                        if a is None or b is None:
                            continue
                        print(f"      {form:14s} {a:+.3f} / {b:+.3f} = {a - b:+.3f}")
            else:
                keys = [k for k in ("f1", "displacement_mae_m", "range_mae_m",
                                    "accuracy") if k in full]
                if keys:
                    print(f"  {task:18s} " +
                          "  ".join(f"{k} {full[k]:.3f}" for k in keys))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(everything, fh, indent=1, default=float)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
