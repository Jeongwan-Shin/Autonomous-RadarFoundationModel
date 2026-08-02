#!/usr/bin/env python3
"""One table over every evaluated checkpoint, ranked by radar grounding.

Held-out loss is not the figure of merit here and ranking by it is actively
misleading: the run with the best loss so far is the one that ignores its radar
most completely. What separates the runs is `shuffled` -- the same model given
another clip's radar. If that costs nothing, the radar is decorative.

Selectivity is the ratio of the shuffled penalty on tasks that need radar to the
penalty on tasks that do not. It is the number to read. A model that is merely
disturbed by any change to its input scores about 1.0 on it; a model that reads
the radar for the questions that require it scores well above 1.

    python -m training.compare_runs
    python -m training.compare_runs --checkpoints /path/to/checkpoints
"""

import argparse
import glob
import json
import os
import sys

# The tasks whose answers change when the radar does. `motion_seg` and
# `depth_range` are only partly radar-bound -- most of their target is an object
# list the camera explains -- but both cite radar evidence explicitly.
RADAR_TASKS = frozenset({"radar_probe", "radar_transfer", "motion_seg",
                         "depth_range", "desc_radar", "desc_complementarity"})
HELD_OUT = "ood_reasoning"


def summarise(path):
    with open(path) as fh:
        data = json.load(fh)
    full, shuffled = data.get("full", {}), data.get("shuffled", {})
    tasks = [t for t in full if t in shuffled]
    if not tasks:
        return None
    gap = {t: shuffled[t]["loss"] - full[t]["loss"] for t in tasks}
    radar = [gap[t] for t in tasks if t in RADAR_TASKS]
    camera = [gap[t] for t in tasks if t not in RADAR_TASKS and t != HELD_OUT]
    mean_radar = sum(radar) / len(radar) if radar else float("nan")
    mean_camera = sum(camera) / len(camera) if camera else float("nan")

    # Accuracy on the digits of a radar answer, and how much of it the model
    # loses when handed another clip's radar. The loss gap can be inflated by
    # any training term that makes the output distribution twitchy; getting the
    # detection count right cannot. When the two disagree, this is the one that
    # says whether the radar was read.
    digits, digit_gap = [], []
    for task in tasks:
        if task not in RADAR_TASKS:
            continue
        a, b = full[task].get("digit_acc"), shuffled[task].get("digit_acc")
        if a is not None:
            digits.append(a)
        if a is not None and b is not None:
            digit_gap.append(a - b)
    return {
        "tasks": len(tasks),
        "mean_full": sum(full[t]["loss"] for t in tasks) / len(tasks),
        "mean_gap": sum(gap.values()) / len(gap),
        "radar_gap": mean_radar,
        "camera_gap": mean_camera,
        # Guarded: a camera gap at or below zero makes the ratio meaningless
        # rather than infinite, and reporting a huge number there would invert
        # the ranking exactly when the model has stopped reading the radar.
        "selectivity": (mean_radar / mean_camera
                        if mean_camera > 1e-4 else float("nan")),
        "digit_acc": sum(digits) / len(digits) if digits else float("nan"),
        "digit_gap": (sum(digit_gap) / len(digit_gap) if digit_gap
                      else float("nan")),
        "held_out": full.get(HELD_OUT, {}).get("loss"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoints", default="/NHNHOME/workspace/checkpoints")
    ap.add_argument("--sort", default="digit_gap",
                    choices=("digit_gap", "selectivity", "radar_gap",
                             "mean_full", "name"))
    args = ap.parse_args(argv)

    rows = []
    for path in sorted(glob.glob(os.path.join(args.checkpoints, "*", "eval.json"))):
        stats = summarise(path)
        if stats:
            rows.append((os.path.basename(os.path.dirname(path)), stats))
    if not rows:
        print("no eval.json found")
        return 1

    if args.sort == "name":
        rows.sort(key=lambda r: r[0])
    else:
        rows.sort(key=lambda r: (r[1][args.sort] if r[1][args.sort] ==
                                 r[1][args.sort] else -1e9), reverse=True)

    print(f"{'run':28s}{'mean full':>10s}{'radar gap':>11s}{'camera gap':>11s}"
          f"{'digit acc':>11s}{'digit gap':>11s}{'held-out':>10s}")
    print("-" * 92)
    for name, s in rows:
        held = "     -" if s["held_out"] is None else f"{s['held_out']:10.4f}"
        digit = ("       n/a" if s["digit_acc"] != s["digit_acc"]
                 else f"{s['digit_acc']*100:10.1f}%")
        dgap = ("       n/a" if s["digit_gap"] != s["digit_gap"]
                else f"{s['digit_gap']*100:+10.2f}p")
        print(f"{name:28s}{s['mean_full']:10.4f}{s['radar_gap']:+11.4f}"
              f"{s['camera_gap']:+11.4f}{digit}{dgap}{held}")
    print()
    print("  radar gap  : shuffled minus full nll on the radar-dependent tasks.")
    print("               Sensitive, but any term that makes the output twitchy")
    print("               inflates it -- the contrast hinge trains it directly")
    print("  digit gap  : the same swap measured in accuracy on the digits of the")
    print("               answer. Nothing optimises this, so it is the honest one")
    print("  held-out   : ood_reasoning loss, never trained on")
    return 0


if __name__ == "__main__":
    sys.exit(main())
