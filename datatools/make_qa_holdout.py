#!/usr/bin/env python3
"""Carve a test set out of task 10, which shipped without one.

All 1,999 QA clips are in the `train` split, so the question-answering task has
been trained on and never evaluated -- every `eval.json` written so far simply
has no `qa` row. This picks a fixed subset to hold out.

The clips have to leave *every* task's training set, not just QA's. A held-out
clip still produces detection, tracking, planning and description items, and if
those stay in training the model will have seen the clip's video and radar
before it is asked a question about it. The holdout is therefore a property of
the clip, matching how train/val/test are drawn everywhere else in this release.

Selection is by a hash of the clip id rather than by shuffling, so the same
clips come out on any machine and adding clips later cannot silently reshuffle
the ones already chosen.

    python -m datatools.make_qa_holdout --count 99
"""

import argparse
import glob
import hashlib
import json
import os
import sys

from . import paths

OUT_NAME = "qa_holdout_clips.json"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--count", type=int, default=99)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    docs = sorted(glob.glob(os.path.join(paths.SPLIT_ROOT,
                                         "10_radar_vision_qa/qa/*.json")))
    clips = []
    for path in docs:
        doc = json.load(open(path))
        clips.append((doc["clip_id"], doc.get("split"), len(doc.get("qa", []))))
    splits = {}
    for _, s, _ in clips:
        splits[s] = splits.get(s, 0) + 1
    print(f"{len(clips):,} QA clips, split distribution {splits}")

    ranked = sorted(clips, key=lambda c: hashlib.sha1(c[0].encode()).hexdigest())
    chosen = ranked[: args.count]
    questions = sum(n for _, _, n in chosen)
    print(f"holding out {len(chosen)} clips, {questions:,} questions "
          f"({questions / max(sum(n for _, _, n in clips), 1) * 100:.1f}% of the task)")

    out = args.out or os.path.join(paths.COMMON_DIR, OUT_NAME)
    with open(out, "w") as fh:
        json.dump({"note": "QA test clips; excluded from training for every task",
                   "count": len(chosen), "questions": questions,
                   "clip_ids": sorted(c for c, _, _ in chosen)}, fh, indent=1)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
