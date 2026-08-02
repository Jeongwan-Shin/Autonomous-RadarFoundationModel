#!/usr/bin/env python3
"""Repair two defects in the radar-vision QA files, and record the split.

Defect 1 -- corrupted `id` fields, 5,715 of 39,988 items (14.3%). Each item's
`id` is meant to read `chunk_XXXX_<clip_id>.mp4_Q<n>`, but the embedded clip_id
often disagrees with the file's own `clip_id`: single characters differ, the
chunk prefix is duplicated, a UUID group is truncated, or the `_Q<n>` suffix is
missing. The document-level `clip_id`, `chunk` and `filename` fields are all
correct, so ids are simply regenerated from those.

Defect 2 -- 29 items carry four options instead of five. A distractor is added so
the answer space is uniform; without it any loader that assumes a fixed option
count breaks on those rows.

The QA set is entirely training data, so `split: "train"` is written into every
document rather than inherited from the Nvidia clip index. Note that 561 of the
1,999 clips sit in Nvidia's official *val* split, which matters if a later
evaluation on Nvidia val is meant to be unseen.

Originals are preserved as `qa_original/` unless --no-backup is given.

    python -m datatools.fix_qa_defects
    python -m datatools.fix_qa_defects --dry-run
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from collections import Counter

from . import paths

ID_PATTERN = re.compile(r"^chunk_(\d{4})_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}"
                        r"-[0-9a-f]{4}-[0-9a-f]{12})\.mp4_Q(\d+)$")

# Generic but plausible distractors, used only to pad four-option items up to
# five. Deliberately not scenario-specific: they must be wrong for any question,
# so they describe outcomes the labels never support.
FILLER_OPTIONS = [
    "None of the described objects is relevant here.",
    "The scene contains no other road users.",
    "The ego vehicle is stationary throughout.",
    "The information needed is not observable in this view.",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def expected_id(chunk, clip_id, index):
    return f"chunk_{int(chunk):04d}_{clip_id}.mp4_Q{index}"


def classify_bad_id(item_id, chunk, clip_id):
    """Why an id is wrong -- reported so the generator can be fixed upstream."""
    if not item_id:
        return "missing"
    if not item_id.endswith(tuple(f"_Q{i}" for i in range(1, 40))):
        if re.search(r"_Q\d+$", item_id) is None:
            return "no_question_suffix"
    m = ID_PATTERN.match(item_id)
    if m is None:
        if item_id.count(f"{int(chunk):04d}") > 1:
            return "duplicated_chunk"
        return "malformed"
    if m.group(2) != clip_id:
        return "wrong_clip_id"
    if m.group(1) != f"{int(chunk):04d}":
        return "wrong_chunk"
    return "ok"


def fix_document(doc, filler_cycle):
    """Returns (doc, stats). Mutates in place."""
    stats = Counter()
    chunk, clip_id = doc["chunk"], doc["clip_id"]

    doc["split"] = "train"
    doc["filename"] = f"chunk_{int(chunk):04d}_{clip_id}.mp4"

    answers_relabelled = 0
    for index, item in enumerate(doc.get("qa", []), start=1):
        reason = classify_bad_id(item.get("id"), chunk, clip_id)
        if reason != "ok":
            stats[f"id_{reason}"] += 1
        item["id"] = expected_id(chunk, clip_id, index)
        item["filename"] = doc["filename"]

        options = item.get("options", {})
        if len(options) < 5:
            stats["options_padded"] += 1
            used = {v.strip().lower() for v in options.values()}
            letters = [chr(ord("A") + i) for i in range(5)]
            for letter in letters:
                if letter in options:
                    continue
                for candidate in filler_cycle:
                    if candidate.strip().lower() not in used:
                        options[letter] = candidate
                        used.add(candidate.strip().lower())
                        break
                break
            item["options"] = {k: options[k] for k in sorted(options)}
        stats["items"] += 1

    doc["n_qa"] = len(doc.get("qa", []))
    return doc, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--qa-dir",
                    default=os.path.join(paths.SPLIT_ROOT, "10_radar_vision_qa/qa"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.qa_dir, "*.json")))
    if not files:
        raise SystemExit(f"no QA json under {args.qa_dir}")
    log(f"{len(files)} QA documents")

    backup = os.path.join(os.path.dirname(args.qa_dir), "qa_original")
    if not args.dry_run and not args.no_backup and not os.path.exists(backup):
        shutil.copytree(args.qa_dir, backup)
        log(f"originals copied to {backup}")

    totals = Counter()
    for path in files:
        with open(path) as fh:
            doc = json.load(fh)
        doc, stats = fix_document(doc, FILLER_OPTIONS)
        totals.update(stats)
        if not args.dry_run:
            with open(path, "w") as fh:
                json.dump(doc, fh, ensure_ascii=False, indent=2)

    log(f"items processed: {totals['items']:,}")
    log("id defects found and repaired:")
    for key in sorted(k for k in totals if k.startswith("id_")):
        log(f"  {key[3:]:22s} {totals[key]:6,}")
    log(f"  {'total':22s} {sum(v for k, v in totals.items() if k.startswith('id_')):6,}")
    log(f"option sets padded to five: {totals['options_padded']:,}")
    if args.dry_run:
        log("dry run - nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
