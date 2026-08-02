#!/usr/bin/env python3
"""Remove QA items whose numeric errors could not be safely corrected.

`correct_qa_numbers` rewrote the 674 disagreeing claims that were pure context.
The remaining 1,047 cannot be patched without breaking the item:

  arithmetic        845  the number feeds a shown calculation
  answer_dependent  135  the number also appears in the options, so the answer key
                         depends on it
  comparative        67  the number is an operand of a comparison that could flip

This drops every item carrying at least one of those, then renumbers the `id`
fields. Renumbering matters: `fix_qa_defects` enforces `..._Q<n>` matching the
item's position, so leaving gaps would make a later run of it silently rewrite
every id. The pre-deletion id is kept as `original_id`.

    python -m datatools.drop_bad_qa_items --dry-run
    python -m datatools.drop_bad_qa_items
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time
from collections import Counter

from . import paths

UNSAFE_VERDICTS = {"arithmetic", "answer_dependent", "comparative", "range"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def unsafe_verdicts(item):
    return {c.get("verdict") for c in item.get("corrections", [])} & UNSAFE_VERDICTS


def process(path, dry_run):
    with open(path) as fh:
        doc = json.load(fh)

    stats = Counter()
    kept = []
    for item in doc.get("qa", []):
        bad = unsafe_verdicts(item)
        if bad:
            stats["items_dropped"] += 1
            for verdict in bad:
                stats[f"dropped_by_{verdict}"] += 1
            continue
        kept.append(item)
    stats["items_kept"] += len(kept)

    if not stats["items_dropped"]:
        return stats, False

    chunk, clip_id = int(doc["chunk"]), doc["clip_id"]
    for index, item in enumerate(kept, start=1):
        if "original_id" not in item:
            item["original_id"] = item.get("id")
        item["id"] = f"chunk_{chunk:04d}_{clip_id}.mp4_Q{index}"

    doc["qa"] = kept
    doc["n_qa"] = len(kept)
    doc["n_qa_dropped"] = stats["items_dropped"]
    if "verification_summary" in doc:
        # Stale the moment items move; verify_qa + flag_qa_verification rebuild it.
        doc.pop("verification_summary")

    if not dry_run:
        if kept:
            with open(path, "w") as fh:
                json.dump(doc, fh, ensure_ascii=False, indent=2)
        else:
            os.remove(path)
            stats["documents_removed"] += 1
    elif not kept:
        stats["documents_removed"] += 1
    return stats, True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--qa-dir",
                    default=os.path.join(paths.SPLIT_ROOT, "10_radar_vision_qa/qa"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.qa_dir, "*.json")))
    log(f"{len(files)} QA documents")

    backup = os.path.join(os.path.dirname(args.qa_dir), "qa_before_drop")
    if not args.dry_run and not args.no_backup and not os.path.exists(backup):
        shutil.copytree(args.qa_dir, backup)
        log(f"pre-drop copy at {backup}")

    totals = Counter()
    touched = 0
    for path in files:
        stats, changed = process(path, args.dry_run)
        totals.update(stats)
        touched += int(changed)

    before = totals["items_kept"] + totals["items_dropped"]
    log("")
    log(f"items before : {before:,}")
    log(f"items dropped: {totals['items_dropped']:,} "
        f"({totals['items_dropped']/before*100:.1f}%)")
    log(f"items kept   : {totals['items_kept']:,}")
    log("dropped because they contained")
    for verdict in sorted(UNSAFE_VERDICTS):
        key = f"dropped_by_{verdict}"
        if totals[key]:
            log(f"  {verdict:17s} {totals[key]:5,}")
    log(f"documents edited : {touched:,}")
    if totals["documents_removed"]:
        log(f"documents removed (no items left): {totals['documents_removed']:,}")
    log("")
    log("verification_summary was cleared; re-run verify_qa then "
        "flag_qa_verification to rebuild it against the new indices.")
    if args.dry_run:
        log("dry run - nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
