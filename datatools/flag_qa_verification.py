#!/usr/bin/env python3
"""Attach the rationale verification verdict to each QA item.

`verify_qa.py` recomputes the numeric claims in every rationale and writes the
per-item result to `verification.parquet`. This folds that verdict back into the
QA documents so a loader does not have to join anything: each item gains a
`verification` block, and each document a `verification_summary`.

Flagging rather than deleting. Roughly 1,204 items (3.0% of 39,988) disagree with
the labels on at least one claim, but the errors are mostly small -- median 0.04
m/s on ego speed, 0.44 m on agent position -- so throwing them out would discard
usable supervision. Keeping the verdict lets training down-weight or filter at
will, and lets the tolerances be revisited without another pass over the data.

Three states per item:

  agrees      every recomputed claim matched within tolerance
  disagrees   at least one claim did not
  unchecked   the rationale states no number that can be recomputed (82% of the
              set) -- this is not a pass, it is an absence of evidence

    python -m datatools.flag_qa_verification
    python -m datatools.flag_qa_verification --dry-run
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

from . import paths

QA_DIR = os.path.join(paths.SPLIT_ROOT, "10_radar_vision_qa/qa")
VERIFICATION = os.path.join(paths.SPLIT_ROOT,
                            "10_radar_vision_qa/verification.parquet")

# Which per-quantity errors to carry over, and the tolerance each was judged at.
QUANTITIES = [
    ("ego_pos", "ego_pos_err_m", "m"),
    ("ego_speed", "ego_speed_err_ms", "m/s"),
    ("agent_pos", "agent_pos_err_m", "m"),
    ("distance", "distance_err_m", "m"),
    ("future_pos", "future_pos_err_m", "m"),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean(value):
    """JSON-safe number, or None for NaN/inf."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return round(number, 4)


def build_lookup(frame):
    """(clip_id, qa_index) -> verification block."""
    lookup = {}
    for row in frame.itertuples(index=False):
        checks = int(getattr(row, "n_checks", 0) or 0)
        passes = int(getattr(row, "n_pass", 0) or 0)
        if checks == 0:
            status = "unchecked"
        elif passes == checks:
            status = "agrees"
        else:
            status = "disagrees"
        block = {"status": status, "n_checks": checks, "n_pass": passes}
        errors = {}
        for name, column, unit in QUANTITIES:
            value = clean(getattr(row, column, None))
            if value is not None:
                errors[f"{name}_err"] = value
                errors[f"{name}_unit"] = unit
        if errors:
            block["errors"] = errors
        lookup[(row.clip_id, int(row.qa_index))] = block
    return lookup


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--qa-dir", default=QA_DIR)
    ap.add_argument("--verification", default=VERIFICATION)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not os.path.exists(args.verification):
        raise SystemExit(f"missing {args.verification} - run datatools.verify_qa first")

    frame = pd.read_parquet(args.verification)
    lookup = build_lookup(frame)
    log(f"verification rows: {len(frame):,}")

    files = sorted(glob.glob(os.path.join(args.qa_dir, "*.json")))
    log(f"QA documents: {len(files):,}")

    totals = Counter()
    unmatched = 0
    for path in files:
        with open(path) as fh:
            doc = json.load(fh)
        clip_id = doc["clip_id"]
        per_doc = Counter()
        for index, item in enumerate(doc.get("qa", [])):
            block = lookup.get((clip_id, index))
            if block is None:
                unmatched += 1
                block = {"status": "unchecked", "n_checks": 0, "n_pass": 0,
                         "note": "no verification row"}
            item["verification"] = block
            per_doc[block["status"]] += 1
            totals[block["status"]] += 1
        doc["verification_summary"] = {
            "agrees": per_doc["agrees"],
            "disagrees": per_doc["disagrees"],
            "unchecked": per_doc["unchecked"],
            "any_disagreement": per_doc["disagrees"] > 0,
        }
        if not args.dry_run:
            with open(path, "w") as fh:
                json.dump(doc, fh, ensure_ascii=False, indent=2)

    total = sum(totals.values())
    log(f"items tagged: {total:,}")
    for status in ("agrees", "disagrees", "unchecked"):
        log(f"  {status:11s} {totals[status]:6,}  ({totals[status]/total*100:5.1f}%)")
    if unmatched:
        log(f"  !! {unmatched:,} items had no verification row")

    checked = totals["agrees"] + totals["disagrees"]
    if checked:
        log(f"among checkable items, agreement: "
            f"{totals['agrees']/checked*100:.1f}%")
    log("\nfiltering recipes")
    log("  keep everything, down-weight the bad ones:")
    log("    w = 0.3 if item['verification']['status'] == 'disagrees' else 1.0")
    log("  train only on positively verified items (strictest, loses 82%):")
    log("    item['verification']['status'] == 'agrees'")
    log("  drop only the contradicted ones (recommended):")
    log("    item['verification']['status'] != 'disagrees'")
    if args.dry_run:
        log("dry run - nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
