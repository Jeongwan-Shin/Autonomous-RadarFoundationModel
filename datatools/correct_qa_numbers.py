#!/usr/bin/env python3
"""Replace wrong numbers in QA rationales with the values recomputed from labels.

Only where it is safe. A rationale is an argument, not a list of facts, so
substituting a number can break the argument that surrounds it:

  answer_dependent  the stated number also appears in the answer options, so the
                    correct answer is a function of it. "ego 5.74, sedan 5.22,
                    difference 0.52" with option E "Ego 0.52 m/s faster" -- fix
                    5.74 and the correct difference is no longer on the list.
                    The whole item has to be regenerated; editing the rationale
                    alone leaves it contradicting its own answer key.
  arithmetic        the number feeds a shown calculation ("5.74 - 5.22 = 0.52").
                    Replacing one operand leaves the stated result wrong.
  comparative       the number is an operand of a comparison whose direction
                    could flip ("faster than", "closer than", ">"). The
                    conclusion may invert.
  safe              the number is descriptive context. Replacing it keeps the
                    argument intact.

Only `safe` claims are rewritten. Everything else is left alone and recorded, so
the item can be dropped, down-weighted, or regenerated later.

Run `verify_qa` first; this reads the claim-level results it stores.

    python -m datatools.correct_qa_numbers --dry-run
    python -m datatools.correct_qa_numbers
"""

import argparse
import glob
import io
import json
import os
import re
import shutil
import sys
import time
import zipfile
from collections import Counter

import numpy as np
import pandas as pd

from . import paths
from .qa_claims import RANGE, extract
from .verify_qa import TOLERANCE, evaluate, load_clip

COMPARATIVE = re.compile(
    r"\b(faster|slower|greater|larger|smaller|less|more|closer|farther|further|"
    r"nearer|exceeds?|higher|lower|ahead of|behind|than)\b|[<>]", re.I)
ARITHMETIC = re.compile(r"=|\s[-+]\s|\bdifference\b|\bsum\b|\bminus\b|\bplus\b", re.I)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def format_like(original_text, value):
    """Render `value` with the same decimal precision the text used."""
    if "." in original_text:
        decimals = len(original_text.split(".")[1])
    else:
        decimals = 0
    return f"{value:.{decimals}f}"


def option_numbers(item):
    """Every number appearing in the question or the options."""
    blob = item.get("question", "") + " " + " ".join(item.get("options", {}).values())
    return {m for m in re.findall(r"-?\d+(?:\.\d+)?", blob)}


def classify(claim, item):
    """Why this claim can or cannot be rewritten."""
    sentence = claim.get("sentence", "")
    stated_strings = {claim.get("_stated_text", "")}
    if stated_strings & option_numbers(item):
        return "answer_dependent"
    if RANGE.search(sentence):
        # "around 17.6-17.9 m/s" states a band; replacing one endpoint is
        # meaningless and, before the NUM lookbehind was added, also corrupted
        # the text by swallowing the hyphen.
        return "range"
    if ARITHMETIC.search(sentence):
        return "arithmetic"
    if COMPARATIVE.search(sentence):
        return "comparative"
    return "safe"


def computed_components(claim, computed):
    """Line up the recomputed value(s) with the claim's spans."""
    if claim["kind"] in ("ego_pos", "agent_pos", "future_pos"):
        return list(computed) if isinstance(computed, (list, tuple)) else None
    return [computed]


def process_document(path, clip, dry_run):
    with open(path) as fh:
        doc = json.load(fh)

    stats = Counter()
    for item in doc.get("qa", []):
        rationale = item.get("rationale", "")
        claims = extract(rationale)
        edits = []          # (start, end, new_text, record)
        records = []

        for claim in claims:
            error, computed = evaluate(clip, claim)
            if error is None or error <= TOLERANCE[claim["kind"]]:
                continue
            components = computed_components(claim, computed)
            if components is None or len(components) != len(claim["spans"]):
                stats["skipped_shape"] += 1
                continue

            claim["_stated_text"] = rationale[claim["spans"][0][0]:claim["spans"][0][1]]
            verdict = classify(claim, item)
            record = {"kind": claim["kind"], "frame": claim["frame"],
                      "error": round(float(error), 3),
                      "verdict": verdict,
                      "stated": [rationale[a:b] for a, b in claim["spans"]],
                      "computed": [format_like(rationale[a:b], v)
                                   for (a, b), v in zip(claim["spans"], components)]}
            records.append(record)
            stats[verdict] += 1
            if verdict != "safe":
                continue
            for (start, end), value in zip(claim["spans"], components):
                edits.append((start, end, format_like(rationale[start:end], value)))

        if not records:
            continue

        if edits:
            # Apply right to left so earlier offsets stay valid.
            fixed = rationale
            for start, end, text in sorted(edits, key=lambda e: -e[0]):
                fixed = fixed[:start] + text + fixed[end:]
            if not dry_run:
                item.setdefault("rationale_original", rationale)
                item["rationale"] = fixed
            stats["items_rewritten"] += 1

        if not dry_run:
            item["corrections"] = records

    if not dry_run:
        with open(path, "w") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--qa-dir",
                    default=os.path.join(paths.SPLIT_ROOT, "10_radar_vision_qa/qa"))
    ap.add_argument("--nvidia-root", default=paths.NVIDIA_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    clips = pd.read_parquet(os.path.join(paths.COMMON_DIR, "nvidia_clips.parquet"))
    files = sorted(glob.glob(os.path.join(args.qa_dir, "*.json")))
    if args.limit:
        files = files[: args.limit]
    log(f"{len(files)} QA documents")

    backup = os.path.join(os.path.dirname(args.qa_dir), "qa_before_correction")
    if not args.dry_run and not args.no_backup and not os.path.exists(backup):
        shutil.copytree(args.qa_dir, backup)
        log(f"pre-correction copy at {backup}")

    totals = Counter()
    started = time.monotonic()
    for i, path in enumerate(files, 1):
        clip_id = os.path.basename(path)[:-5]
        if clip_id not in clips.index:
            continue
        row = clips.loc[clip_id]
        try:
            clip = load_clip(args.nvidia_root,
                             {"egomotion_zip": row["egomotion_zip"],
                              "egomotion_member": row["egomotion_member"],
                              "obstacle_zip": row["obstacle_zip"],
                              "obstacle_member": row["obstacle_member"]})
        except Exception:
            totals["load_failed"] += 1
            continue
        totals.update(process_document(path, clip, args.dry_run))
        if i % 400 == 0 or i == len(files):
            log(f"  {i}/{len(files)}  {i/(time.monotonic()-started):.1f}/s")

    log("")
    log("disagreeing claims by verdict")
    order = ["safe", "comparative", "arithmetic", "range", "answer_dependent",
             "skipped_shape"]
    total = sum(totals[k] for k in order)
    for key in order:
        if totals[key]:
            log(f"  {key:17s} {totals[key]:6,}  ({totals[key]/total*100:5.1f}%)")
    log(f"  {'total':17s} {total:6,}")
    log(f"items with a rewritten rationale: {totals['items_rewritten']:,}")
    if totals["load_failed"]:
        log(f"clips that failed to load: {totals['load_failed']:,}")
    if args.dry_run:
        log("dry run - nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
