#!/usr/bin/env python3
"""A scorer per task, because one metric cannot describe eleven of them.

Teacher-forced loss is comparable across tasks and says almost nothing about
any of them. A detection answer listing the right cars in a different order
scores badly; a fabricated list of the right length scores well. A waypoint
answer that is two metres out and one that is two hundred metres out differ by
a few nats. So every task is generated and then scored on its own terms:

  object lists   parsed into objects and matched by position, the way a
                 detector is scored -- order-free, and a miss is a miss
  waypoints      mean displacement error in metres against the reference path
  trajectories   range and azimuth error at each horizon
  quantities     correlation and absolute error of the number itself
  tags           set overlap, since order carries no meaning
  choices        exact match on the letter
  free text      left to loss; there is no honest reference-free score for a
                 sentence, and pretending otherwise would be worse than saying so

Each scorer takes (generated, reference) strings and returns a dict of metrics
plus `n`, so results aggregate by summing and dividing.
"""

import math
import re

# "automobile 34 m az +13 deg moving", "#20 automobile 34 m visible 3.1 s"
OBJECT = re.compile(
    r"(?:#(?P<tid>\d+)\s+)?(?P<cls>[a-z_]+)\s+(?P<rng>-?\d+(?:\.\d+)?)\s*m"
    r"(?:\s*az\s*(?P<az>[+-]?\d+(?:\.\d+)?)\s*deg)?"
    r"(?P<motion>\s+moving|\s+stationary)?")
# "+1s (+12.5, +0.2)"
WAYPOINT = re.compile(r"\+(\d)s\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)")
# "+1s 22 m az +16 deg"
HORIZON = re.compile(r"\+(\d)s\s+(-?\d+(?:\.\d+)?)\s*m\s*az\s*([+-]?\d+(?:\.\d+)?)")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
MATCH_THRESHOLD_M = 2.0
# Task 02's answers carry no azimuth, so both sides land on the x-axis and any
# two objects at a similar range match regardless of true bearing. Measured, that
# inflated its F1 to six times task 01's on the same model. A range-only match
# has to be tighter to mean the same thing.
RANGE_ONLY_THRESHOLD_M = 1.0


def parse_objects(text):
    out = []
    for part in (text or "").split(";"):
        m = OBJECT.search(part)
        if not m:
            continue
        az = m.group("az")
        out.append({"tid": int(m.group("tid")) if m.group("tid") else None,
                    "cls": m.group("cls"), "rng": float(m.group("rng")),
                    "az": float(az) if az is not None else 0.0,
                    "has_az": az is not None,
                    "moving": (m.group("motion") or "").strip() == "moving"})
    return out


def _xy(o):
    a = math.radians(o["az"])
    return o["rng"] * math.cos(a), o["rng"] * math.sin(a)


def _match(predicted, truth, threshold=None):
    """Greedy nearest-first pairing, order-free."""
    if threshold is None:
        has_az = any(o["has_az"] for o in truth) or any(o["has_az"] for o in predicted)
        threshold = MATCH_THRESHOLD_M if has_az else RANGE_ONLY_THRESHOLD_M
    candidates = []
    for i, p in enumerate(predicted):
        px, py = _xy(p)
        for j, t in enumerate(truth):
            tx, ty = _xy(t)
            d = math.hypot(px - tx, py - ty)
            if d <= threshold:
                candidates.append((d, i, j))
    pairs, used_p, used_t = [], set(), set()
    for _, i, j in sorted(candidates):
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        pairs.append((predicted[i], truth[j]))
    return pairs


def score_objects(generated, reference):
    predicted, truth = parse_objects(generated), parse_objects(reference)
    pairs = _match(predicted, truth)
    out = {"n": 1, "tp": len(pairs), "fp": len(predicted) - len(pairs),
           "fn": len(truth) - len(pairs), "matched": len(pairs),
           "class_ok": 0, "motion_ok": 0, "id_ok": 0, "id_total": 0,
           "range_err": 0.0, "az_err": 0.0, "az_n": 0}
    for p, t in pairs:
        out["class_ok"] += p["cls"] == t["cls"]
        out["motion_ok"] += p["moving"] == t["moving"]
        out["range_err"] += abs(p["rng"] - t["rng"])
        if p["has_az"] and t["has_az"]:
            out["az_err"] += abs(p["az"] - t["az"])
            out["az_n"] += 1
        if t["tid"] is not None:
            out["id_total"] += 1
            out["id_ok"] += p["tid"] == t["tid"]
    return out


def score_waypoints(generated, reference):
    """Mean displacement error in metres, over the horizons both mention."""
    got = {int(h): (float(x), float(y)) for h, x, y in WAYPOINT.findall(generated or "")}
    want = {int(h): (float(x), float(y)) for h, x, y in WAYPOINT.findall(reference or "")}
    shared = sorted(set(got) & set(want))
    out = {"n": 1, "horizons": len(shared), "expected": len(want), "err": 0.0}
    for h in shared:
        out["err"] += math.hypot(got[h][0] - want[h][0], got[h][1] - want[h][1])
    return out


def score_trajectory(generated, reference):
    got = {int(h): (float(r), float(a)) for h, r, a in HORIZON.findall(generated or "")}
    want = {int(h): (float(r), float(a)) for h, r, a in HORIZON.findall(reference or "")}
    shared = sorted(set(got) & set(want))
    out = {"n": 1, "horizons": len(shared), "expected": len(want),
           "range_err": 0.0, "az_err": 0.0}
    for h in shared:
        out["range_err"] += abs(got[h][0] - want[h][0])
        out["az_err"] += abs(got[h][1] - want[h][1])
    return out


def score_quantity(generated, reference):
    """First number of each. Correlation is computed later, over all items."""
    got, want = NUMBER.findall(generated or ""), NUMBER.findall(reference or "")
    if not got or not want:
        return {"n": 1, "parsed": 0, "pred": None, "truth": None,
                "abs_err": 0.0, "rel_err": 0.0, "scored": 0}
    p, t = float(got[0]), float(want[0])
    return {"n": 1, "parsed": 1, "pred": p, "truth": t, "scored": 1,
            "abs_err": abs(p - t), "rel_err": abs(p - t) / max(abs(t), 1.0)}


def score_tags(generated, reference):
    """Semicolon-separated tags, compared as sets."""
    split = lambda s: {p.strip().lower() for p in (s or "").split(";") if p.strip()}
    got, want = split(generated), split(reference)
    hit = len(got & want)
    return {"n": 1, "tp": hit, "fp": len(got - want), "fn": len(want - got)}


def score_choice(generated, reference):
    letter = lambda s: next((c for c in (s or "").strip() if c in "ABCD"), None)
    return {"n": 1, "correct": int(letter(generated) is not None
                                   and letter(generated) == letter(reference))}


def score_text(generated, reference):
    """No reference-free sentence score is honest here; loss carries this task."""
    return {"n": 1}


SCORERS = {
    "det_objects": score_objects,
    "track_identity": score_objects,
    "motion_seg": score_objects,
    "plan_ego": score_waypoints,
    "agent_traj": score_trajectory,
    "radar_probe": score_quantity,
    "radar_transfer": score_quantity,
    "radar_structure": score_quantity,
    "depth_range": score_quantity,
    "world_model": score_quantity,
    "retrieval": score_tags,
    "qa": score_choice,
}


def scorer_for(task):
    return SCORERS.get(task, score_text)


def summarise(task, records, correlation=None):
    """Fold per-item dicts into the numbers worth printing."""
    if not records:
        return {}
    total = {}
    for r in records:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                total[k] = total.get(k, 0) + v
    n = total.get("n", len(records))
    fn = scorer_for(task)

    if fn is score_objects:
        tp, fp, miss = total["tp"], total["fp"], total["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + miss, 1)
        matched = max(total["matched"], 1)
        return {"metric": "detection", "n": n,
                "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / max(precision + recall, 1e-9),
                "class_acc": total["class_ok"] / matched,
                "motion_acc": total["motion_ok"] / matched,
                "range_mae": total["range_err"] / matched,
                "az_mae": total["az_err"] / max(total["az_n"], 1),
                "matched": total["matched"],
                "id_acc": (total["id_ok"] / total["id_total"]
                           if total["id_total"] else None)}
    if fn is score_waypoints:
        return {"metric": "waypoints", "n": n,
                "displacement_mae_m": total["err"] / max(total["horizons"], 1),
                "coverage": total["horizons"] / max(total["expected"], 1)}
    if fn is score_trajectory:
        h = max(total["horizons"], 1)
        return {"metric": "trajectory", "n": n,
                "range_mae_m": total["range_err"] / h,
                "az_mae_deg": total["az_err"] / h,
                "coverage": total["horizons"] / max(total["expected"], 1)}
    if fn is score_quantity:
        scored = max(total["scored"], 1)
        return {"metric": "quantity", "n": n,
                "parsed": total["parsed"] / n,
                "abs_err": total["abs_err"] / scored,
                "rel_err": total["rel_err"] / scored,
                "corr": correlation}
    if fn is score_tags:
        tp, fp, miss = total["tp"], total["fp"], total["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + miss, 1)
        return {"metric": "tags", "n": n, "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / max(precision + recall, 1e-9)}
    if fn is score_choice:
        return {"metric": "choice", "n": n, "accuracy": total["correct"] / n}
    return {"metric": "text (loss only)", "n": n}
