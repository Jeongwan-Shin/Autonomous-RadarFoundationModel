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


# "automobile (34.2, 7.9, 0.8) moving" -- the same objects in rig coordinates.
# Two output formats exist because the instruction selects between them, and the
# matcher works in Cartesian space either way: a polar answer is converted, so a
# Cartesian one simply skips a conversion whose rounding error grows with range
# (one degree of azimuth is 0.6 m at 34 m and 1.7 m at 100 m).
# "automobile (9.7, -13.0, 0.9) 4.5x1.9x1.5 yaw +12" -- a 3D box. The extent
# and the heading are optional in the pattern so an answer that gives only a
# centre still parses: a model that has not learnt to emit them should be
# scored on what it did emit, not dropped as unreadable.
OBJECT_XYZ = re.compile(
    r"(?:#(?P<tid>\d+)\s+)?(?P<cls>[a-z_]+)\s*\(\s*"
    r"(?P<x>[+-]?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y>[+-]?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<z>[+-]?\d+(?:\.\d+)?)\s*\)"
    r"(?:\s+(?:size\s+)?(?P<l>\d+(?:\.\d+)?)\s*x\s*(?P<w>\d+(?:\.\d+)?)"
    r"\s*x\s*(?P<h>\d+(?:\.\d+)?)(?:\s*m)?)?"
    r"(?:\s+yaw\s+(?P<yaw>[+-]?\d+(?:\.\d+)?)(?:\s*deg)?)?"
    # Heading is now a sector index 0..11 rather than degrees; the old form is
    # kept so saved runs still parse.
    r"(?:\s+heading\s+(?P<sector>\d+))?"
    r"(?P<motion>\s+moving|\s+stationary)?")


# "#1 automobile [117, 445, 387, 772] moving" -- an image box instead of a
# world position. Matching these by metres is meaningless, so items carrying a
# box are paired by intersection-over-union instead.
OBJECT_BBOX = re.compile(
    r"(?:#(?P<tid>\d+)\s+)?(?P<cls>[a-z_]+)\s*\[\s*"
    r"(?P<x1>\d+)\s*,\s*(?P<y1>\d+)\s*,\s*"
    r"(?P<x2>\d+)\s*,\s*(?P<y2>\d+)\s*\]"
    r"(?P<motion>\s+moving|\s+stationary)?")
IOU_THRESHOLD = 0.3


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    overlap = ix * iy
    union = ((ax2-ax1) * (ay2-ay1)) + ((bx2-bx1) * (by2-by1)) - overlap
    return overlap / union if union > 0 else 0.0


def parse_objects(text):
    """Every object in an answer, whatever separates them.

    This used to split on ';', which the object lists use but the moving /
    stationary answer does not -- it separates with commas, inside a sentence
    that also contains commas within coordinate tuples. The effect was silent:
    one object parsed out of four, so that task's F1 was computed from a quarter
    of its answer. Scanning the whole string with each form's pattern and
    dropping overlaps has no separator to get wrong.
    """
    found = []
    for pattern, kind in ((OBJECT_BBOX, "bbox"), (OBJECT_XYZ, "xyz"),
                          (OBJECT, "polar")):
        for m in pattern.finditer(text or ""):
            found.append((m.start(), m.end(), kind, m))
    out, taken = [], []
    # Longest first at each position, so "automobile [1, 2, 3, 4]" is read as a
    # box rather than as a class with no geometry.
    for start, end, kind, m in sorted(found, key=lambda f: (f[0], -(f[1] - f[0]))):
        if any(start < b and a < end for a, b in taken):
            continue
        taken.append((start, end))
        tid = int(m.group("tid")) if m.groupdict().get("tid") else None
        moving = (m.groupdict().get("motion") or "").strip() == "moving"
        if kind == "bbox":
            out.append({"tid": tid, "cls": m.group("cls"),
                        "bbox": [float(m.group(k)) for k in ("x1", "y1", "x2", "y2")],
                        "x": None, "y": None, "z": None, "rng": None, "az": None,
                        "has_az": False, "moving": moving})
        elif kind == "xyz":
            x, y = float(m.group("x")), float(m.group("y"))
            g = m.groupdict()
            size = ([float(g[k]) for k in ("l", "w", "h")]
                    if g.get("l") is not None else None)
            yaw = float(g["yaw"]) if g.get("yaw") is not None else None
            if yaw is None and g.get("sector") is not None:
                yaw = (int(g["sector"]) * 30.0 + 180.0) % 360.0 - 180.0
            out.append({"tid": tid, "cls": m.group("cls"), "x": x, "y": y,
                        "z": float(m.group("z")), "rng": math.hypot(x, y),
                        "az": math.degrees(math.atan2(y, x)), "has_az": True,
                        "bbox": None, "moving": moving,
                        "size": size, "yaw": yaw})
        else:
            az = m.group("az")
            out.append({"tid": tid, "cls": m.group("cls"),
                        "rng": float(m.group("rng")),
                        "az": float(az) if az is not None else 0.0,
                        "has_az": az is not None, "x": None, "y": None,
                        "z": None, "bbox": None, "moving": moving})
    return out


def _xy(o):
    if o.get("x") is not None:
        return o["x"], o["y"]
    a = math.radians(o["az"])
    return o["rng"] * math.cos(a), o["rng"] * math.sin(a)


def _match(predicted, truth, threshold=None):
    """Greedy best-first pairing, order-free.

    Image boxes are paired by IoU, world positions by metres. Mixing the two
    would compare a pixel distance against a metric one.
    """
    if any(o.get("bbox") for o in truth) or any(o.get("bbox") for o in predicted):
        candidates = []
        for i, p in enumerate(predicted):
            if not p.get("bbox"):
                continue
            for j, q in enumerate(truth):
                if not q.get("bbox"):
                    continue
                overlap = _iou(p["bbox"], q["bbox"])
                if overlap >= IOU_THRESHOLD:
                    candidates.append((-overlap, i, j))
        pairs, used_p, used_t = [], set(), set()
        for _, i, j in sorted(candidates):
            if i in used_p or j in used_t:
                continue
            used_p.add(i)
            used_t.add(j)
            pairs.append((predicted[i], truth[j]))
        return pairs
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


def _per_class(total):
    """{class: {f1, recall, precision, n}} from the cls_* counters.

    One F1 hides the imbalance rather than reporting it. 84.5% of labelled
    objects are automobiles and 10.3% are people; the remaining eight classes
    together are 5.2%, and animal is 0.09%. A model that never detects a
    cyclist loses well under a point of overall F1, so the aggregate cannot
    tell it apart from a model that detects everything.

    The mix cannot be evened out by dropping items either -- the task is "list
    every road user", and 93.95% of detection answers contain an automobile.
    Keeping only the answers that carry a rare class still leaves 61.4%
    automobiles and throws away three quarters of the data; keeping only the
    answers with no automobile leaves person at 80.5% and throws away 93%. So
    the class mix stays as the road is, and this reports what happens per
    class instead of distorting the scenes to make one number look even.
    """
    names = {k[4:].rsplit("_", 1)[0] for k in total if k.startswith("cls_")}
    out = {}
    for c in sorted(names):
        tp = total.get(f"cls_{c}_tp", 0)
        fn = total.get(f"cls_{c}_fn", 0)
        fp = total.get(f"cls_{c}_fp", 0)
        if tp + fn == 0:                 # never in the truth: nothing to score
            continue
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out[c] = {"f1": 2 * prec * rec / max(prec + rec, 1e-9),
                  "recall": rec, "precision": prec, "n": tp + fn}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"])) or None


def score_objects(generated, reference):
    predicted, truth = parse_objects(generated), parse_objects(reference)
    pairs = _match(predicted, truth)
    out = {"n": 1, "tp": len(pairs), "fp": len(predicted) - len(pairs),
           "fn": len(truth) - len(pairs), "matched": len(pairs),
           "class_ok": 0, "motion_ok": 0, "id_ok": 0, "id_total": 0,
           "range_err": 0.0, "az_err": 0.0, "az_n": 0,
           "z_err": 0.0, "z_n": 0,
           "size_err": 0.0, "size_n": 0, "yaw_err": 0.0, "yaw_n": 0}
    for p, t in pairs:
        out[f"cls_{t['cls']}_tp"] = out.get(f"cls_{t['cls']}_tp", 0) + 1
        out["class_ok"] += p["cls"] == t["cls"]
        out["motion_ok"] += p["moving"] == t["moving"]
        if p.get("size") and t.get("size"):
            # Mean absolute error over length, width and height, so one number
            # is comparable across objects of very different size.
            out["size_err"] += sum(abs(a - b) for a, b
                                   in zip(p["size"], t["size"])) / 3.0
            out["size_n"] += 1
        if p.get("yaw") is not None and t.get("yaw") is not None:
            # Wrapped: +179 and -179 are two degrees apart, not 358.
            d = abs(p["yaw"] - t["yaw"]) % 360.0
            out["yaw_err"] += min(d, 360.0 - d)
            out["yaw_n"] += 1
        if p.get("rng") is not None and t.get("rng") is not None:
            out["range_err"] += abs(p["rng"] - t["rng"])
        if p.get("z") is not None and t.get("z") is not None:
            out["z_err"] += abs(p["z"] - t["z"])
            out["z_n"] += 1
        if p["has_az"] and t["has_az"]:
            out["az_err"] += abs(p["az"] - t["az"])
            out["az_n"] += 1
        if t["tid"] is not None:
            out["id_total"] += 1
            out["id_ok"] += p["tid"] == t["tid"]
    seen_t = {id(o) for _, o in pairs}
    seen_p = {id(o) for o, _ in pairs}
    for o in truth:
        if id(o) not in seen_t:
            out[f"cls_{o['cls']}_fn"] = out.get(f"cls_{o['cls']}_fn", 0) + 1
    for o in predicted:
        if id(o) not in seen_p:
            out[f"cls_{o['cls']}_fp"] = out.get(f"cls_{o['cls']}_fp", 0) + 1
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


# "+1s 8.9 m/s, yaw +0.4 deg/s"
CONTROL = re.compile(r"\+(\d)s\s+(-?\d+(?:\.\d+)?)\s*m/s\s*,\s*yaw\s*"
                     r"([+-]?\d+(?:\.\d+)?)\s*deg/s")


def score_control(generated, reference):
    """Speed and yaw rate at each horizon, over the horizons both mention."""
    got = {int(h): (float(v), float(w)) for h, v, w in CONTROL.findall(generated or "")}
    want = {int(h): (float(v), float(w)) for h, v, w in CONTROL.findall(reference or "")}
    shared = sorted(set(got) & set(want))
    out = {"n": 1, "horizons": len(shared), "expected": len(want),
           "speed_err": 0.0, "yaw_err": 0.0}
    for h in shared:
        out["speed_err"] += abs(got[h][0] - want[h][0])
        out["yaw_err"] += abs(got[h][1] - want[h][1])
    return out


LEAVES = "leaves the forward sector"
# "+2s leaves the forward sector"
HORIZON_GONE = re.compile(r"\+(\d)s\s+leaves the forward sector")
# "+1s [117, 445, 387, 772]"
HORIZON_BBOX = re.compile(r"\+(\d)s\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*"
                          r"(\d+)\s*,\s*(\d+)\s*\]")


def _horizons(text):
    """{horizon: value}: a polar pair, an (x, y) offset, a box, or gone."""
    out = {}
    for h, r, a in HORIZON.findall(text or ""):
        out[int(h)] = ("polar", float(r), float(a))
    for h, x, y in WAYPOINT.findall(text or ""):
        out[int(h)] = ("xy", float(x), float(y))
    for h, x1, y1, x2, y2 in HORIZON_BBOX.findall(text or ""):
        out[int(h)] = ("bbox", [float(x1), float(y1), float(x2), float(y2)])
    for h in HORIZON_GONE.findall(text or ""):
        out[int(h)] = ("gone",)
    return out


def score_trajectory(generated, reference):
    """Where the named object goes, in whichever form the instruction asked for.

    This read only the polar form, so `agent_traj_bbox` -- whose answer is
    "+1s [478, 674, 487, 686]" -- matched nothing and reported coverage 0.00 on
    every item. That reads as a model producing no answer; it was producing
    boxes within ten pixels of the truth. `_horizons` already handled all three
    forms, including "leaves the forward sector"; this simply never called it.

    Boxes are scored by IoU rather than metres, and an exit is right only when
    the reference also exits -- claiming an object left when it stayed is a
    miss, not a free pass.
    """
    got, want = _horizons(generated), _horizons(reference)
    shared = sorted(set(got) & set(want))
    out = {"n": 1, "horizons": 0, "expected": len(want),
           "range_err": 0.0, "az_err": 0.0, "iou": 0.0, "iou_n": 0,
           "gone_ok": 0, "gone_n": 0, "disp_err": 0.0, "disp_n": 0,
           "final_err": 0.0, "final_n": 0, "polar_n": 0}
    for h in shared:
        a, b = got[h], want[h]
        if b[0] == "gone":
            out["gone_n"] += 1
            out["gone_ok"] += a[0] == "gone"
            out["horizons"] += 1
        elif a[0] == b[0] == "polar":
            out["horizons"] += 1
            out["polar_n"] += 1
            out["range_err"] += abs(a[1] - b[1])
            out["az_err"] += abs(a[2] - b[2])
        elif a[0] == b[0] == "xy":
            # Displacement in metres -- what ADE and FDE mean. The polar branch
            # above cannot produce this: it sums a range error in metres and a
            # bearing error in degrees separately, and those do not add.
            out["horizons"] += 1
            err = math.hypot(a[1] - b[1], a[2] - b[2])
            out["disp_err"] += err
            out["disp_n"] += 1
            if h == max(want):
                out["final_err"] += err
                out["final_n"] += 1
        elif a[0] == b[0] == "bbox":
            out["horizons"] += 1
            out["iou"] += _iou(a[1], b[1])
            out["iou_n"] += 1
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


# The QA set offers five options, not four. Matching only A-D scored every
# E-answer wrong no matter what the model wrote, and E is 21.2% of the test
# questions -- so the ceiling on this task was 78.8% and the floor for guessing
# was misstated too.
CHOICES = "ABCDE"
# Scanning for the first character in "ABCDE" also matches the A inside
# "Answer:", so any model that prefixed its choice was graded on the prefix.
# An option letter stands alone: not part of a longer word.
CHOICE_LETTER = re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])")


def score_choice(generated, reference):
    def letter(s):
        found = CHOICE_LETTER.findall((s or "").strip())
        return found[0] if found else None
    return {"n": 1, "correct": int(letter(generated) is not None
                                   and letter(generated) == letter(reference))}


def score_text(generated, reference):
    """No reference-free sentence score is honest here; loss carries this task."""
    return {"n": 1}


SCORERS = {
    "det_objects_azdeg": score_objects,
    "det_objects_3dbbox": score_objects,
    "motion_seg_azdeg": score_objects,
    "motion_seg_bbox": score_objects,
    "plan_ego_xy": score_waypoints,
    "plan_ego_control": score_control,
    "agent_traj_azdeg": score_trajectory,
    "agent_traj_bbox": score_trajectory,
    "agent_traj_xy": score_trajectory,
    "radar_probe": score_quantity,
    "radar_transfer": score_quantity,
    "radar_structure": score_quantity,
    "radar_objects": score_quantity,
    "depth_range": score_quantity,
    "world_model": score_quantity,
    "retrieval": score_tags,
    "qa": score_choice,
}


def scorer_for(task):
    return SCORERS.get(task, score_text)


def summarise_with(fn, records, correlation=None):
    """`summarise` keyed on the scorer rather than the task name."""
    return summarise("", records, correlation, _fn=fn)


def summarise(task, records, correlation=None, _fn=None):
    """Fold per-item dicts into the numbers worth printing."""
    if not records:
        return {}
    total = {}
    for r in records:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                total[k] = total.get(k, 0) + v
    n = total.get("n", len(records))
    fn = _fn or scorer_for(task)
    if hasattr(fn, "plain"):
        out = summarise_with(fn.plain, records, correlation)
        out["metric"] = "cot " + out["metric"]
        if total.get("cot_parsed") is not None:
            out["cot_parsed"] = total["cot_parsed"] / n
        return out

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
                "size_mae": (total["size_err"] / total["size_n"]
                             if total.get("size_n") else None),
                "yaw_mae": (total["yaw_err"] / total["yaw_n"]
                            if total.get("yaw_n") else None),
                "id_acc": (total["id_ok"] / total["id_total"]
                           if total["id_total"] else None),
                "per_class": _per_class(total)}
    if fn is score_waypoints:
        return {"metric": "waypoints", "n": n,
                "displacement_mae_m": total["err"] / max(total["horizons"], 1),
                "coverage": total["horizons"] / max(total["expected"], 1)}
    if fn is score_trajectory:
        # Each error is divided by the horizons that actually carried its form.
        # Dividing all of them by `horizons` made an (x, y) task report
        # `range_mae_m` 0.0 -- not "exact", but "no polar answer was read",
        # which is the failure this scorer has already been bitten by twice.
        polar = max(total.get("polar_n", 0), 1)
        return {"metric": "trajectory", "n": n,
                "range_mae_m": (total["range_err"] / polar
                                if total.get("polar_n") else None),
                "az_mae_deg": (total["az_err"] / polar
                               if total.get("polar_n") else None),
                "ade_m": (total["disp_err"] / total["disp_n"]
                          if total.get("disp_n") else None),
                "fde_m": (total["final_err"] / total["final_n"]
                          if total.get("final_n") else None),
                "iou": (total["iou"] / total["iou_n"]
                        if total.get("iou_n") else None),
                "gone_acc": (total["gone_ok"] / total["gone_n"]
                             if total.get("gone_n") else None),
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
    # `score_tracking` and `score_control` were never given a branch here, so
    # six tasks -- the four tracking forms and both plan_ego control forms --
    # reported "text (loss only)" and produced no number at all. They were
    # generated, scored per item and then thrown away at the fold.
    if fn is score_tracking:
        tp, fp, miss = total["tp"], total["fp"], total["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + miss, 1)
        matched = max(total["matched"], 1)
        return {"metric": "tracking", "n": n,
                "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / max(precision + recall, 1e-9),
                "class_acc": total["class_ok"] / matched,
                "id_carried": (total["id_carried"] / total["id_checkable"]
                               if total["id_checkable"] else None),
                "id_checkable": total["id_checkable"]}
    if fn is score_control:
        h = max(total["horizons"], 1)
        return {"metric": "control", "n": n,
                "speed_mae_ms": total["speed_err"] / h,
                "yaw_mae_degs": total["yaw_err"] / h,
                "coverage": total["horizons"] / max(total["expected"], 1)}
    return {"metric": "text (loss only)", "n": n}


# --------------------------------------------------------------------------
# rewards
# --------------------------------------------------------------------------
#
# GRPO needs one number in [0, 1] per generated answer, and the honest source of
# that number is the scorer the task is already evaluated with. Deriving the
# reward from the scorer keeps the two from drifting: an answer that scores well
# is by construction an answer the reward paid for.
#
# Until now `train_grpo.py` had a single reward that read numbers out of the
# text, so only the tasks whose answer *is* a number could be trained -- three
# of eleven. On `det_objects` it would have graded the first integer of an
# object list, which is not the task. These map each scorer's output onto a
# reward instead, so every task with a checkable answer can be trained on its
# own terms.

# Tolerances: the error at which a prediction is worth half credit. Set from the
# scale each task works at, not tuned -- ego waypoints live within a couple of
# metres over 3 s, an agent's range spans tens.
HALF_CREDIT = {"plan_ego": 1.0, "agent_traj": 2.0, "agent_traj_az": 5.0}


def _f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom else 0.0


def _decay(error, half):
    """1 at no error, 0.5 at `half`, asymptotically 0. Graded everywhere, which
    is the whole reason for preferring a reward to cross-entropy."""
    return float(half / (half + max(error, 0.0)))


def reward_objects(generated, reference):
    """Detection F1, then credit for getting each matched object right."""
    s = score_objects(generated, reference)
    f1 = _f1(s["tp"], s["fp"], s["fn"])
    if not s["matched"]:
        return f1
    detail = (s["class_ok"] + s["motion_ok"]) / (2.0 * s["matched"])
    place = _decay(s["range_err"] / s["matched"], 5.0)
    # Half the reward for finding the objects, half for describing them: F1
    # alone is satisfied by a list of plausible objects at invented ranges.
    return float(0.5 * f1 + 0.5 * f1 * (0.5 * detail + 0.5 * place))


def reward_waypoints(generated, reference):
    s = score_waypoints(generated, reference)
    if not s["horizons"]:
        return 0.0
    covered = s["horizons"] / max(s["expected"], 1)
    return float(covered * _decay(s["err"] / s["horizons"], HALF_CREDIT["plan_ego"]))


def reward_trajectory(generated, reference):
    """Per horizon, and "it will be gone" counts as an answer.

    A track survives all three horizons only 51.4% of the time, so predicting
    that it leaves the sector is the common case rather than an edge one, and
    scoring it as a missing horizon would pay a model for saying nothing.
    Boxes are scored by IoU and world positions by metres; the horizons are
    compared one by one so a right answer at +1s still counts when the object is
    gone by +2s.
    """
    got, want = _horizons(generated), _horizons(reference)
    if not want:
        return 0.0
    total = 0.0
    for h, truth in want.items():
        mine = got.get(h)
        if mine is None or mine[0] != truth[0]:
            continue
        if truth[0] == "gone":
            total += 1.0
        elif truth[0] == "bbox":
            total += _iou(mine[1], truth[1])
        elif truth[0] == "xy":
            # One displacement, not a range and a bearing averaged. Falling
            # through to the polar branch would have read x as a range and y as
            # an azimuth and paid the model on two decays that mean nothing.
            total += _decay(math.hypot(mine[1] - truth[1], mine[2] - truth[2]),
                            HALF_CREDIT["agent_traj"])
        else:
            rng = _decay(abs(mine[1] - truth[1]), HALF_CREDIT["agent_traj"])
            az = _decay(abs(mine[2] - truth[2]), HALF_CREDIT["agent_traj_az"])
            total += 0.5 * (rng + az)
    return float(total / len(want))


def reward_control(generated, reference):
    """Half speed, half yaw rate, scaled by how many horizons were answered.

    Half credit at 1 m/s and at 3 deg/s. Those are the scales the quantities
    move on over three seconds -- the ego's speed spans 6.5 to 10.8 m/s within a
    single clip -- so an error of that size is a real miss rather than rounding.
    """
    s = score_control(generated, reference)
    if not s["horizons"]:
        return 0.0
    covered = s["horizons"] / max(s["expected"], 1)
    speed = _decay(s["speed_err"] / s["horizons"], 1.0)
    yaw = _decay(s["yaw_err"] / s["horizons"], 3.0)
    return float(covered * 0.5 * (speed + yaw))


def reward_quantity(generated, reference):
    """Every number in the answer, not just the first: several of these ask two
    quantities at once and grading only the first leaves the second free."""
    got, want = NUMBER.findall(generated or ""), NUMBER.findall(reference or "")
    if not got or not want:
        return 0.0
    scores = []
    for i, w in enumerate(want):
        if i >= len(got):
            scores.append(0.0)
            continue
        t, p = float(w), float(got[i])
        scores.append(max(0.0, 1.0 - abs(p - t) / max(abs(t), 1.0)))
    return float(sum(scores) / len(scores))


def reward_tags(generated, reference):
    s = score_tags(generated, reference)
    return _f1(s["tp"], s["fp"], s["fn"])


def reward_choice(generated, reference):
    return float(score_choice(generated, reference)["correct"])


REWARDS = {
    "det_objects_azdeg": reward_objects,
    "det_objects_3dbbox": reward_objects,
    "motion_seg_azdeg": reward_objects,
    "motion_seg_bbox": reward_objects,
    "plan_ego_xy": reward_waypoints,
    "plan_ego_control": reward_control,
    "agent_traj_azdeg": reward_trajectory,
    "agent_traj_bbox": reward_trajectory,
    "agent_traj_xy": reward_trajectory,
    "radar_probe": reward_quantity,
    "radar_transfer": reward_quantity,
    "depth_range": reward_quantity,
    "world_model": reward_quantity,
    "radar_objects": reward_quantity,
    "retrieval": reward_tags,
    "qa": reward_choice,
}


def reward_for(task):
    """None where no checkable reward exists. The `desc_*` tasks are free text
    and any reward invented for them would be a preference model in disguise,
    which is the thing RLVR exists to avoid."""
    return REWARDS.get(task)


# --------------------------------------------------------------------------
# descriptions, and rationales
# --------------------------------------------------------------------------
#
# The `desc_*` tasks were left out of RLVR on the grounds that free text has no
# checkable answer. That was wrong. Every description in this dataset is written
# from measured quantities and states them outright -- "the short-range radar
# returns 244 detections, 170 of which are not explained by the ego's own
# motion", "Ahead: 2 cars. The closest is a car at 20 m." Those claims are as
# checkable as any number `radar_probe` asks for; what is not checkable is the
# prose around them, and a reward that ignores the prose and grades the claims
# is still a verifiable reward, not a preference model in disguise.

# Words that carry a claim rather than decoration. Getting "braking" where the
# truth says "accelerating" is an error of the same kind as a wrong number.
CLAIM_WORDS = (
    "accelerating", "braking", "steady", "turning", "left", "right",
    "moving", "stationary", "closing", "receding",
    "none", "all", "some", "camera", "radar",
    "car", "cars", "truck", "trucks", "person", "people", "bus", "rider",
)


def _claims(text):
    words = re.findall(r"[a-z]+", (text or "").lower())
    return {w for w in words if w in CLAIM_WORDS}


def score_description(generated, reference):
    """Numbers and claim words, each compared against the reference."""
    got, want = NUMBER.findall(generated or ""), NUMBER.findall(reference or "")
    matched = 0
    for i, w in enumerate(want):
        if i < len(got):
            t, p = float(w), float(got[i])
            matched += max(0.0, 1.0 - abs(p - t) / max(abs(t), 1.0))
    a, b = _claims(generated), _claims(reference)
    return {"n": 1, "numbers": len(want), "number_credit": matched,
            "claim_tp": len(a & b), "claim_fp": len(a - b), "claim_fn": len(b - a)}


# Saying the opposite is worse than saying nothing, and set F1 cannot express
# that: "None are moving" against "All 2 are moving" shares every other word and
# scored 0.708 before this. Each contradiction is charged directly.
ANTONYMS = (("all", "none"), ("moving", "stationary"), ("closing", "receding"),
            ("accelerating", "braking"), ("left", "right"))
CONTRADICTION_COST = 0.35


def _contradictions(got, want):
    n = 0
    for a, b in ANTONYMS:
        if (a in got and b in want) or (b in got and a in want):
            n += 1
    return n


def reward_description(generated, reference):
    """Half for the numbers, half for the claim words, minus contradictions.

    Numbers alone would let a model recite the right counts inside a sentence
    that says the opposite -- "0 of the 2 objects are moving" and "all 2 are
    moving" share their numbers. Claim words alone would reward the shape of the
    sentence and nothing about the scene."""
    s = score_description(generated, reference)
    numeric = s["number_credit"] / s["numbers"] if s["numbers"] else None
    claim = _f1(s["claim_tp"], s["claim_fp"], s["claim_fn"])
    base = claim if numeric is None else 0.5 * numeric + 0.5 * claim
    got, want = _claims(generated), _claims(reference)
    return float(max(0.0, base - CONTRADICTION_COST * _contradictions(got, want)))


RATIONALE_JSON = re.compile(
    r'"rationale"\s*:\s*"(.*?)"\s*,\s*"answer"\s*:\s*"(.*?)"\s*\}', re.S)


def split_rationale(text):
    """(rationale, answer) from a {"rationale": ..., "answer": ...} generation.

    Falls back to (None, whole text) when the model did not produce the format,
    so a task without chain-of-thought scores exactly as it did before and a
    model that breaks format is graded on what it actually wrote.
    """
    m = RATIONALE_JSON.search(text or "")
    return (m.group(1), m.group(2)) if m else (None, text)


def _call_reward(fn, generated, reference, prompt):
    """Call a reward, passing the prompt only to the ones that take one."""
    try:
        return fn(generated, reference, prompt)
    except TypeError:
        return fn(generated, reference)


def reward_with_rationale(task, generated, reference, rationale_weight=0.3,
                          prompt=None):
    """Score the answer, and separately score the reasoning that produced it.

    The rationale in this dataset is not commentary: it states the radar
    evidence -- "18 returns on track #489 at 16 m, median radial velocity
    +0.8 m/s" -- which is computed from the labels and therefore checkable. A
    reward on the answer alone pays equally for a right number reached by
    reciting a prior and one reached by reading the scan, and this project has
    spent a long time establishing that the model does the former. Paying for
    the evidence is the point.

    Format is not rewarded on its own. A model that emits the JSON and fills it
    with nothing scores zero on both halves, so there is nothing to game.
    """
    base = reward_for(task) or reward_description
    got_rationale, got_answer = split_rationale(generated)
    want_rationale, want_answer = split_rationale(reference)
    answer = _call_reward(base, got_answer, want_answer, prompt)
    if want_rationale is None:
        return float(answer)
    if got_rationale is None:
        # The reference reasons and the generation did not: the answer still
        # counts, the evidence half is simply unearned.
        return float((1.0 - rationale_weight) * answer)
    evidence = reward_description(got_rationale, want_rationale)
    return float((1.0 - rationale_weight) * answer + rationale_weight * evidence)


for _task in ("desc_radar", "desc_objects", "desc_ego_maneuver",
              "desc_complementarity", "desc_clip_summary", "description"):
    REWARDS[_task] = reward_description


def _cot_reward(base):
    """A `_cot` task is its base task plus an account of the evidence.

    Registered by closure over the base name rather than mapped to
    `reward_quantity`, which is what `agent_traj_cot` and `motion_seg_cot` were
    doing: that graded the first number of a JSON blob whose first number lives
    inside the rationale, so a right answer with an invented rationale and a
    wrong answer with a copied one were scored the same.
    """
    def reward(generated, reference, prompt=None):
        return reward_with_rationale(base, generated, reference, prompt=prompt)
    reward.__name__ = f"reward_with_rationale[{base}]"
    return reward


for _base in ("det_objects_azdeg", "det_objects_3dbbox", "plan_ego_xy",
              "plan_ego_control", "agent_traj_azdeg", "agent_traj_bbox",
              "agent_traj_xy", "motion_seg_azdeg", "motion_seg_bbox",
              "qa", "track_step_azdeg", "track_step_bbox"):
    REWARDS[f"{_base}_cot"] = _cot_reward(_base)


# --------------------------------------------------------------------------
# tracking
# --------------------------------------------------------------------------
#
# Identity is scored as consistency, not equality. If the model calls the first
# car #2 where the label calls it #1, that is not an error: what matters is that
# the same physical object keeps the same number from one instant to the next.
# So predicted ids are mapped onto label ids by the assignment that explains the
# most matched detections, and everything is counted after that mapping -- which
# is what IDF1 measures in the tracking literature.

PREV_BLOCK = re.compile(r"t-(\d+)s:\s*(.*)")


def prompt_history(prompt):
    """{frames back: [objects]} from the `Previous detections:` block."""
    out = {}
    for line in (prompt or "").split("\n"):
        m = PREV_BLOCK.match(line.strip())
        if m:
            out[int(m.group(1))] = parse_objects(m.group(2))
    return out


def score_tracking(generated, reference, prompt=None):
    """Detection quality at this instant, plus whether ids were carried forward.

    The id an object should get is fixed by the history in the prompt, not by
    the label, so this is checkable per item -- during training, where the
    history is the truth, and at evaluation, where it is the model's own output
    from the previous step.
    """
    predicted, truth = parse_objects(generated), parse_objects(reference)
    pairs = _match(predicted, truth)
    out = {"n": 1, "tp": len(pairs), "fp": len(predicted) - len(pairs),
           "fn": len(truth) - len(pairs), "matched": len(pairs),
           "class_ok": sum(p["cls"] == t["cls"] for p, t in pairs),
           "id_carried": 0, "id_checkable": 0}
    if prompt is None:
        return out
    # An object is "carried" when it appeared in the history and the answer gave
    # it the same id. Objects that are genuinely new cannot be checked this way,
    # so they are left out of the denominator rather than counted as failures.
    history = prompt_history(prompt)
    if not history:
        return out
    recent = history[min(history)]
    for p, t in pairs:
        before = [h for h in recent if h["tid"] == t["tid"]]
        if not before:
            continue
        out["id_checkable"] += 1
        out["id_carried"] += p["tid"] == t["tid"]
    return out


def reward_tracking(generated, reference, prompt=None):
    """Half for finding the objects, half for keeping their identities."""
    s = score_tracking(generated, reference, prompt)
    detection = _f1(s["tp"], s["fp"], s["fn"])
    if not s["matched"]:
        return 0.0
    named = s["class_ok"] / s["matched"]
    if s["id_checkable"]:
        identity = s["id_carried"] / s["id_checkable"]
        return float(0.5 * detection + 0.25 * detection * named
                     + 0.25 * detection * identity)
    return float(0.5 * detection + 0.5 * detection * named)


def idf1(sequence):
    """IDF1 over a rollout: [(predicted, truth)] one pair per instant.

    The predicted-to-label id mapping is solved once for the whole sequence, so
    a model that renames everything consistently scores as well as one that
    happened to guess the label's numbering.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    co, p_total, t_total = {}, {}, {}
    for generated, reference in sequence:
        predicted, truth = parse_objects(generated), parse_objects(reference)
        for o in predicted:
            p_total[o["tid"]] = p_total.get(o["tid"], 0) + 1
        for o in truth:
            t_total[o["tid"]] = t_total.get(o["tid"], 0) + 1
        for p, t in _match(predicted, truth):
            co[(p["tid"], t["tid"])] = co.get((p["tid"], t["tid"]), 0) + 1
    if not co:
        return {"idf1": 0.0, "id_tp": 0, "id_fp": sum(p_total.values()),
                "id_fn": sum(t_total.values())}
    pids = sorted({k[0] for k in co})
    tids = sorted({k[1] for k in co})
    cost = np.zeros((len(pids), len(tids)))
    for (a, b), n in co.items():
        cost[pids.index(a), tids.index(b)] = -n
    rows, cols = linear_sum_assignment(cost)
    tp = int(-cost[rows, cols].sum())
    fp = sum(p_total.values()) - tp
    fn = sum(t_total.values()) - tp
    denom = 2 * tp + fp + fn
    return {"idf1": (2.0 * tp / denom) if denom else 0.0,
            "id_tp": tp, "id_fp": fp, "id_fn": fn}


for _form in ("azdeg", "bbox"):
    SCORERS[f"track_step_{_form}"] = score_tracking
    REWARDS[f"track_step_{_form}"] = reward_tracking


def cot_scorer(plain):
    """Score a CoT generation on its answer, with the reason kept alongside.

    Without this every `_cot` task fell through to `score_text`, which compares
    strings -- so the eleven tasks the redefinition exists for were graded on
    character overlap with a JSON blob, and a model that reasoned perfectly and
    a model that copied the prompt would land near the same number.

    The answer is what the plain twin's scorer already knows how to grade. What
    is added is whether the format held at all, since a rationale the parser
    cannot find is a rationale the reward cannot see either.
    """
    def score(generated, reference, prompt=None):
        _, got = split_rationale(generated)
        _, want = split_rationale(reference)
        try:
            out = plain(got, want, prompt)
        except TypeError:
            out = plain(got, want)
        out = dict(out)
        out["cot_parsed"] = float(RATIONALE_JSON.search(generated or "")
                                  is not None)
        out["n"] = out.get("n", 1)
        return out
    score.__name__ = f"cot_{getattr(plain, '__name__', 'score')}"
    # `summarise` picks its branch with `fn is score_objects`, which a closure
    # can never satisfy. Carrying the wrapped scorer lets it unwrap and fold
    # the numbers the same way the plain task's are folded.
    score.plain = plain
    return score


# Every `_cot` task borrows its plain twin's scorer. `qa_cot` is included: its
# answer is still a single letter, and the letter is what the choice scorer
# grades.
for _plain in ("det_objects_azdeg", "det_objects_3dbbox", "track_step_azdeg",
               "track_step_bbox", "plan_ego_xy", "plan_ego_control",
               "agent_traj_azdeg", "agent_traj_bbox", "agent_traj_xy",
               "motion_seg_azdeg", "motion_seg_bbox", "qa"):
    SCORERS[f"{_plain}_cot"] = cot_scorer(SCORERS[_plain])


# nuScenes 식 mAP. 중심거리 임계 네 개(0.5, 1, 2, 4 m)에서 평균한다.
MAP_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


def detection_map(items, thresholds=MAP_THRESHOLDS):
    """{class: AP} 와 전체 mAP. `items` 는 (예측, 정답, 신뢰도) 의 목록.

    F1 은 탐욕적 디코딩이 우연히 멈춘 한 지점의 값이라, 문턱을 낮추면 어떻게
    되는지 말해 주지 않는다. mAP 는 검출을 확신 순으로 정렬해 정밀도-재현율
    곡선을 그리므로, 그 정렬이 있어야 계산된다 -- 우리 출력에는 점수가 없었고,
    그래서 지금까지 F1 하나만 낼 수 있었다.

    임계는 nuScenes 를 따라 중심거리로 잰다. IoU 가 아닌 이유는 그쪽이
    상자 크기 오차와 위치 오차를 섞기 때문이고, 우리 크기 예측은 클래스
    평균에 가까워서 섞으면 위치가 보이지 않는다.
    """
    per_class = {}
    for cls in {o["cls"] for _, truth, _ in items for o in truth}:
        aps = []
        for th in thresholds:
            rows, n_truth = [], 0
            for pred, truth, conf in items:
                p = [o for o in pred if o["cls"] == cls]
                t = [o for o in truth if o["cls"] == cls]
                n_truth += len(t)
                taken = set()
                scored = sorted(
                    zip(p, list(conf) + [0.0] * len(p)),
                    key=lambda kv: -kv[1])
                for o, c in scored:
                    best, hit = th, None
                    for j, q in enumerate(t):
                        if j in taken:
                            continue
                        d = math.hypot(*(a - b for a, b in zip(_xy(o), _xy(q))))
                        if d < best:
                            best, hit = d, j
                    if hit is not None:
                        taken.add(hit)
                    rows.append((c, hit is not None))
            if not n_truth:
                continue
            rows.sort(key=lambda kv: -kv[0])
            tp = fp = 0
            recalls, precisions = [], []
            for _, ok in rows:
                tp, fp = tp + bool(ok), fp + (not ok)
                recalls.append(tp / n_truth)
                precisions.append(tp / (tp + fp))
            # 재현율을 따라 정밀도의 최대값을 뒤에서부터 채운 뒤 사다리꼴 적분
            best = 0.0
            for i in range(len(precisions) - 1, -1, -1):
                best = max(best, precisions[i])
                precisions[i] = best
            ap, last = 0.0, 0.0
            for r, pr in zip(recalls, precisions):
                ap += (r - last) * pr
                last = r
            aps.append(ap)
        if aps:
            per_class[cls] = sum(aps) / len(aps)
    return {"per_class": per_class,
            "mAP": sum(per_class.values()) / len(per_class) if per_class else None}
