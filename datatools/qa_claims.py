"""Pull numeric claims out of a QA rationale, sentence by sentence.

The first extractor searched the whole rationale at once and paired every number
with the first frame number mentioned. Two things went wrong.

It attributed positions to the wrong entity. `ego[^.]{0,80}?pos=` will happily
span an intervening clause, so "the ego's predicted path ... the pedestrian is at
pos=(36,3)" matched the pedestrian's coordinates as the ego's.

It used one timestamp for the whole text. Rationales walk through several
frames, and a claim in the third sentence often refers to a later frame than the
first sentence did. Re-evaluating the 622 disagreeing ego-position claims at
shifted frame offsets dropped the median error from 17.96 m to 2.26 m, which says
most of those "errors" were timing, not wrong numbers.

So: split into sentences, carry the frame forward from whichever sentence last
named one, and only accept a claim when the subject sits immediately in front of
the number with no other road user in between. Ambiguous constructions are
skipped rather than guessed at -- a claim wrongly attributed then "corrected"
would write a bad number into the data.
"""

import re

# Road users that could own a position or a speed. "ego vehicle" is stripped out
# before this list is consulted, so `vehicle` here means some other vehicle.
OTHER_ENTITY = re.compile(
    r"\b(sedan|car|cars|truck|suv|van|bus|pickup|trailer|motorcycle|motorbike|"
    r"bicycle|bike|cyclist|rider|pedestrian|person|people|child|animal|dog|"
    r"scooter|vehicle|vehicles|object)\b", re.I)

EGO = re.compile(r"\bego(?:[- ]vehicle)?(?:'s)?\b", re.I)
FRAME = re.compile(r"[Ff]rames?\s+(\d+)")

# The leading minus must not be preceded by a digit or dot. Without the
# lookbehind, "a speed around 17.6-17.9 m/s" captured "-17.9": the hyphen of a
# numeric range read as a sign. Substituting that span then dropped the hyphen
# and fused the two numbers into "17.617.9".
NUM = r"((?<![\d.])-?\d+(?:\.\d+)?)"
POS = rf"\(\s*{NUM}\s*,\s*{NUM}(?:\s*,\s*{NUM})?\s*\)"

# Tight, subject-adjacent forms. The gap allowed between subject and value is
# short and must not contain another road user.
#
# The gap class is `[^;]`, not `[^.;]`: excluding the period also excluded
# decimal points, so "the ego is at pos=(238.88,0.09) with a speed of 7.57 m/s"
# never matched the speed -- the intervening coordinates contain dots. Sentences
# are already split on `[.!?]\s+`, so a period still inside one is a decimal.
P_EGO_POS = re.compile(rf"{EGO.pattern}[^;]{{0,40}}?"
                       rf"pos(?:ition)?\s*(?:=|is|of|at)?\s*{POS}", re.I)
P_EGO_SPEED = re.compile(rf"{EGO.pattern}[^;]{{0,60}}?"
                         rf"(?:speed|velocity|travelling at|moving at)"
                         rf"[^;]{{0,15}}?{NUM}\s*m/s", re.I)
P_FUTURE = re.compile(rf"(?:predicted|projected|will be|expected)[^;]{{0,40}}?"
                      rf"t\s*=\s*{NUM}\s*s[^(;]{{0,40}}{POS}", re.I)
P_ANY_POS = re.compile(rf"pos(?:ition)?\s*(?:=|is|of|at)?\s*{POS}", re.I)
# Only explicit distance phrasings. A bare "is X m" also matched the "m" in
# "speed is 5.74 m/s", which inflated distance claims to 18,391 -- more than
# every other kind combined.
P_DISTANCE = re.compile(
    rf"(?:at\s+a\s+distance\s+of|distance\s+of|is)\s+{NUM}\s*m(?!/)\b"
    rf"(?=\s*(?:away|from|ahead|behind|in front|$|[.,;]))", re.I)

# "17.6-17.9", "63-64 km/h": a value expressed as a range. Rewriting one end of
# one is never right, so the corrector treats these as unsafe.
RANGE = re.compile(r"\d(?:\.\d+)?\s*(?:-|–|to)\s*\d")

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    """Sentences with their start offset in the original string."""
    out = []
    cursor = 0
    for piece in SENTENCE_SPLIT.split(text):
        index = text.find(piece, cursor)
        if index < 0:
            index = cursor
        out.append((piece, index))
        cursor = index + len(piece)
    return out


def _no_other_entity_between(sentence, subject_end, value_start):
    gap = sentence[subject_end:value_start]
    gap = EGO.sub(" ", gap)          # "ego vehicle" must not count as an entity
    return OTHER_ENTITY.search(gap) is None


def extract(rationale):
    """Claims found in a rationale.

    Each claim is a dict with `kind`, `frame` (or None if the text never named
    one), and the stated value(s).
    """
    claims = []
    current_frame = None
    for sentence, base in split_sentences(rationale):
        frames = FRAME.findall(sentence)
        if frames:
            current_frame = int(frames[0])
        frame = current_frame

        future_spans = []
        for m in P_FUTURE.finditer(sentence):
            claims.append({"kind": "future_pos", "frame": frame,
                           "horizon_s": float(m.group(1)),
                           "x": float(m.group(2)), "y": float(m.group(3)),
                           "spans": [(base + m.start(2), base + m.end(2)),
                                     (base + m.start(3), base + m.end(3))],
                           "sentence": sentence})
            future_spans.append(m.span())

        def value_is_future(value_at):
            """Judge by where the number sits, not by span containment.

            A future match starts at "predicted", while the ego-position match
            starts earlier at "ego", so neither span contains the other and a
            containment test lets the predicted coordinates through as if they
            were the current pose.
            """
            return any(a <= value_at < b for a, b in future_spans)

        for m in P_EGO_POS.finditer(sentence):
            ego = EGO.search(m.group(0))
            # Locate the value by its capture group, not by the first "(" in the
            # match. "the ego's velocity is (17.94, 1.53) m/s and its position is
            # (90.22, 1.35)" puts the velocity parenthesis first, so find("(")
            # pointed at the wrong tuple; the resulting offset mismatch let the
            # same coordinates through as an agent position as well, and the
            # corrector then overwrote a correct ego position with a nearby
            # object's coordinates.
            value_at = m.start(1)
            if value_is_future(value_at):
                continue
            if not _no_other_entity_between(sentence, m.start() + ego.end(), value_at):
                continue
            claims.append({"kind": "ego_pos", "frame": frame,
                           "x": float(m.group(1)), "y": float(m.group(2)),
                           "spans": [(base + m.start(1), base + m.end(1)),
                                     (base + m.start(2), base + m.end(2))],
                           "sentence": sentence})

        for m in P_EGO_SPEED.finditer(sentence):
            ego = EGO.search(m.group(0))
            if not _no_other_entity_between(sentence, m.start() + ego.end(),
                                            m.start(1)):
                continue
            claims.append({"kind": "ego_speed", "frame": frame,
                           "value": float(m.group(1)),
                           "spans": [(base + m.start(1), base + m.end(1))],
                           "sentence": sentence})

        # Positions that are not the ego's and not a prediction belong to some
        # other road user; identity is not recoverable from prose, so these are
        # only checked for "is anything labelled here".
        ego_value_at = {m.start(1) for m in P_EGO_POS.finditer(sentence)}
        for m in P_ANY_POS.finditer(sentence):
            value_at = m.start(1)
            if value_is_future(value_at) or value_at in ego_value_at:
                continue
            claims.append({"kind": "agent_pos", "frame": frame,
                           "x": float(m.group(1)), "y": float(m.group(2)),
                           "spans": [(base + m.start(1), base + m.end(1)),
                                     (base + m.start(2), base + m.end(2))],
                           "sentence": sentence})

        for m in P_DISTANCE.finditer(sentence):
            claims.append({"kind": "distance", "frame": frame,
                           "value": float(m.group(1)),
                           "spans": [(base + m.start(1), base + m.end(1))],
                           "sentence": sentence})

    return claims
