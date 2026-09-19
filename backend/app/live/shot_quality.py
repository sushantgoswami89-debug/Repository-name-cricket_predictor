"""Detects cricket signals visible ONLY in commentary prose -- the exact
gap a user example named directly, 2026-09-18: an edge that races away
for four is recorded by every structured feature in this codebase as
"4 runs, no wicket", indistinguishable from a well-middled boundary, and
if anything it *lowers* the model's perceived wicket risk (recent
scoring looks fine) when the real signal is the opposite: the batter
mistimed it, was not in control, and is at genuinely elevated risk on
the very next ball.

These are real, named concepts in professional cricket analysis
(ESPNcricinfo's own "false shot"/control percentage stat exists for
exactly this reason; a dropped catch or a survived close shout is
standard commentary vocabulary every broadcast tracks) that have never
been available to this project before, because they can only be read
from PROSE commentary -- and Cricsheet, the entire 6,767-match
historical training corpus, has zero prose commentary at all (confirmed
2026-09-18: every delivery there is only
{batter, bowler, non_striker, runs}). That means none of this can be
trained into run-range/wicket the way every other feature in this
project has been (build a candidate, retrain on Cricsheet, evaluate on
real holdout) -- there is no historical ground truth to correlate it
against at that scale. It can only start as a live-only annotation,
logged alongside real outcomes match by match, until enough real
evidence accumulates to say something reliable (same accumulation
problem already blocking the live ensemble shadows' promotion).

Confidence per pattern is marked explicitly -- some are built from real
commentary text confirmed live 2026-09-18 (searching several real
completed matches' cached windows, same verification convention
app/live/cricbuzz_reader.py already uses), others from standard,
unambiguous cricket-broadcast vocabulary not yet spot-checked against a
real live example of that specific case. Treat the latter as "should be
right, confirm on the next live match," not yet fully verified -- same
honesty standard as everything else logged as unconfirmed in
docs/next_live_field_test_checklist.md.
"""

from __future__ import annotations

import re

# Confirmed real 2026-09-18: "gets an outside edge in the air towards
# point" (a caught dismissal), "chops it off the inside edge into the
# on-side" (an edge that went for extra runs off a no-ball, NOT a
# dismissal -- exactly the shape this module exists for).
_EDGE_RE = re.compile(
    r"\b(gets?|got|takes?|off|via)\s+(?:an?\s+)?"
    r"(outside|inside|thick|thin|faint|top|leading|bottom|under)?[\s-]*edge\b"
    r"|\bedged?\b|\b(?:top|leading|inside|outside|thick|thin|bottom)[\s-]edge[sd]?\b",
    re.IGNORECASE,
)
# Confirmed real 2026-09-19 (Sri Lanka v England, 3rd T20I): "tries to go
# downtown but gets it off the inner half towards long-on" -- contact,
# but not the middle of the bat. Weaker than an edge (ball off the bat,
# not the edge), stronger evidence of a batter forcing the shot than a
# clean stroke.
_MISTIMED_RE = re.compile(
    r"\b(?:inner|outer)\s+half\b|\bmis-?timed?\b|\bmis-?hit\b|\btoe-?end(?:ed|s)?\b"
    r"|\bskied\b|\bballooned?\b|\bcloses?\s+the\s+face\b",
    re.IGNORECASE,
)
# Confirmed real 2026-09-18: "connects only thin air and the ball goes
# past the outside edge" -- a genuine beat, no contact at all, distinct
# from an edge (real contact). Checked before _EDGE_RE so a beat
# mentioning "past the edge" is never misread as an edge.
_BEATEN_RE = re.compile(
    r"\b(goes?|went|flies?|flew)\s+past\s+the\s+(?:outside|inside)?\s*edge\b"
    r"|\bbeats?\s+(?:the\s+)?bat\b"
    r"|\bplay(?:ed|s)?\s+and\s+miss(?:ed|es)?\b"
    r"|\bconnects?\s+only\s+thin\s+air\b"
    r"|\bcompletely\s+beaten\b",
    re.IGNORECASE,
)
# Not yet spot-checked against a real live example -- standard broadcast
# vocabulary for clean, controlled contact ("timed to perfection",
# "middled", "creamed"), confirm on the next live match.
_WELL_TIMED_RE = re.compile(
    r"\bmiddled\b|\bcreamed\b|\btimed\s+(?:it\s+|that\s+)?(?:to\s+perfection|beautifully|well)\b"
    r"|\bsweetly\s+timed\b|\bmeaty\s+connection\b",
    re.IGNORECASE,
)

# Not yet spot-checked against a real live example -- standard broadcast
# vocabulary for a dropped catch, confirm on the next live match.
_DROPPED_CATCH_RE = re.compile(
    r"\bdrops?\s+(?:it|him|her|the\s+catch)\b|\bspills?\s+(?:it|the\s+catch)\b"
    r"|\bshould\s+have\s+(?:been\s+)?(?:taken|caught)\b|\bgrasses?\s+(?:it|the\s+catch)\b",
    re.IGNORECASE,
)
# Not yet spot-checked against a real live example -- standard broadcast
# vocabulary for a close shout that didn't result in a wicket (huge
# appeal turned down, a review that fails, umpire's call), confirm on
# the next live match.
_SURVIVED_CLOSE_CALL_RE = re.compile(
    r"\b(?:huge|big|loud)\s+appeal\b.*?\bnot\s+out\b"
    r"|\bnot\s+out\b.*?\b(?:huge|big|loud)\s+appeal\b"
    r"|\bumpire.?s\s+call\b|\breview\s+(?:fails|turned\s+down|is\s+unsuccessful)\b"
    r"|\bsurvives?\s+(?:the\s+)?(?:review|appeal|close\s+call)\b",
    re.IGNORECASE,
)


def classify_shot_quality(commentary: str) -> str | None:
    """Returns "edge", "beaten", "mistimed", "well_timed", or None (no signal
    detected -- most deliveries; not the same as "clean shot"). "beaten"
    is checked before "edge" since a clear beat sometimes mentions
    "edge" only in the "went past the edge" sense (no contact)."""
    if not commentary:
        return None
    if _BEATEN_RE.search(commentary):
        return "beaten"
    if _EDGE_RE.search(commentary):
        return "edge"
    if _MISTIMED_RE.search(commentary):
        return "mistimed"
    if _WELL_TIMED_RE.search(commentary):
        return "well_timed"
    return None


def classify_near_wicket(commentary: str) -> str | None:
    """Returns "dropped_catch", "survived_close_call", or None. Distinct
    from classify_shot_quality: this is about the FIELDING/UMPIRING side
    of a near-dismissal (a real wicket chance the scorecard's
    wicket_kind=None makes invisible), not the batter's own shot."""
    if not commentary:
        return None
    if _DROPPED_CATCH_RE.search(commentary):
        return "dropped_catch"
    if _SURVIVED_CLOSE_CALL_RE.search(commentary):
        return "survived_close_call"
    return None


# Pitch/weather remarks -- match-level context, not a shot property.
# "heavy"/"plugging in" confirmed real 2026-09-19 ("a lot of rain early in
# the morning so the outfield is heavy and the ball is plugging in it");
# "seaming in"/"swirling in the wind" confirmed same match. The rest
# (turn, dew, two-paced, sticky) are standard commentary vocabulary,
# not yet seen live.
_CONDITION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("heavy_outfield", re.compile(
        r"\bheavy\s+outfield\b|\boutfield\s+is\s+(?:heavy|slow|wet)\b"
        r"|\bplugg?ing\s+in\b|\bball\s+(?:stops|sticks)\s+in\s+the\s+outfield\b",
        re.IGNORECASE)),
    ("swing_seam", re.compile(
        r"\bseam(?:ing|s|ed)\b|\bswing(?:ing|s)?\b|\bswirl(?:ing|s)?\b"
        r"|\bmoves?\s+(?:away|in)\s+off\s+the\s+(?:seam|pitch)\b",
        re.IGNORECASE)),
    ("turn_grip", re.compile(
        r"\bgrips?\s+and\s+turns?\b|\bturn(?:ing|s)?\s+(?:sharply|square|big)\b"
        r"|\bspinning\s+(?:sharply|square)\b|\bgripped\b",
        re.IGNORECASE)),
    ("dew", re.compile(r"\bdew\b|\bwet\s+ball\b", re.IGNORECASE)),
    ("slow_pitch", re.compile(
        r"\btwo-?paced\b|\bsticky\b|\bslow\s+(?:pitch|surface|wicket)\b|\bstopping\s+on\b",
        re.IGNORECASE)),
)


def classify_conditions(commentary: str) -> list[str]:
    """Returns every pitch/weather condition named in this delivery's
    commentary (possibly empty). A list, not a single label: one line can
    name several ("swing" and "heavy outfield")."""
    if not commentary:
        return []
    return [name for name, pattern in _CONDITION_PATTERNS if pattern.search(commentary)]
