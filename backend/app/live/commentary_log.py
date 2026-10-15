"""Persistent per-delivery commentary log, for detailed post-match review.

User-requested 2026-09-18, directly motivated by how today's real bugs
were actually found: the multi-run extras bug (docs/
finding_cricbuzz_event_types_2026-09-18.md) and the "caught"/"bowled"
classification false-alarm were both only caught by reading a delivery's
FULL raw commentary text carefully, not the short tagged-event lines the
live drivers print to console (app/live/cricbuzz_session.py's
_delivery_event_lines only emits a line for deliveries with a
boundary/extra/wicket tag, and even those are truncated by a human
skimming console output). This writes every delivery -- tagged or not --
to an append-only JSONL file, so a real post-match review (or a future
debugging session) can grep/scan the full text for parsing mismatches
across every ball, not just the ones that happened to trip a known tag.

Source-agnostic: works on the shared LiveSnapshot/LiveDelivery contract
(app/live/live_snapshot.py), so both the Cricbuzz session and the TOI
backup polling loop in run_live_auto.py can use the same function.

Also logs each delivery's app/live/shot_quality.py classification
(shot_quality/near_wicket) -- signals visible only in the prose, not in
the structured total_runs/wicket_kind fields already logged. See that
module's docstring for why this matters and why it can't be trained
into the live models yet: this is the raw material for the accumulation
process that will eventually make it testable.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.live.delivery_detail import extract_delivery_detail
from app.live.shot_quality import classify_conditions, classify_near_wicket, classify_shot_quality


def append_commentary_log(
    path: Path,
    match_id: str,
    snapshot,
    logged_keys: set[tuple[int, int, int]],
) -> None:
    """Append every delivery in `snapshot` not already in `logged_keys`
    as one JSON line each, then mark it logged. Never raises -- a logging
    failure must never interrupt the actual live prediction flow.

    A fresh (empty) `logged_keys` is seeded from any existing file first,
    so a driver restart or crash mid-match doesn't re-log the balls still
    in the feed's rolling window (seen 2026-09-19 dry runs)."""
    if not logged_keys and path.exists():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                logged_keys.add((row["innings"], row["over"], row["ball"]))
        except (OSError, ValueError, KeyError):
            pass
    new_lines: list[str] = []
    for delivery in snapshot.deliveries:
        if delivery.key in logged_keys:
            continue
        logged_keys.add(delivery.key)
        new_lines.append(
            json.dumps(
                {
                    "match_id": match_id,
                    "innings": delivery.innings,
                    "over": delivery.over,
                    "ball": delivery.ball,
                    "bowler": delivery.bowler,
                    "striker": delivery.striker,
                    "total_runs": delivery.total_runs,
                    "batter_runs": delivery.batter_runs,
                    "extras": delivery.extras,
                    "wicket_kind": delivery.wicket_kind,
                    "shot_quality": classify_shot_quality(delivery.commentary),
                    "near_wicket": classify_near_wicket(delivery.commentary),
                    "conditions": classify_conditions(delivery.commentary),
                    "detail": extract_delivery_detail(delivery.commentary),
                    "commentary": delivery.commentary,
                }
            )
        )
    if not new_lines:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for line in new_lines:
                handle.write(line + "\n")
    except OSError:
        pass
