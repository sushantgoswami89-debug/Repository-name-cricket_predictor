"""Run the live prediction pipeline against a single Cricbuzz match.

Fallback path for 2026-09-15's field test: TOI never published tonight's
Afghanistan vs India 2nd T20I at all (see
docs/live_field_test_2026-09-15.md), so this points the same
VerifiedLivePredictionPipeline at app/live/cricbuzz_reader.py's
CricbuzzLiveReader instead of ToiLiveReader. Single-match only, no
discovery -- Cricbuzz's own match discovery isn't built yet, so the URL
is passed explicitly.

Usage:
    /usr/bin/python3 run_cricbuzz_match.py "<cricbuzz live-cricket-scores URL>" [--shadow]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from app.live.cricbuzz_reader import CricbuzzFeedError, CricbuzzLiveReader
from app.live.pipeline import VerifiedLivePredictionPipeline
from app.live.telegram import TelegramPublisher
from app.live.verification import VerificationError


def _telegram(project_root: Path) -> TelegramPublisher:
    path = project_root / "backend" / ".telegram.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    return TelegramPublisher(config["bot_token"], config["chat_id"])


def _match_result_text(snapshot) -> str | None:
    """None while the match is still live. A 2nd-innings snapshot is only
    ever handed to us once the chase has genuinely finished one way or
    the other (innings_complete, or target reached) -- see
    docs/live_field_test_2026-09-15.md item 17: this driver had no
    completion detection at all before, so it never announced the result
    or exited, just looped forever on the same frozen final scorecard."""
    if snapshot.innings < 2:
        return None
    target_reached = snapshot.target > 0 and snapshot.score >= snapshot.target
    if target_reached:
        wickets_in_hand = 10 - snapshot.wickets
        return (
            f"{snapshot.batting_team} won by {wickets_in_hand} wicket"
            f"{'s' if wickets_in_hand != 1 else ''} "
            f"({snapshot.score}/{snapshot.wickets})"
        )
    if snapshot.innings_complete:
        margin = snapshot.target - 1 - snapshot.score
        return (
            f"{snapshot.bowling_team} won by {margin} run"
            f"{'s' if margin != 1 else ''} "
            f"({snapshot.batting_team} {snapshot.score}/{snapshot.wickets})"
        )
    return None


_EXTRA_LABELS = {"wides": "WIDE", "noballs": "NO-BALL", "byes": "BYE", "legbyes": "LEG-BYE"}


def _delivery_event_lines(snapshot, seen_keys: set[tuple[int, int, int]]) -> list[str]:
    """Surface every new delivery's extras/wicket kind explicitly.

    docs/live_field_test_2026-09-15.md item 5 (still open): only
    four/wicket event tags were ever confirmed against real Cricbuzz
    data; six/wide/noball/bye/legbye and every wicket kind except
    "caught" were unvalidated. This makes each one visible the moment it
    happens live instead of requiring a manual archived-match audit.
    """
    lines = []
    for delivery in snapshot.deliveries:
        if delivery.key in seen_keys:
            continue
        seen_keys.add(delivery.key)
        tags = []
        if delivery.batter_runs == 6:
            tags.append("SIX")
        for extra_key, label in _EXTRA_LABELS.items():
            if extra_key in delivery.extras:
                tags.append(label)
        if delivery.wicket_kind:
            tags.append(f"WICKET({delivery.wicket_kind})")
        if tags:
            lines.append(
                f"[event] inn{delivery.innings} {delivery.over}.{delivery.ball} "
                f"{'/'.join(tags)} -- {delivery.striker} off {delivery.bowler}"
            )
    return lines


def _scorecard_line(snapshot) -> str:
    deliveries = snapshot.deliveries
    latest = deliveries[-1] if deliveries else None
    # snapshot.deliveries only covers balls seen since this reader
    # attached (Cricbuzz's rolling window, not full innings history --
    # see docs/live_field_test_2026-09-15.md item 8), so legal-ball count
    # for run rate must come from the innings-wide snapshot.overs figure,
    # not from counting the (partial) delivery list.
    overs_completed, _, balls_into_over = str(snapshot.overs).partition(".")
    legal_balls = int(overs_completed or 0) * 6 + int(balls_into_over or 0)
    run_rate = (snapshot.score * 6 / legal_balls) if legal_balls else 0.0
    batter = latest.striker if latest else "?"
    bowler = latest.bowler if latest else "?"
    return (
        f"[scorecard] {snapshot.batting_team} {snapshot.score}/{snapshot.wickets} "
        f"({snapshot.overs} ov, RR {run_rate:.2f}) -- {bowler} to {batter}"
    )


def _progress(snapshot) -> tuple[int, int]:
    overs_completed, _, balls_into_over = str(snapshot.overs).partition(".")
    legal_balls = int(overs_completed or 0) * 6 + int(balls_into_over or 0)
    return snapshot.innings, legal_balls


def _fetch_freshest(reader: CricbuzzLiveReader, match_url: str, attempts: int = 3):
    """Try the fetch a few times within one poll and keep the best.

    docs/live_field_test_2026-09-17.md: confirmed live tonight that three
    back-to-back fresh requests for the same match can return three
    different snapshots from Cricbuzz (one genuinely current, two stale)
    -- not a client-side caching artifact (no-cache headers and a
    cache-busting query param made no difference), a real upstream
    inconsistency. One extra fetch or two per poll meaningfully raises
    the odds of landing on the current one instead of waiting a full
    --poll-seconds cycle to try again.
    """
    best_snapshot = None
    best_rank = None
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            snapshot = reader.fetch(match_url)
        except (CricbuzzFeedError, requests.exceptions.RequestException) as error:
            last_error = error
            continue
        rank = _progress(snapshot)
        if best_rank is None or rank > best_rank:
            best_snapshot, best_rank = snapshot, rank
    return best_snapshot, last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("match_url", help="Cricbuzz live-cricket-scores URL")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--shadow", action="store_true", help="Do not publish to Telegram."
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    reader = CricbuzzLiveReader()
    publisher = None if args.shadow else _telegram(project_root)
    match_id = reader.match_id_from_url(args.match_url)
    output_file = project_root / "data" / "live" / f"cricbuzz-{match_id}.json"
    pipeline = VerifiedLivePredictionPipeline(publisher=publisher, output_file=output_file)

    mode = "shadow (no Telegram)" if args.shadow else "LIVE (publishing to Telegram)"
    print(f"CricketBaba watching Cricbuzz match {match_id} -- mode: {mode}")

    seen_delivery_keys: set[tuple[int, int, int]] = set()
    last_over_sent: dict[int, int] = {}
    # docs/live_field_test_2026-09-17.md: an over-checkpoint mismatch can
    # get permanently stuck -- the exception fires before this poll's
    # progress is saved, so the next poll re-derives the identical (or,
    # seen live tonight, a *different* but still wrong) total from the
    # same unstable upstream window, forever, with zero predictions sent.
    # Three occurrences in one match, each only recovered by a full
    # process restart, which works because a fresh CricbuzzLiveReader()
    # reseeds its running total from Cricbuzz's own latest checkpoint
    # instead of replaying the bad accumulation. Do that reseed
    # automatically instead of requiring a human to notice and restart.
    consecutive_feed_errors = 0
    RESEED_AFTER_CONSECUTIVE_ERRORS = 3
    # docs/live_field_test_2026-09-17.md: confirmed directly tonight that
    # Cricbuzz's feed is NOT monotonic poll-to-poll -- one successful
    # fetch returned real innings-2 progress (a wicket at over 6.2), and
    # the very next successful fetch (no error at all) regressed straight
    # back to innings 1's stale final scorecard. A cache-busting query
    # param and no-cache headers on the reader's own requests (already
    # tried) did not fix this -- it's the upstream feed itself
    # disagreeing with itself between requests, not a client-side caching
    # problem. Defend at this layer instead: never let a snapshot that's
    # behind the furthest point already seen reach the scorecard print or
    # the prediction pipeline.
    best_progress: tuple[int, int] | None = None

    while True:
        snapshot, error = _fetch_freshest(reader, args.match_url)
        if snapshot is None:
            label = "Deferred" if isinstance(error, CricbuzzFeedError) else "Deferred (network)"
            print(f"{label}: {error}")
            consecutive_feed_errors += 1
            if consecutive_feed_errors >= RESEED_AFTER_CONSECUTIVE_ERRORS:
                print(
                    f"Reseeding Cricbuzz reader after {consecutive_feed_errors} "
                    "consecutive feed errors."
                )
                reader = CricbuzzLiveReader()
                consecutive_feed_errors = 0
            time.sleep(max(1.0, args.poll_seconds))
            continue

        consecutive_feed_errors = 0

        progress = _progress(snapshot)
        if best_progress is not None and progress < best_progress:
            print(
                f"Stale snapshot skipped: innings {snapshot.innings} at "
                f"{snapshot.overs} ov is behind the innings {best_progress[0]} / "
                f"{best_progress[1]} legal balls already seen."
            )
            time.sleep(max(1.0, args.poll_seconds))
            continue
        best_progress = progress if best_progress is None else max(best_progress, progress)

        print(_scorecard_line(snapshot))
        for line in _delivery_event_lines(snapshot, seen_delivery_keys):
            print(line)

        try:
            output = pipeline.process(snapshot)
        except VerificationError as error:
            print(f"Verification error: {error}")
            time.sleep(max(1.0, args.poll_seconds))
            continue

        if output:
            print(
                f"{'Sent' if publisher else 'Recorded'} innings {output['innings']} "
                f"over {output['over']} prediction."
            )
            # docs/live_field_test_2026-09-15.md item 3 (unresolved): over 14
            # was silently skipped once with no prediction and no error
            # logged anywhere. Flag any gap in the served-over sequence the
            # moment it happens, rather than discovering it after the fact.
            innings = output["innings"]
            over = output["over"]
            prior = last_over_sent.get(innings)
            if prior is not None and over > prior + 1:
                print(
                    f"OVER GAP DETECTED: innings {innings} jumped from over "
                    f"{prior} to {over} -- over(s) {prior + 1}-{over - 1} never "
                    f"got a prediction."
                )
            last_over_sent[innings] = over

        result = _match_result_text(snapshot)
        if result:
            print(f"MATCH COMPLETE: {result}")
            if publisher is not None:
                publisher.publish(f"result:{match_id}", f"🏆 {result}")
            break

        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCricketBaba stopped.")
