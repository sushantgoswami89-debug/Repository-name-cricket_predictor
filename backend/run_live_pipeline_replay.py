"""Replay a pre-recorded (Cricsheet) match through the REAL live pipeline
(VerifiedLivePredictionPipeline + LiveDeliveryVerifier), not just the raw
PredictionEngine that run_full_match_replay.py exercises.

Why: docs/live_match_test_checklist.md item #3 flags that a full
predict -> verify -> publish cycle has never been observed end-to-end --
every live TOI test so far got stuck on feed reconciliation first. This
builds a Cricsheet-to-ToiSnapshot adapter that feeds the pipeline
ball-by-ball exactly the way TOI would (growing cumulative delivery list,
cumulative score/wickets, cricket-notation overs string), using real
historical deliveries as a stand-in for a live feed. It's not a substitute
for a genuinely live TOI test (item #2's feed-reconciliation-quality
question and item #4's real-Telegram-timing question still need a real
match), but it directly answers: does the verifier accept a full innings
of ball-by-ball deliveries without raising VerificationError, does exactly
one prediction get made per over, and does the predict->verify->score
cycle complete cleanly over a whole match.

Usage:
    .venv/bin/python3 backend/run_live_pipeline_replay.py [path/to/match.json]

Defaults to the same sample match run_full_match_replay.py uses. Uses a
dry-run publisher (no Telegram) and no state file (no disk writes).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.live.pipeline import VerifiedLivePredictionPipeline
from app.live.toi_reader import ToiDelivery, ToiSnapshot

MATCH_PATH = sys.argv[1] if len(sys.argv) > 1 else "../data/raw/cricsheet/t20i/1442989.json"


class DryRunPublisher:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def publish(self, key, text) -> bool:
        self.messages.append((key, text))
        return True


def _is_wicket(delivery: dict) -> bool:
    return bool(delivery.get("wickets"))


def _wicket_kind(delivery: dict) -> str | None:
    wickets = delivery.get("wickets") or []
    return str(wickets[0].get("kind")) if wickets else None


def replay_innings(
    match_id: str,
    match_format: str,
    innings_number: int,
    batting_team: str,
    bowling_team: str,
    scheduled_overs: int,
    target: int,
    overs_data: list[dict],
    pipeline: VerifiedLivePredictionPipeline,
) -> dict:
    deliveries: list[ToiDelivery] = []
    score = 0
    wickets = 0
    outputs = 0
    reviews = []
    errors = []
    timestamp = 0

    for source_over in overs_data:
        over_number = int(source_over["over"]) + 1
        legal_in_over = 0
        for ball_index, delivery in enumerate(source_over.get("deliveries", []), start=1):
            timestamp += 1000
            extras_raw = delivery.get("extras", {}) or {}
            is_legal = "wides" not in extras_raw and "noballs" not in extras_raw
            total_runs = int(delivery.get("runs", {}).get("total", 0))
            batter_runs = int(delivery.get("runs", {}).get("batter", 0))
            score += total_runs
            wickets += int(_is_wicket(delivery))
            if is_legal:
                legal_in_over += 1
            deliveries.append(
                ToiDelivery(
                    match_id=match_id,
                    innings=innings_number,
                    over=over_number,
                    ball=ball_index,
                    bowler=str(delivery.get("bowler", "")),
                    striker=str(delivery.get("batter", "")),
                    total_runs=total_runs,
                    batter_runs=batter_runs,
                    extras={k: int(v) for k, v in extras_raw.items()},
                    wicket_kind=_wicket_kind(delivery),
                    feed_total=score,
                    feed_wickets=wickets,
                    timestamp_ms=timestamp,
                    commentary="",
                )
            )
            prior_full_overs = over_number - 1
            overs_string = f"{prior_full_overs}.{legal_in_over}"
            snapshot = ToiSnapshot(
                match_id=match_id,
                match_format=match_format,
                batting_team=batting_team,
                bowling_team=bowling_team,
                innings=innings_number,
                score=score,
                wickets=wickets,
                overs=overs_string,
                is_live=True,
                deliveries=tuple(deliveries),
                target=target,
                scheduled_overs=scheduled_overs,
            )
            try:
                output = pipeline.process(snapshot)
            except Exception as exc:  # noqa: BLE001 -- deliberately broad for a diagnostic replay
                errors.append(f"{over_number}.{ball_index}: {type(exc).__name__}: {exc}")
                continue
            if output is not None:
                outputs += 1
                review = output.get("previous_over_evaluation")
                if review:
                    reviews.append(review)
                pred = output.get("prediction", {})
                print(
                    f"  over {output.get('over'):>2} published | "
                    f"score {output.get('score_before_over')}/{output.get('wickets_before_over')} | "
                    f"predicted {pred.get('expected_runs', 0):.1f} runs, "
                    f"wkt% {pred.get('wicket_probability', 0)*100:.0f}% | "
                    f"prev-over review: {'range HIT' if review and review.get('range_covered') else 'range miss' if review else '-'} "
                    f"({review.get('stars') if review else '-'}★)"
                )

    # final snapshot marking innings complete
    final_snapshot = ToiSnapshot(
        match_id=match_id,
        match_format=match_format,
        batting_team=batting_team,
        bowling_team=bowling_team,
        innings=innings_number,
        score=score,
        wickets=wickets,
        overs=f"{len(overs_data)}.0",
        is_live=False,
        deliveries=tuple(deliveries),
        target=target,
        scheduled_overs=scheduled_overs,
        innings_complete=True,
    )
    try:
        pipeline.process(final_snapshot)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"final: {type(exc).__name__}: {exc}")

    return {
        "final_score": score,
        "final_wickets": wickets,
        "outputs_published": outputs,
        "reviews_scored": len(reviews),
        "errors": errors,
    }


def main() -> None:
    raw = json.loads(Path(MATCH_PATH).read_text(encoding="utf-8"))
    info = raw["info"]
    teams = info.get("teams", ["Team A", "Team B"])
    match_format = str(info.get("match_type", "T20"))
    scheduled_overs = int(info.get("overs", 20))
    match_id = Path(MATCH_PATH).stem

    print("=" * 72)
    print(f"LIVE PIPELINE REPLAY: {teams[0]} vs {teams[1]} ({match_id})")
    print("=" * 72)

    regular_innings = [i for i in raw.get("innings", []) if not i.get("super_over")]
    first_total = None
    for innings_number, innings in enumerate(regular_innings, start=1):
        batting_team = str(innings.get("team", ""))
        bowling_team = next((t for t in teams if t != batting_team), teams[-1])
        target = first_total + 1 if innings_number == 2 and first_total is not None else 0

        publisher = DryRunPublisher()
        pipeline = VerifiedLivePredictionPipeline(publisher=publisher)

        print(f"\n--- Innings {innings_number}: {batting_team} batting (target={target or 'n/a'}) ---")
        summary = replay_innings(
            match_id, match_format, innings_number, batting_team, bowling_team,
            scheduled_overs, target, innings.get("overs", []), pipeline,
        )
        if innings_number == 1:
            first_total = summary["final_score"]

        print(
            f"\nInnings {innings_number} summary: final {summary['final_score']}/"
            f"{summary['final_wickets']} | {summary['outputs_published']} predictions "
            f"published | {summary['reviews_scored']} prior-over reviews scored | "
            f"{len(publisher.messages)} publisher.publish() calls | "
            f"{len(summary['errors'])} errors"
        )
        if summary["errors"]:
            print("  ERRORS:")
            for error in summary["errors"][:10]:
                print(f"    {error}")


if __name__ == "__main__":
    main()
