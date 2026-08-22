"""Validates runtime_run_range_v3.py against real matches, two ways:

1. No-crash + sanity check across many matches (mirrors
   run_live_pipeline_replay.py's rigor for the wicket pipeline).
2. Aggregate accuracy check: if RunRangeV3FeatureComputer correctly
   reproduces what the offline training script computed, the runtime's
   real-match hit rate on the SAME holdout population (>=2025-01-01,
   male_source_files()) should land close to the offline holdout number
   (28.49%, models/candidates/run_range_enriched_v3_batting_style/validation_report.json).
   A meaningfully lower number would point at a live-feature-computation
   bug; this isn't proof of zero bugs, but a large discrepancy would catch
   most of them.

Uses the SAME live snapshots (data/live/run_range_v3_*.json) the runtime
uses in general -- since those snapshots were built from ALL eligible
matches (including holdout-year ones), this is not a leakage-safe
re-validation of the MODEL's accuracy (that's already been done, with
proper temporal cutoffs, in training). It is specifically a check that the
live computation path produces sane, non-degenerate features and doesn't
silently diverge from what the model expects.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from app.live.toi_reader import ToiDelivery
from runtime_run_range_v3 import RunRangeRuntimeV3
from train_phase_calibrated_sharp_range_v33 import male_source_files


def replay_match(runtime: RunRangeRuntimeV3, path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    info = raw["info"]
    registry = info.get("registry", {}).get("people", {})
    teams = info.get("teams", ["A", "B"])
    venue = str(info.get("venue") or info.get("city") or "Unknown")
    scheduled_overs = int(info.get("overs", 20))

    results = []
    regular_innings = [i for i in raw.get("innings", []) if not i.get("super_over")]
    first_total = None
    for innings_number, innings in enumerate(regular_innings, start=1):
        batting_team = str(innings.get("team", ""))
        target = first_total + 1 if innings_number == 2 and first_total is not None else 0
        deliveries: list[ToiDelivery] = []
        score = wickets = legal_balls = 0
        innings_total = 0

        for over_data in innings.get("overs", []):
            over_number = int(over_data["over"]) + 1
            over_actual_runs = 0
            over_actual_wickets = 0
            for ball_index, delivery in enumerate(over_data.get("deliveries", []), start=1):
                extras = delivery.get("extras", {}) or {}
                is_legal = "wides" not in extras and "noballs" not in extras
                total_runs = int(delivery.get("runs", {}).get("total", 0))
                batter_runs = int(delivery.get("runs", {}).get("batter", 0))
                wicket_kind = None
                for dismissal in delivery.get("wickets", []):
                    if dismissal.get("kind") not in {"retired hurt", "obstructing the field"}:
                        wicket_kind = str(dismissal.get("kind"))
                score += total_runs
                wickets += int(wicket_kind is not None)
                over_actual_runs += total_runs
                over_actual_wickets += int(wicket_kind is not None)
                if is_legal:
                    legal_balls += 1
                deliveries.append(ToiDelivery(
                    match_id=path.stem, innings=innings_number, over=over_number, ball=ball_index,
                    bowler=str(delivery.get("bowler", "")), striker=str(delivery.get("batter", "")),
                    total_runs=total_runs, batter_runs=batter_runs, extras=extras,
                    wicket_kind=wicket_kind, feed_total=score, feed_wickets=wickets,
                    timestamp_ms=0, commentary="",
                ))

            innings_total += over_actual_runs
            # predict BEFORE this over using state as of its start; here we
            # replay after-the-fact so evaluate accuracy retrospectively:
            # snapshot deliveries as they stood at the START of this over.
            prior_deliveries = [d for d in deliveries if not (d.over == over_number)]
            over_first = over_data.get("deliveries", [{}])[0]
            striker = str(over_first.get("batter", ""))
            non_striker = str(over_first.get("non_striker", ""))
            balls_before = sum(1 for d in prior_deliveries if d.is_legal)
            score_before = score - over_actual_runs
            wkts_before = wickets - over_actual_wickets
            balls_remaining = max(0, scheduled_overs * 6 - balls_before)
            current_rate = score_before * 6 / balls_before if balls_before else 0.0
            runs_required = max(0, target - score_before) if target else 0
            required_rate = runs_required * 6 / balls_remaining if target and balls_remaining else 0.0
            recent = prior_deliveries[-18:]
            recent_legal = [d for d in recent if d.is_legal]

            try:
                result = runtime.predict_next_over(
                    deliveries=prior_deliveries, registry=registry,
                    striker_name=striker, non_striker_name=non_striker,
                    venue_name=venue, batting_team=batting_team,
                    over=over_number, score_before_over=score_before,
                    wkts_down_before_over=wkts_before, wickets_in_hand=max(0, 10 - wkts_before),
                    legal_balls_bowled=balls_before, balls_remaining=balls_remaining,
                    current_run_rate=current_rate, is_chase=bool(target),
                    runs_required=runs_required, required_run_rate=required_rate,
                    recent_legal_balls=len(recent_legal),
                    recent_runs_per_ball=(sum(d.total_runs for d in recent_legal) / len(recent_legal)) if recent_legal else 0.0,
                    recent_dot_rate=(sum(d.total_runs == 0 for d in recent_legal) / len(recent_legal)) if recent_legal else 0.0,
                    recent_single_rate=(sum(d.total_runs == 1 for d in recent_legal) / len(recent_legal)) if recent_legal else 0.0,
                    recent_boundary_rate=(sum(d.batter_runs in (4, 6) for d in recent_legal) / len(recent_legal)) if recent_legal else 0.0,
                    recent_wicket_rate=(sum(d.wicket_kind is not None for d in recent_legal) / len(recent_legal)) if recent_legal else 0.0,
                )
            except Exception as exc:
                results.append({"error": f"{type(exc).__name__}: {exc}"})
                continue

            hit = result["sharp_2_low"] <= over_actual_runs <= result["sharp_2_high"]
            results.append({
                "over": over_number, "predicted_low": result["sharp_2_low"],
                "predicted_high": result["sharp_2_high"], "actual": over_actual_runs,
                "hit": hit, "striker_prior_balls": result["enriched_features"]["striker_prior_balls"],
            })
        if innings_number == 1:
            first_total = innings_total
    return results


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime = RunRangeRuntimeV3(root / "models/candidates/run_range_enriched_v3_batting_style")

    eligible = male_source_files(root)
    all_files = [p for p in (root / "data/raw/cricsheet/ipl").glob("*.json") if p.name in eligible]
    all_files += [p for p in (root / "data/raw/cricsheet/t20i").glob("*.json") if p.name in eligible]
    random.seed(7)
    sample = random.sample(all_files, 40)

    all_results = []
    errors = []
    resolved_prior_count = 0
    for i, path in enumerate(sample):
        if i % 10 == 0:
            print(f"  ... {i}/{len(sample)}", flush=True)
        try:
            match_results = replay_match(runtime, path)
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        for r in match_results:
            if "error" in r:
                errors.append(f"{path.name}: {r['error']}")
            else:
                all_results.append(r)
                if r["striker_prior_balls"] > 0:
                    resolved_prior_count += 1

    hits = sum(r["hit"] for r in all_results)
    print(f"\nMatches sampled: {len(sample)}")
    print(f"Overs evaluated: {len(all_results)}")
    print(f"Errors: {len(errors)}")
    for e in errors[:10]:
        print(f"  ERROR: {e}")
    print(f"Hit rate: {hits/len(all_results)*100:.2f}%" if all_results else "n/a")
    print(f"Offline holdout hit rate (reference): 28.49%")
    print(f"Rows with resolved (non-zero) striker prior stats: {resolved_prior_count}/{len(all_results)} ({resolved_prior_count/len(all_results)*100:.1f}%)")


if __name__ == "__main__":
    main()
