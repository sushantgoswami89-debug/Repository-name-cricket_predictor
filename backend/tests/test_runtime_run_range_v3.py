from __future__ import annotations

from pathlib import Path

import pytest
from app.live.toi_reader import ToiDelivery
from runtime_run_range_v3 import RunRangeRuntimeV3

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "models/candidates/run_range_enriched_v3_batting_style"


def _sample_deliveries() -> list[ToiDelivery]:
    return [
        ToiDelivery(
            match_id="test", innings=1, over=1, ball=ball,
            bowler="Bowler", striker="Batter", total_runs=1, batter_runs=1,
            extras={}, wicket_kind=None, feed_total=ball, feed_wickets=0,
            timestamp_ms=ball, commentary="",
        )
        for ball in range(1, 7)
    ]


def test_runtime_loads_verified_artifacts_and_predicts_inclusive_band() -> None:
    runtime = RunRangeRuntimeV3(ARTIFACTS, verify_integrity=True)

    result = runtime.predict_next_over(
        deliveries=_sample_deliveries(), registry={},
        striker_name="Batter", non_striker_name="Partner",
        venue_name="Unknown", batting_team="Team A",
        over=2, score_before_over=6, wkts_down_before_over=0, wickets_in_hand=10,
        legal_balls_bowled=6, balls_remaining=114, current_run_rate=6.0,
        is_chase=False, runs_required=0, required_run_rate=0.0,
        recent_legal_balls=6, recent_runs_per_ball=1.0, recent_dot_rate=0.0,
        recent_single_rate=1.0, recent_boundary_rate=0.0, recent_wicket_rate=0.0,
        width=2,
    )

    assert result["sharp_2_high"] - result["sharp_2_low"] == 2
    assert 0.0 <= result["sharp_2_prob"] <= 1.0
    assert result["phase"] == "powerplay"


def test_runtime_rejects_bad_artifact_dir() -> None:
    with pytest.raises(FileNotFoundError):
        RunRangeRuntimeV3(ROOT / "models/candidates/does_not_exist")


def test_feature_computer_handles_unknown_players_gracefully() -> None:
    runtime = RunRangeRuntimeV3(ARTIFACTS, verify_integrity=True)

    result = runtime.predict_next_over(
        deliveries=[], registry={},
        striker_name="Totally Unknown Player XYZ", non_striker_name="Also Unknown",
        venue_name="A Ground Nobody Has Heard Of", batting_team="Team A",
        over=1, score_before_over=0, wkts_down_before_over=0, wickets_in_hand=10,
        legal_balls_bowled=0, balls_remaining=120, current_run_rate=0.0,
        is_chase=False, runs_required=0, required_run_rate=0.0,
        recent_legal_balls=0, recent_runs_per_ball=0.0, recent_dot_rate=0.0,
        recent_single_rate=0.0, recent_boundary_rate=0.0, recent_wicket_rate=0.0,
        width=2,
    )

    assert result["enriched_features"]["striker_prior_balls"] == 0
    assert result["enriched_features"]["striker_batting_style"] == "__UNKNOWN__"
    assert result["enriched_features"]["venue_par_source"] in {"global", "default"}
    assert 0.0 <= result["sharp_2_prob"] <= 1.0


def test_name_alias_resolves_known_player_with_no_registry() -> None:
    """Simulates a genuine live TOI feed: no Cricsheet registry available,
    only a plain name string. Should still resolve a well-known player via
    the identity_key-normalized name-alias snapshot, not just fall back to
    unknown -- this is the fix for the gap flagged when this runtime was
    first built (see docs/candidate_run_range_enriched_v2.md)."""
    runtime = RunRangeRuntimeV3(ARTIFACTS, verify_integrity=True)

    result = runtime.predict_next_over(
        deliveries=[], registry={},  # deliberately empty -- no registry
        striker_name="V Kohli", non_striker_name="F du Plessis",
        venue_name="M Chinnaswamy Stadium", batting_team="Royal Challengers Bangalore",
        over=8, score_before_over=55, wkts_down_before_over=2, wickets_in_hand=8,
        legal_balls_bowled=42, balls_remaining=78, current_run_rate=7.86,
        is_chase=False, runs_required=0, required_run_rate=0.0,
        recent_legal_balls=12, recent_runs_per_ball=1.0, recent_dot_rate=0.2,
        recent_single_rate=0.6, recent_boundary_rate=0.1, recent_wicket_rate=0.0,
        width=2,
    )

    assert result["enriched_features"]["striker_prior_balls"] > 0
    assert result["enriched_features"]["striker_batting_style"] != "__UNKNOWN__"


def test_full_name_form_resolves_via_full_name_fallback() -> None:
    """The test above uses "V Kohli" -- Cricsheet's OWN naming convention,
    which is why the original name-alias validation never caught this:
    real TOI feeds consistently announce players by their common first
    name ("Virat Kohli"), not Cricsheet's (often abbreviated) spelling.
    Found 2026-08-22 testing this runtime directly: "Virat Kohli" and
    "Jasprit Bumrah" both failed to resolve at all despite being about as
    famous as players get. Fixed via a full_name-derived first+last-token
    fallback tier (app.ml.ipl_identities.build_full_name_alias_index)."""
    runtime = RunRangeRuntimeV3(ARTIFACTS, verify_integrity=True)

    result = runtime.predict_next_over(
        deliveries=[], registry={},
        striker_name="Virat Kohli", non_striker_name="",
        venue_name="M Chinnaswamy Stadium", batting_team="Royal Challengers Bangalore",
        over=8, score_before_over=55, wkts_down_before_over=2, wickets_in_hand=8,
        legal_balls_bowled=42, balls_remaining=78, current_run_rate=7.86,
        is_chase=False, runs_required=0, required_run_rate=0.0,
        recent_legal_balls=12, recent_runs_per_ball=1.0, recent_dot_rate=0.2,
        recent_single_rate=0.6, recent_boundary_rate=0.1, recent_wicket_rate=0.0,
        bowler_name="Mitchell Starc", width=2,
    )

    assert result["enriched_features"]["striker_prior_balls"] > 0
    assert result["enriched_features"]["striker_batting_style"] != "__UNKNOWN__"
    assert result["enriched_features"]["bowl_career_overs_bowled"] > 0
    assert result["enriched_features"]["bowler_type"] != "__UNKNOWN__"
