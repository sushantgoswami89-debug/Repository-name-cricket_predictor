from app.ml.cold_start_profiles import (
    PromotionEvidence,
    TemporaryProfile,
    aggressiveness_band,
    route_profile,
)
from datetime import date
import json
from pathlib import Path
import pandas as pd


def test_unverified_eligible_player_waits_for_review():
    evidence = PromotionEvidence(
        seasons=1,
        max_consecutive_starts=5,
        first_five_innings=(),
    )
    assert evidence.eligible
    assert evidence.status == "ELIGIBLE_REVIEW"


def test_verified_important_first_five_promotes():
    evidence = PromotionEvidence(
        seasons=1,
        max_consecutive_starts=2,
        first_five_innings=((31, 17),),
        external_source_verified=True,
    )
    assert evidence.important_first_five
    assert evidence.status == "MAIN"


def test_aggressiveness_bands_are_explicit():
    assert aggressiveness_band(176.79) == "VERY_AGGRESSIVE"
    assert aggressiveness_band(145.0) == "AGGRESSIVE"
    assert aggressiveness_band(None) == "UNKNOWN"


def test_generated_registry_has_no_role_anomalies():
    root = Path(__file__).resolve().parents[2]
    report = json.loads(
        (root / "data/reports/ipl_cold_start_v1/registry.json").read_text()
    )
    assert report["summary"]["unresolved_roles"] == 0
    assert all(p["final_role"] != "BOWLER" for p in report["profiles"])
    assert all(
        p["final_role"] == "BOWLER" for p in report["excluded_bowlers"]
    )


def test_canonical_candidate_has_no_unresolved_player_ids():
    root = Path(__file__).resolve().parents[2]
    data = pd.read_csv(
        root / "data/candidates/ipl_canonical_v2/training_overs.csv",
        usecols=["striker", "non_striker"],
    )
    assert not data["striker"].str.contains("unresolved", na=False).any()
    assert not data["non_striker"].str.contains("unresolved", na=False).any()


def test_source_dated_profile_is_unavailable_before_verification():
    profile = TemporaryProfile.from_dict(
        {
            "player_id": "p1",
            "player_name": "New Batter",
            "role": "BATTER",
            "usual_position": "OPENER",
            "batting_average": 31.0,
            "strike_rate": 145.0,
            "experience": ["SMAT"],
            "source_url": "https://example.test/profile",
            "source_title": "Verified profile",
            "source_published": "2024-01-01",
            "verified_at": "2024-01-03",
        }
    )
    evidence = PromotionEvidence(0, 0, ())
    assert route_profile(
        role="BATTER",
        evidence=evidence,
        temporary_profile=profile,
        match_date=date(2024, 1, 2),
    ) == "NEUTRAL"
    assert route_profile(
        role="BATTER",
        evidence=evidence,
        temporary_profile=profile,
        match_date=date(2024, 1, 4),
    ) == "TEMP"


def test_specialist_bowler_never_routes_to_batter_prior():
    profile = TemporaryProfile.from_dict(
        {
            "player_id": "p1",
            "player_name": "Bowler",
            "role": "BATTER",
            "usual_position": "OPENER",
            "source_url": "https://example.test/profile",
            "source_title": "Bad profile",
            "source_published": "2023-01-01",
            "verified_at": "2023-01-02",
        }
    )
    assert route_profile(
        role="BOWLER",
        evidence=PromotionEvidence(0, 0, ()),
        temporary_profile=profile,
        match_date=date(2024, 1, 1),
    ) == "BOWLER_SAFEGUARD"
