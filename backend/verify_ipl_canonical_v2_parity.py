"""Verify canonical v2 changes identities only, except corrected venue history."""

import json
from pathlib import Path

import pandas as pd

KEYS = ["source_file", "match_date", "innings", "over"]
TARGETS = ["runs_in_over", "wicket_in_over"]
IDENTITIES = ["striker", "non_striker", "batting_team", "venue_name"]
VENUE_DERIVED = [
    "venue_prior_innings",
    "venue_par_score",
    "venue_par_source",
    "venue_scoring_regime",
    "phase_venue_regime",
    "batting_team_venue_context",
]
PLAYER_IDENTITY_DERIVED = [
    "striker_prior_balls",
    "striker_prior_runs_per_ball",
    "striker_prior_dot_rate",
    "striker_prior_boundary_rate",
    "striker_prior_dismissal_rate",
    "partner_prior_balls",
    "partner_prior_runs_per_ball",
    "partner_prior_dot_rate",
    "partner_prior_boundary_rate",
]

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    venue = pd.read_csv(root / "data/candidates/ipl_venue_regime/features.csv")
    identity = pd.read_csv(root / "data/candidates/ipl_phase_moe_v1/features.csv")
    current = pd.read_csv(
        root / "data/candidates/ipl_canonical_v2/training_overs.csv"
    )
    for frame in (base, venue, identity, current):
        frame["match_date"] = frame["match_date"].astype(str)
    old = base.merge(venue, on=KEYS, validate="one_to_one").merge(
        identity, on=KEYS, validate="one_to_one"
    )
    compared = old.merge(
        current, on=KEYS, suffixes=("_old", "_new"), validate="one_to_one"
    )
    common = sorted(
        (set(old.columns) & set(current.columns))
        - set(KEYS)
        - set(IDENTITIES)
        - set(VENUE_DERIVED)
        - set(PLAYER_IDENTITY_DERIVED)
    )
    unexpected = {}
    for column in common:
        left = compared[f"{column}_old"]
        right = compared[f"{column}_new"]
        unequal = ~(left.eq(right) | (left.isna() & right.isna()))
        if unequal.any():
            unexpected[column] = int(unequal.sum())
    changed = {}
    for column in IDENTITIES + VENUE_DERIVED + PLAYER_IDENTITY_DERIVED:
        left = compared[f"{column}_old"]
        right = compared[f"{column}_new"]
        changed[column] = int(
            (~(left.eq(right) | (left.isna() & right.isna()))).sum()
        )
    report = {
        "parity_version": "ipl_canonical_v2_parity_v1",
        "passed": (
            len(old) == len(current) == len(compared)
            and not unexpected
            and all(
                compared[f"{target}_old"].equals(compared[f"{target}_new"])
                for target in TARGETS
            )
        ),
        "old_rows": len(old),
        "new_rows": len(current),
        "matched_rows": len(compared),
        "targets_identical": {
            target: bool(
                compared[f"{target}_old"].equals(compared[f"{target}_new"])
            )
            for target in TARGETS
        },
        "identity_and_venue_changes": changed,
        "unexpected_feature_changes": unexpected,
        "split_counts": {
            "training_through_2023": int(
                (pd.to_datetime(current["match_date"]).dt.year <= 2023).sum()
            ),
            "calibration_2024": int(
                (pd.to_datetime(current["match_date"]).dt.year == 2024).sum()
            ),
            "holdout_2025_plus": int(
                (pd.to_datetime(current["match_date"]).dt.year >= 2025).sum()
            ),
        },
    }
    output = root / "data/reports/ipl_canonical_v2_parity"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(1)
