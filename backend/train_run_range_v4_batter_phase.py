"""Batters have never had a phase split anywhere in this codebase --
`run_range_enriched_v3_batting_style` (the currently-live run-range model)
gives the batter only career-wide prior stats
(striker_prior_runs_per_ball etc. from build_ipl_phase_moe_features), no
distinction between how a batter performs in the powerplay vs. death
overs specifically. Bowlers already have this (bowl_phase_avg_runs, and
BOWLER_FEATURES more broadly), but the batter side of the same idea was
never built until `player_venue_phase_dataset.py`'s batter_phase_* columns
(2026-08-22).

This tests whether that gap matters: same feature set as v3_batting_style
plus batter_phase_balls/runs_per_ball/boundary_rate/dismissal_rate (the
venue-agnostic phase aggregate, NOT the sparser venue x phase version --
that one already tested neutral on the wicket side in
contract22_wicket_v4_venue_phase). Same architecture, split, and
bowler-known/unknown evaluation as v3 otherwise. Candidate-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from app.ml.ipl_phase_moe_dataset import build_ipl_phase_moe_features
from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset
from app.ml.player_venue_phase_dataset import build_player_venue_phase_features
from train_contract22_rigorous import BOWLER_FEATURES, build_enriched
from train_phase_calibrated_sharp_range_v33 import (
    MAX_RUN_CLASS, best_bands, male_source_files, nll, temperature_scale,
)
from train_run_range_enriched_v3_batting_style import (
    BASE_FEATURES, MOE_EXTRA_FEATURES, VENUE_FEATURES,
    _load_batting_style_lookup, features,
)

VERSION = "run_range_v4_batter_phase"
BATTER_PHASE_FEATURES = [
    "batter_phase_balls", "batter_phase_runs_per_ball",
    "batter_phase_boundary_rate", "batter_phase_dismissal_rate",
]
FEATURES = BASE_FEATURES + MOE_EXTRA_FEATURES + VENUE_FEATURES + BOWLER_FEATURES + BATTER_PHASE_FEATURES
CATEGORICAL = [
    "phase", "active_batter_state", "wickets_remaining_bucket",
    "state_regime", "chase_pressure", "venue_scoring_regime",
    "venue_par_source", "batting_team_venue_context", "phase_venue_regime",
    "striker_batting_style", "bowler_type",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "models/candidates" / VERSION
    output.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    eligible = male_source_files(root)
    base = base[base["source_file"].isin(eligible)].copy()
    base["match_date"] = pd.to_datetime(base["match_date"])

    print("Building enriched MOE + venue features (scopes=ipl+t20i)...")
    moe = build_ipl_phase_moe_features(root, canonical_identities=True, scopes=("ipl", "t20i"))
    venue = build_ipl_venue_regime_dataset(root, scopes=("ipl", "t20i"))
    print("Building bowler features...")
    enriched = build_enriched(root, eligible)
    print("Building batter phase features (the new signal being tested)...")
    batter_phase = build_player_venue_phase_features(root, scopes=("ipl", "t20i"))
    keys = ["source_file", "innings", "over"]
    moe_columns = [c for c in moe.columns if c not in ("match_date",) and c not in keys]
    venue_columns = [c for c in venue.columns if c not in ("match_date",) and c not in keys]
    data = base.merge(moe[keys + moe_columns], on=keys, how="inner", validate="one_to_one")
    data = data.merge(venue[keys + venue_columns], on=keys, how="inner", validate="one_to_one")
    data = data.merge(enriched[keys + BOWLER_FEATURES], on=keys, how="inner", validate="one_to_one")
    data = data.merge(
        batter_phase[keys + BATTER_PHASE_FEATURES], on=keys, how="inner", validate="one_to_one"
    )
    print(f"Merged rows: {len(data)} (base was {len(base)})")

    style_lookup = _load_batting_style_lookup(root)
    data["striker_batting_style"] = data["striker"].map(style_lookup)

    train = data[data["match_date"] <= "2023-12-31"].reset_index(drop=True)
    calibration = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
    holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)

    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=MAX_RUN_CLASS + 1,
        n_estimators=60, learning_rate=0.08, num_leaves=25, max_depth=6,
        min_child_samples=100, subsample=0.85, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=42, verbose=-1,
    )
    train_actual = np.minimum(train["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
    model.fit(features(train, FEATURES, CATEGORICAL), train_actual, categorical_feature=CATEGORICAL)

    calibration_raw = model.predict_proba(features(calibration, FEATURES, CATEGORICAL))
    holdout_raw = model.predict_proba(features(holdout, FEATURES, CATEGORICAL))
    calibration_actual = np.minimum(calibration["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
    holdout_actual = np.minimum(holdout["runs_in_over"].to_numpy(), MAX_RUN_CLASS)

    temperatures: dict[str, float] = {}
    calibration_scaled = calibration_raw.copy()
    holdout_scaled = holdout_raw.copy()
    for phase in ("powerplay", "middle", "death"):
        calibration_mask = calibration["phase"].astype(str).to_numpy() == phase
        holdout_mask = holdout["phase"].astype(str).to_numpy() == phase
        result = minimize_scalar(
            lambda value: nll(
                temperature_scale(calibration_raw[calibration_mask], value),
                calibration_actual[calibration_mask],
            ),
            bounds=(0.5, 3.0), method="bounded",
        )
        temperatures[phase] = float(result.x)
        calibration_scaled[calibration_mask] = temperature_scale(calibration_raw[calibration_mask], temperatures[phase])
        holdout_scaled[holdout_mask] = temperature_scale(holdout_raw[holdout_mask], temperatures[phase])

    calibration_low, calibration_high = best_bands(calibration_scaled)
    holdout_low, holdout_high = best_bands(holdout_scaled)
    calibration_hit = np.mean((calibration_actual >= calibration_low) & (calibration_actual <= calibration_high))
    holdout_hit = np.mean((holdout_actual >= holdout_low) & (holdout_actual <= holdout_high))
    raw_calibration_nll = nll(calibration_raw, calibration_actual)
    scaled_calibration_nll = nll(calibration_scaled, calibration_actual)

    v3_hit = 0.2849  # documented run_range_enriched_v3_batting_style holdout hit rate -- the currently-live model
    beats_v3 = bool(holdout_hit > v3_hit and scaled_calibration_nll <= raw_calibration_nll)

    holdout_unknown_raw = model.predict_proba(
        features(holdout, FEATURES, CATEGORICAL, bowler_known=False)
    )
    holdout_unknown_scaled = holdout_unknown_raw.copy()
    for phase in ("powerplay", "middle", "death"):
        mask = holdout["phase"].astype(str).to_numpy() == phase
        holdout_unknown_scaled[mask] = temperature_scale(holdout_unknown_raw[mask], temperatures[phase])
    unknown_low, unknown_high = best_bands(holdout_unknown_scaled)
    holdout_hit_bowler_unknown = float(
        np.mean((holdout_actual >= unknown_low) & (holdout_actual <= unknown_high))
    )

    phase_breakdown = {}
    for phase in ("powerplay", "middle", "death"):
        mask = holdout["phase"].astype(str).to_numpy() == phase
        if mask.sum() == 0:
            continue
        phase_breakdown[phase] = {
            "rows": int(mask.sum()),
            "hit_rate": float(np.mean((holdout_actual[mask] >= holdout_low[mask]) & (holdout_actual[mask] <= holdout_high[mask]))),
        }

    importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    importance_rank = {name: rank + 1 for rank, (name, _) in enumerate(importances)}

    report = {
        "candidate_version": VERSION,
        "candidate_only": True,
        "production_changed": False,
        "run_model_changed": False,
        "note": (
            "run_range_enriched_v3_batting_style (currently live, 28.49% "
            "holdout hit rate) + batter_phase_balls/runs_per_ball/"
            "boundary_rate/dismissal_rate -- the venue-agnostic batter x "
            "phase profile that never existed anywhere in this codebase "
            "before player_venue_phase_dataset.py."
        ),
        "split": {
            "train_rows": len(train), "calibration_rows": len(calibration),
            "holdout_rows": len(holdout), "holdout_matches": int(holdout["source_file"].nunique()),
        },
        "phase_temperatures": temperatures,
        "calibration": {
            "raw_nll": raw_calibration_nll, "phase_scaled_nll": scaled_calibration_nll,
            "hit_rate": float(calibration_hit),
        },
        "holdout": {
            "v3_reference_hit_rate": v3_hit,
            "new_hit_rate": float(holdout_hit),
            "new_hit_rate_bowler_unknown": holdout_hit_bowler_unknown,
            "relative_improvement_over_v3": float((holdout_hit - v3_hit) / v3_hit),
        },
        "holdout_phase_breakdown": phase_breakdown,
        "beats_v3_and_calibration_gate": beats_v3,
        "batter_phase_feature_importance_rank": {
            f: importance_rank[f] for f in BATTER_PHASE_FEATURES
        },
        "total_features": len(FEATURES),
        "decision": "promote_candidate" if beats_v3 else "reject_keep_research",
    }
    joblib.dump(model, output / "sharp_range_model.pkl")
    joblib.dump(temperatures, output / "phase_temperatures.pkl")
    joblib.dump(FEATURES, output / "feature_cols.pkl")
    joblib.dump(CATEGORICAL, output / "categorical_cols.pkl")
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
