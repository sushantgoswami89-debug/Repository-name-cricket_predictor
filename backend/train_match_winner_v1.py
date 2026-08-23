"""First cut at a live win-probability model -- a third prediction
alongside run-range and wicket-in-over (not a replacement, user-
confirmed 2026-08-23). Genuinely different target: not "what happens in
the next over" but "does the team currently batting go on to win the
match." See app/ml/match_winner_dataset.py's docstring for the label
definition and leakage reasoning.

Reuses the richest feature set already validated for wicket
(contract22_wicket_v16_team_composition's full lineage: recency-weighted
batter/bowler form, phase-specific batter recency, partnership rate,
team role composition) plus every pre-match/contextual signal rejected
this session for OVER-level prediction (toss, home/away venue context,
team-vs-team H2H, recency-weighted venue par score) -- each of those was
rejected specifically because it didn't move a single-over target, with
the finding docs explicitly noting they plausibly matter more for
whole-match outcomes. This is the first real test of that hypothesis.

Same chronological split as everything else this session
(train<=2023, calibration=2024, holdout>=2025), same IPL/T20I split
evaluation, same bowler-known/unknown evaluation for consistency (though
expected to matter less here than for wicket-in-over). No prior live
model to compare against -- baseline is a simple 3-feature logistic
regression (current run rate vs required run rate, wickets in hand,
balls remaining) as an honesty floor, not a strawman.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from app.ml.ipl_phase_moe_dataset import build_ipl_phase_moe_features
from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset
from app.ml.match_winner_dataset import build_match_winner_dataset
from app.ml.partnership_dataset import build_partnership_dataset, build_partnership_trend_dataset
from app.ml.recency_weighted_prior_dataset import (
    build_batter_phase_recency_dataset, build_recency_weighted_prior_dataset,
)
from app.ml.team_composition_dataset import build_team_composition_dataset
from app.ml.team_h2h_dataset import build_team_h2h_dataset
from app.ml.toss_dataset import build_toss_dataset
from app.ml.venue_recency_dataset import build_venue_recency_dataset
from train_contract22_rigorous import (
    BASE_FEATURES, BATTER_FEATURES, BOWLER_FEATURES, MATCHUP_FEATURES,
    VENUE_FEATURES, build_enriched,
)
from train_contract22_wicket_v2_batter_state import STATE_FEATURES
from train_phase_calibrated_sharp_range_v33 import male_source_files

VERSION = "match_winner_v1"

RECENCY_FEATURES = [
    "striker_recency_balls", "striker_recency_runs_per_ball", "striker_recency_dot_rate",
    "striker_recency_boundary_rate", "striker_recency_dismissal_rate",
    "partner_recency_runs_per_ball",
    "bowler_recency_balls", "bowler_recency_economy", "bowler_recency_wicket_rate",
]
BATTER_PHASE_RECENCY_FEATURES = [
    "striker_recency_phase_balls", "striker_recency_phase_runs_per_ball",
    "striker_recency_phase_boundary_rate", "striker_recency_phase_dismissal_rate",
]
PARTNERSHIP_FEATURES = ["partnership_runs", "partnership_balls", "partnership_run_rate"]
TEAM_COMPOSITION_FEATURES = [
    "batting_team_allrounder_count", "batting_team_specialist_batter_count",
    "bowling_team_pace_count", "bowling_team_spin_count", "bowling_team_allrounder_count",
]
TEAM_H2H_FEATURES = ["h2h_matches_played", "h2h_batting_team_win_rate_shrunk"]
TOSS_FEATURES = ["toss_decision", "batting_team_won_toss"]
VENUE_CONTEXT_FEATURES = ["batting_team_venue_context"]
VENUE_RECENCY_FEATURES = ["venue_recency_par_score", "venue_recency_prior_innings"]
PARTNERSHIP_TREND_FEATURES = ["partnership_recent_run_rate", "partnership_trend"]

FEATURES = (
    BASE_FEATURES + BATTER_FEATURES + BOWLER_FEATURES + MATCHUP_FEATURES
    + VENUE_FEATURES + STATE_FEATURES + RECENCY_FEATURES + PARTNERSHIP_FEATURES
    + BATTER_PHASE_RECENCY_FEATURES + TEAM_COMPOSITION_FEATURES + TEAM_H2H_FEATURES
    + TOSS_FEATURES + VENUE_CONTEXT_FEATURES + VENUE_RECENCY_FEATURES
)
CATEGORICAL = [
    "phase", "striker_batting_style", "bowler_type", "venue_scoring_regime",
    "venue_par_source", "active_batter_state", "toss_decision", "batting_team_venue_context",
]
BASELINE_FEATURES = ["current_run_rate", "required_run_rate", "wickets_in_hand", "balls_remaining"]


def frame(d: pd.DataFrame, bowler_known: bool) -> pd.DataFrame:
    result = d[FEATURES].copy()
    if not bowler_known:
        for col in BOWLER_FEATURES + MATCHUP_FEATURES + [
            "bowler_recency_balls", "bowler_recency_economy", "bowler_recency_wicket_rate",
        ]:
            if col in CATEGORICAL:
                result[col] = "__UNKNOWN__"
            else:
                result[col] = 0.0
    for column in CATEGORICAL:
        result[column] = result[column].fillna("__UNKNOWN__").astype("category")
    return result


def _platt_fit(raw: np.ndarray, actual: np.ndarray) -> LogisticRegression:
    clipped = np.clip(raw, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, random_state=42)
    calibrator.fit(logit, actual)
    return calibrator


def _platt_apply(calibrator: LogisticRegression, raw: np.ndarray) -> np.ndarray:
    clipped = np.clip(raw, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logit)[:, 1]


def _metrics(actual: np.ndarray, proba: np.ndarray) -> dict:
    if len(np.unique(actual)) < 2:
        return {"rows": int(len(actual)), "event_rate": float(actual.mean())}
    return {
        "rows": int(len(actual)),
        "event_rate": float(actual.mean()),
        "auc": float(roc_auc_score(actual, proba)),
        "brier": float(brier_score_loss(actual, proba)),
    }


def build_dataset(root: Path) -> pd.DataFrame:
    """Shared by main() and by anything re-evaluating/re-tuning this
    target (train/calibration/holdout split + is_ipl already applied) --
    factored out so those callers don't duplicate this merge logic."""
    eligible = male_source_files(root)
    base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    base = base[base["source_file"].isin(eligible)].copy()
    base["match_date"] = pd.to_datetime(base["match_date"])

    print("Building match-winner labels...", flush=True)
    winner = build_match_winner_dataset(root, scopes=("ipl", "t20i"))
    print("Building enriched/venue/state features...", flush=True)
    enriched = build_enriched(root, eligible)
    venue = build_ipl_venue_regime_dataset(root, scopes=("ipl", "t20i"))
    moe = build_ipl_phase_moe_features(root, canonical_identities=True, scopes=("ipl", "t20i"))
    recency = build_recency_weighted_prior_dataset(root, scopes=("ipl", "t20i"))
    partnership = build_partnership_dataset(root, scopes=("ipl", "t20i"))
    partnership_trend = build_partnership_trend_dataset(root, scopes=("ipl", "t20i"))
    batter_phase_recency = build_batter_phase_recency_dataset(root, scopes=("ipl", "t20i"))
    team_composition = build_team_composition_dataset(root, scopes=("ipl", "t20i"))
    print("Building the contextual signals rejected for over-level prediction "
          "(the hypothesis being tested: do they help match-level instead)...", flush=True)
    team_h2h = build_team_h2h_dataset(root, scopes=("ipl",))
    toss = build_toss_dataset(root, scopes=("ipl", "t20i"))
    venue_recency = build_venue_recency_dataset(root, scopes=("ipl", "t20i"))

    keys = ["source_file", "innings", "over"]
    data = base.merge(winner, on=keys, how="inner", validate="one_to_one")
    data = data.merge(enriched, on=keys, how="inner", validate="one_to_one")
    data = data.merge(
        venue[keys + [
            "venue_par_score", "venue_prior_innings", "venue_scoring_regime",
            "venue_par_source", "batting_team_venue_context",
        ]],
        on=keys, how="inner", validate="one_to_one",
    )
    data = data.merge(moe[keys + STATE_FEATURES], on=keys, how="inner", validate="one_to_one")
    data = data.merge(recency[keys + RECENCY_FEATURES], on=keys, how="inner", validate="one_to_one")
    data = data.merge(partnership[keys + PARTNERSHIP_FEATURES], on=keys, how="inner", validate="one_to_one")
    data = data.merge(partnership_trend[keys + PARTNERSHIP_TREND_FEATURES], on=keys, how="inner", validate="one_to_one")
    data = data.merge(batter_phase_recency[keys + BATTER_PHASE_RECENCY_FEATURES], on=keys, how="inner", validate="one_to_one")
    data = data.merge(team_composition[keys + TEAM_COMPOSITION_FEATURES], on=keys, how="inner", validate="one_to_one")
    data = data.merge(team_h2h[keys + TEAM_H2H_FEATURES], on=keys, how="left", validate="one_to_one")
    data = data.merge(toss[keys + TOSS_FEATURES], on=keys, how="left", validate="one_to_one")
    data = data.merge(venue_recency[keys + VENUE_RECENCY_FEATURES], on=keys, how="left", validate="one_to_one")
    print(f"Merged rows: {len(data)} (base was {len(base)}, winner-labeled {len(winner)})")

    data["h2h_matches_played"] = data["h2h_matches_played"].fillna(0).astype(int)
    data["h2h_batting_team_win_rate_shrunk"] = data["h2h_batting_team_win_rate_shrunk"].fillna(0.5)
    data["toss_decision"] = data["toss_decision"].fillna("unknown")
    data["batting_team_won_toss"] = data["batting_team_won_toss"].fillna(0).astype(int)

    ipl_files = {p.name for p in (root / "data/raw/cricsheet/ipl").glob("*.json")}
    data["is_ipl"] = data["source_file"].isin(ipl_files)
    return data


DEFAULT_LGBM_PARAMS = dict(
    objective="binary", n_estimators=300, learning_rate=0.03, max_depth=5,
    num_leaves=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
)


def main(
    lgbm_params: dict | None = None,
    extra_features: list[str] | None = None,
    version: str = VERSION,
) -> dict:
    root = Path(__file__).resolve().parents[1]
    output = root / "models/candidates" / version
    output.mkdir(parents=True, exist_ok=True)

    data = build_dataset(root)
    features = FEATURES + (extra_features or [])

    train = data[data["match_date"] <= "2023-12-31"].reset_index(drop=True)
    calibration = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
    holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)
    holdout_is_ipl = holdout["is_ipl"].to_numpy()
    print(f"train={len(train)} calibration={len(calibration)} holdout={len(holdout)}")
    print(f"holdout event rate (batting_team_won): {holdout['batting_team_won'].mean():.4f}")

    # Honesty-floor baseline: a plain 3-feature logistic regression, no
    # player/team-context features at all -- match state alone.
    baseline_model = LogisticRegression(max_iter=1000, C=1.0)
    baseline_model.fit(train[BASELINE_FEATURES], train["batting_team_won"])
    baseline_holdout_proba = baseline_model.predict_proba(holdout[BASELINE_FEATURES])[:, 1]

    def frame_local(d: pd.DataFrame, bowler_known: bool) -> pd.DataFrame:
        result = d[features].copy()
        if not bowler_known:
            for col in BOWLER_FEATURES + MATCHUP_FEATURES + [
                "bowler_recency_balls", "bowler_recency_economy", "bowler_recency_wicket_rate",
            ]:
                if col in CATEGORICAL:
                    result[col] = "__UNKNOWN__"
                else:
                    result[col] = 0.0
        for column in CATEGORICAL:
            result[column] = result[column].fillna("__UNKNOWN__").astype("category")
        return result

    params = {**DEFAULT_LGBM_PARAMS, **(lgbm_params or {})}
    model = lgb.LGBMClassifier(**params)
    model.fit(frame_local(train, bowler_known=True), train["batting_team_won"], categorical_feature=CATEGORICAL)

    def evaluate(bowler_known: bool) -> dict:
        calibration_raw = model.predict_proba(frame_local(calibration, bowler_known))[:, 1]
        holdout_raw = model.predict_proba(frame_local(holdout, bowler_known))[:, 1]
        actual_cal = calibration["batting_team_won"].to_numpy()
        actual = holdout["batting_team_won"].to_numpy()

        platt = _platt_fit(calibration_raw, actual_cal)
        platt_proba = _platt_apply(platt, holdout_raw)

        return {
            "auc": float(roc_auc_score(actual, holdout_raw)),
            "brier_platt": float(brier_score_loss(actual, platt_proba)),
            "brier_uncalibrated": float(brier_score_loss(actual, holdout_raw)),
            "event_rate": float(actual.mean()),
            "rows": len(actual),
            "ipl_platt": _metrics(actual[holdout_is_ipl], platt_proba[holdout_is_ipl]),
            "t20i_platt": _metrics(actual[~holdout_is_ipl], platt_proba[~holdout_is_ipl]),
        }

    ceiling = evaluate(bowler_known=True)
    realistic = evaluate(bowler_known=False)

    holdout_actual = holdout["batting_team_won"].to_numpy()
    baseline_metrics = {
        "auc": float(roc_auc_score(holdout_actual, baseline_holdout_proba)),
        "brier": float(brier_score_loss(holdout_actual, baseline_holdout_proba)),
    }

    # AUC by phase, to see whether the model is honestly uncertain early
    # (as it should be -- T20 is highly random in the powerplay) and
    # sharpens toward the death overs (as real win-probability models do).
    by_phase = {}
    holdout_raw_known = model.predict_proba(frame_local(holdout, bowler_known=True))[:, 1]
    platt_known = _platt_fit(
        model.predict_proba(frame_local(calibration, bowler_known=True))[:, 1],
        calibration["batting_team_won"].to_numpy(),
    )
    holdout_platt_known = _platt_apply(platt_known, holdout_raw_known)
    for phase_name in ("powerplay", "middle", "death"):
        mask = (holdout["phase"].astype(str) == phase_name).to_numpy()
        by_phase[phase_name] = _metrics(holdout_actual[mask], holdout_platt_known[mask])

    importances = sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])
    importance_rank = {name: rank + 1 for rank, (name, _) in enumerate(importances)}

    report = {
        "candidate_version": version,
        "candidate_only": True,
        "production_changed": False,
        "note": (
            "First cut at a live win-probability model, a third prediction "
            "alongside run-range and wicket-in-over (not a replacement). "
            "Reuses the full contract22_wicket_v16_team_composition "
            "feature lineage plus every contextual signal rejected this "
            "session for over-level prediction (toss, home/away, team "
            "H2H, venue recency) -- testing whether they matter for "
            "match-level outcome instead, per each finding doc's own "
            "hypothesis."
        ),
        "split": {"train_rows": len(train), "calibration_rows": len(calibration), "holdout_rows": len(holdout)},
        "excluded_no_winner_matches": "186 of 6767 (2.7%) -- see app/ml/match_winner_dataset.py",
        "baseline_3feature_logreg": baseline_metrics,
        "bowler_known_ceiling": ceiling,
        "bowler_unknown_realistic": realistic,
        "auc_by_phase_known_bowler": by_phase,
        "contextual_feature_importance_rank": {
            f: importance_rank[f] for f in (TEAM_H2H_FEATURES + TOSS_FEATURES + VENUE_CONTEXT_FEATURES + VENUE_RECENCY_FEATURES)
        },
        "total_features": len(features),
        "lgbm_params": params,
        "beats_3feature_baseline": bool(ceiling["auc"] > baseline_metrics["auc"]),
    }
    import joblib

    calibration_raw_prod = model.predict_proba(frame_local(calibration, bowler_known=True))[:, 1]
    production_calibrator = _platt_fit(calibration_raw_prod, calibration["batting_team_won"].to_numpy())
    joblib.dump(model, output / "match_winner_model.pkl")
    joblib.dump(production_calibrator, output / "match_winner_calibrator.pkl")
    joblib.dump(features, output / "feature_cols.pkl")
    joblib.dump(CATEGORICAL, output / "categorical_cols.pkl")
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
