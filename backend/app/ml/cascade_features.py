"""Leakage-safe cascade features: each live model's own prediction, fed
as an input feature to the OTHER live model (2026-08-25).

Standing thread from 2026-08-23: "feed run-range's own expected-runs
prediction as an input feature to the wicket model (and/or wicket's
probability into run-range), testing whether the two already-separate,
already-fault-isolated engines' outputs carry residual signal for each
other." Explicitly NOT a proposal to merge or restructure run-range/
wicket/win-probability -- they stay separate engines; this only adds one
new scalar feature to each model's existing input, generated here.

**Why out-of-fold, not a direct predict() call**: using a model's own
prediction on rows it was trained on leaks information -- the model can
fit those exact rows far more precisely than it ever could on genuinely
unseen data, so a naive "predict on my own training set" feature would
be an artificially strong signal that doesn't reflect what the feature
is worth at real serving time. Standard fix: 5-fold GroupKFold (grouped
by `source_file` so no single match's overs span both a training and a
validation fold -- same-match rows are highly correlated, so a naive
row-level KFold would leak match-specific signal across the split)
over the TRAIN population only. Calibration/holdout rows are already
genuinely unseen once a model is fit on the full training period, so
those get real single-pass out-of-sample predictions, no extra folding
needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from app.ml.ipl_phase_moe_dataset import build_ipl_phase_moe_features
from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset
from app.ml.partnership_dataset import build_partnership_dataset
from app.ml.player_competition_dataset import build_player_competition_features
from app.ml.player_venue_phase_dataset import build_player_venue_phase_features
from app.ml.recency_weighted_prior_dataset import (
    build_batter_phase_recency_dataset, build_recency_weighted_prior_dataset,
)
from app.ml.team_composition_dataset import build_team_composition_dataset
from train_contract22_rigorous import (
    BASE_FEATURES as WICKET_BASE_FEATURES, BATTER_FEATURES, BOWLER_FEATURES,
    MATCHUP_FEATURES, VENUE_FEATURES as WICKET_VENUE_FEATURES, build_enriched,
)
from train_contract22_wicket_v2_batter_state import STATE_FEATURES
from train_contract22_wicket_v16_team_composition import (
    BATTER_PHASE_RECENCY_FEATURES, PARTNERSHIP_FEATURES as WICKET_PARTNERSHIP_FEATURES,
    RECENCY_FEATURES, TEAM_COMPOSITION_FEATURES,
)
from train_phase_calibrated_sharp_range_v33 import MAX_RUN_CLASS, male_source_files
from train_run_range_enriched_v3_batting_style import (
    BASE_FEATURES as RUN_RANGE_BASE_FEATURES, MOE_EXTRA_FEATURES, VENUE_FEATURES as RUN_RANGE_VENUE_FEATURES,
    _load_batting_style_lookup, features as run_range_frame,
)
from train_run_range_v4_batter_phase import BATTER_PHASE_FEATURES

KEYS = ["source_file", "innings", "over"]
N_FOLDS = 5

WICKET_FEATURES = (
    WICKET_BASE_FEATURES + BATTER_FEATURES + BOWLER_FEATURES + MATCHUP_FEATURES
    + WICKET_VENUE_FEATURES + STATE_FEATURES + RECENCY_FEATURES + WICKET_PARTNERSHIP_FEATURES
    + BATTER_PHASE_RECENCY_FEATURES + TEAM_COMPOSITION_FEATURES
)
WICKET_CATEGORICAL = [
    "phase", "striker_batting_style", "bowler_type", "venue_scoring_regime",
    "venue_par_source", "active_batter_state",
]

RUN_RANGE_COMPETITION_FEATURES = [
    "batter_competition_runs_per_ball_shrunk", "batter_competition_boundary_rate_shrunk",
    "batter_competition_dismissal_rate_shrunk", "batter_competition_balls",
    "bowler_competition_economy_shrunk", "bowler_competition_wicket_rate_shrunk",
    "bowler_competition_balls",
]
RUN_RANGE_PARTNERSHIP_FEATURES = ["partnership_runs", "partnership_balls", "partnership_run_rate"]
RUN_RANGE_FEATURES = (
    RUN_RANGE_BASE_FEATURES + MOE_EXTRA_FEATURES + RUN_RANGE_VENUE_FEATURES + BOWLER_FEATURES
    + BATTER_PHASE_FEATURES + RUN_RANGE_COMPETITION_FEATURES + RUN_RANGE_PARTNERSHIP_FEATURES
)
RUN_RANGE_CATEGORICAL = [
    "phase", "active_batter_state", "wickets_remaining_bucket",
    "state_regime", "chase_pressure", "venue_scoring_regime",
    "venue_par_source", "batting_team_venue_context", "phase_venue_regime",
    "striker_batting_style", "bowler_type",
]
SHRINKAGE_BALLS = 150


def _wicket_frame(d: pd.DataFrame, bowler_known: bool = True) -> pd.DataFrame:
    result = d[WICKET_FEATURES].copy()
    if not bowler_known:
        for col in BOWLER_FEATURES + MATCHUP_FEATURES + [
            "bowler_recency_balls", "bowler_recency_economy", "bowler_recency_wicket_rate",
        ]:
            if col in WICKET_CATEGORICAL:
                result[col] = "__UNKNOWN__"
            else:
                result[col] = 0.0
    for column in WICKET_CATEGORICAL:
        result[column] = result[column].fillna("__UNKNOWN__").astype("category")
    return result


def build_wicket_base_dataset(root: Path) -> pd.DataFrame:
    """Same merges as train_contract22_wicket_v16_team_composition.py --
    duplicated deliberately (this project's own convention: each new
    candidate script rebuilds its data from the shared app/ml/*.py
    builders rather than importing another script's top-level state)."""
    eligible = male_source_files(root)
    base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    base = base[base["source_file"].isin(eligible)].copy()
    base["match_date"] = pd.to_datetime(base["match_date"])

    enriched = build_enriched(root, eligible)
    venue = build_ipl_venue_regime_dataset(root, scopes=("ipl", "t20i"))
    moe = build_ipl_phase_moe_features(root, canonical_identities=True, scopes=("ipl", "t20i"))
    recency = build_recency_weighted_prior_dataset(root, scopes=("ipl", "t20i"))
    partnership = build_partnership_dataset(root, scopes=("ipl", "t20i"))
    batter_phase_recency = build_batter_phase_recency_dataset(root, scopes=("ipl", "t20i"))
    team_composition = build_team_composition_dataset(root, scopes=("ipl", "t20i"))

    data = base.merge(enriched, on=KEYS, how="inner", validate="one_to_one")
    data = data.merge(
        venue[KEYS + ["venue_par_score", "venue_prior_innings", "venue_scoring_regime", "venue_par_source"]],
        on=KEYS, how="inner", validate="one_to_one",
    )
    data = data.merge(moe[KEYS + STATE_FEATURES], on=KEYS, how="inner", validate="one_to_one")
    data = data.merge(recency[KEYS + RECENCY_FEATURES], on=KEYS, how="inner", validate="one_to_one")
    data = data.merge(partnership[KEYS + WICKET_PARTNERSHIP_FEATURES], on=KEYS, how="inner", validate="one_to_one")
    data = data.merge(batter_phase_recency[KEYS + BATTER_PHASE_RECENCY_FEATURES], on=KEYS, how="inner", validate="one_to_one")
    data = data.merge(team_composition[KEYS + TEAM_COMPOSITION_FEATURES], on=KEYS, how="inner", validate="one_to_one")

    ipl_files = {p.name for p in (root / "data/raw/cricsheet/ipl").glob("*.json")}
    data["is_ipl"] = data["source_file"].isin(ipl_files)
    return data


def build_run_range_base_dataset(root: Path) -> pd.DataFrame:
    """Same merges as train_run_range_v11_partnership_rate.py -- see
    build_wicket_base_dataset's docstring for why this is a deliberate
    duplication, not an import."""
    eligible = male_source_files(root)
    base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    base = base[base["source_file"].isin(eligible)].copy()
    base["match_date"] = pd.to_datetime(base["match_date"])

    moe = build_ipl_phase_moe_features(root, canonical_identities=True, scopes=("ipl", "t20i"))
    venue = build_ipl_venue_regime_dataset(root, scopes=("ipl", "t20i"))
    enriched = build_enriched(root, eligible)
    batter_phase = build_player_venue_phase_features(root, scopes=("ipl", "t20i"))
    competition = build_player_competition_features(root, scopes=("ipl", "t20i"))
    partnership = build_partnership_dataset(root, scopes=("ipl", "t20i"))

    moe_columns = [c for c in moe.columns if c not in ("match_date",) and c not in KEYS]
    venue_columns = [c for c in venue.columns if c not in ("match_date",) and c not in KEYS]
    data = base.merge(moe[KEYS + moe_columns], on=KEYS, how="inner", validate="one_to_one")
    data = data.merge(venue[KEYS + venue_columns], on=KEYS, how="inner", validate="one_to_one")
    data = data.merge(enriched[KEYS + BOWLER_FEATURES], on=KEYS, how="inner", validate="one_to_one")
    data = data.merge(batter_phase[KEYS + BATTER_PHASE_FEATURES], on=KEYS, how="inner", validate="one_to_one")
    comp_cols = [
        "batter_competition_balls", "batter_competition_runs_per_ball",
        "batter_competition_boundary_rate", "batter_competition_dismissal_rate",
        "bowler_competition_balls", "bowler_competition_economy", "bowler_competition_wicket_rate",
    ]
    data = data.merge(competition[KEYS + comp_cols], on=KEYS, how="inner", validate="one_to_one")
    data = data.merge(partnership[KEYS + RUN_RANGE_PARTNERSHIP_FEATURES], on=KEYS, how="inner", validate="one_to_one")

    style_lookup = _load_batting_style_lookup(root)
    data["striker_batting_style"] = data["striker"].map(style_lookup)
    ipl_files = {p.name for p in (root / "data/raw/cricsheet/ipl").glob("*.json")}
    data["is_ipl"] = data["source_file"].isin(ipl_files)

    w = data["batter_competition_balls"] / (data["batter_competition_balls"] + SHRINKAGE_BALLS)
    data["batter_competition_runs_per_ball_shrunk"] = w * data["batter_competition_runs_per_ball"] + (1 - w) * data["striker_prior_runs_per_ball"]
    data["batter_competition_boundary_rate_shrunk"] = w * data["batter_competition_boundary_rate"] + (1 - w) * data["striker_prior_boundary_rate"]
    data["batter_competition_dismissal_rate_shrunk"] = w * data["batter_competition_dismissal_rate"] + (1 - w) * data["striker_prior_dismissal_rate"]
    wb = data["bowler_competition_balls"] / (data["bowler_competition_balls"] + SHRINKAGE_BALLS)
    bowler_pooled_economy = data["bowl_hist_avg_runs_conceded"] / 6.0
    data["bowler_competition_economy_shrunk"] = wb * data["bowler_competition_economy"] + (1 - wb) * bowler_pooled_economy
    data["bowler_competition_wicket_rate_shrunk"] = wb * data["bowler_competition_wicket_rate"] + (1 - wb) * data["bowl_hist_wicket_rate"]
    return data


def build_run_range_cascade_feature(root: Path) -> pd.DataFrame:
    """Returns keys + `run_range_cascade_expected_runs` (leakage-safe, see
    module docstring) for every over in the full population -- the
    live run_range_v11_partnership_rate architecture's own predicted
    expected runs for that over, usable as a wicket-model input feature."""
    import lightgbm as lgb
    data = build_run_range_base_dataset(root)
    train = data[data["match_date"] <= "2023-12-31"].reset_index(drop=True)
    calibration = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
    holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)

    params = dict(
        objective="multiclass", num_class=MAX_RUN_CLASS + 1,
        n_estimators=60, learning_rate=0.08, num_leaves=25, max_depth=6,
        min_child_samples=100, subsample=0.85, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=42, verbose=-1,
    )
    class_values = np.arange(MAX_RUN_CLASS + 1)
    train_actual = np.minimum(train["runs_in_over"].to_numpy(), MAX_RUN_CLASS)

    print(f"  [run-range cascade] {N_FOLDS}-fold OOF over {len(train)} train rows...", flush=True)
    oof = np.zeros(len(train))
    gkf = GroupKFold(n_splits=N_FOLDS)
    for fold_train_idx, fold_val_idx in gkf.split(train, groups=train["source_file"]):
        fold_model = lgb.LGBMClassifier(**params)
        fold_model.fit(
            run_range_frame(train.iloc[fold_train_idx], RUN_RANGE_FEATURES, RUN_RANGE_CATEGORICAL),
            train_actual[fold_train_idx], categorical_feature=RUN_RANGE_CATEGORICAL,
        )
        proba = fold_model.predict_proba(run_range_frame(train.iloc[fold_val_idx], RUN_RANGE_FEATURES, RUN_RANGE_CATEGORICAL))
        oof[fold_val_idx] = proba @ class_values

    print("  [run-range cascade] full-train fit for calibration/holdout...", flush=True)
    full_model = lgb.LGBMClassifier(**params)
    full_model.fit(run_range_frame(train, RUN_RANGE_FEATURES, RUN_RANGE_CATEGORICAL), train_actual, categorical_feature=RUN_RANGE_CATEGORICAL)
    cal_pred = full_model.predict_proba(run_range_frame(calibration, RUN_RANGE_FEATURES, RUN_RANGE_CATEGORICAL)) @ class_values
    holdout_pred = full_model.predict_proba(run_range_frame(holdout, RUN_RANGE_FEATURES, RUN_RANGE_CATEGORICAL)) @ class_values

    train = train.assign(run_range_cascade_expected_runs=oof)
    calibration = calibration.assign(run_range_cascade_expected_runs=cal_pred)
    holdout = holdout.assign(run_range_cascade_expected_runs=holdout_pred)
    return pd.concat([train, calibration, holdout], ignore_index=True)[KEYS + ["run_range_cascade_expected_runs"]]


def build_wicket_cascade_feature(root: Path) -> pd.DataFrame:
    """Returns keys + `wicket_cascade_probability` (leakage-safe, see
    module docstring) -- the live contract22_wicket_v16_team_composition
    architecture's own predicted wicket_in_over probability, usable as a
    run-range-model input feature."""
    import lightgbm as lgb
    data = build_wicket_base_dataset(root)
    train = data[data["match_date"] <= "2023-12-31"].reset_index(drop=True)
    calibration = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
    holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)

    params = dict(
        objective="binary", n_estimators=300, learning_rate=0.03, max_depth=5,
        num_leaves=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    )
    train_actual = train["wicket_in_over"].to_numpy()

    print(f"  [wicket cascade] {N_FOLDS}-fold OOF over {len(train)} train rows...", flush=True)
    oof = np.zeros(len(train))
    gkf = GroupKFold(n_splits=N_FOLDS)
    for fold_train_idx, fold_val_idx in gkf.split(train, groups=train["source_file"]):
        fold_model = lgb.LGBMClassifier(**params)
        fold_model.fit(_wicket_frame(train.iloc[fold_train_idx]), train_actual[fold_train_idx], categorical_feature=WICKET_CATEGORICAL)
        oof[fold_val_idx] = fold_model.predict_proba(_wicket_frame(train.iloc[fold_val_idx]))[:, 1]

    print("  [wicket cascade] full-train fit for calibration/holdout...", flush=True)
    full_model = lgb.LGBMClassifier(**params)
    full_model.fit(_wicket_frame(train), train_actual, categorical_feature=WICKET_CATEGORICAL)
    cal_pred = full_model.predict_proba(_wicket_frame(calibration))[:, 1]
    holdout_pred = full_model.predict_proba(_wicket_frame(holdout))[:, 1]

    train = train.assign(wicket_cascade_probability=oof)
    calibration = calibration.assign(wicket_cascade_probability=cal_pred)
    holdout = holdout.assign(wicket_cascade_probability=holdout_pred)
    return pd.concat([train, calibration, holdout], ignore_index=True)[KEYS + ["wicket_cascade_probability"]]
