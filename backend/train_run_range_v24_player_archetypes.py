"""Does clustering batters/bowlers into role archetypes (K-means on their
3-phase scoring/economy profile -- powerplay/middle/death simultaneously)
add signal beyond the existing single-current-phase prior features?
Follows external research (2026-08-26 web search: IPL batting-archetype
studies using K-means on phase/style stat profiles) -- genuinely
different mechanism from anything tested today: a generalization/role
layer, not another raw stat or interaction.

Centroids are fit on TRAIN-split rows only (no calibration/holdout rows
touch the clustering itself) using each row's own career-to-date 3-phase
profile from app/ml/player_archetype_dataset.py (already leakage-safe on
the temporal axis) -- then every row (train/cal/holdout) is assigned to
its nearest centroid. K=4 for both batters and bowlers, a reasonable
first cut matching the "anchor/finisher/accumulator/..." style groupings
seen in the external research.

Spot-checked first (2026-08-26): MS Dhoni's death-overs shrunk rate
(1.71) towers over his powerplay/middle (1.18/1.12) -- matches his real
finisher reputation. Kohli/Rohit show stronger, more even profiles,
matching their accumulator/opener roles.

Tested on top of the current live run_range_v11_partnership_rate, same
architecture/split/IPL-T20I methodology, corrected IPL band width
(2026-08-26 fix).
"""
from __future__ import annotations

import json
from pathlib import Path

import hashlib

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.cluster import KMeans

from app.ml.ipl_phase_moe_dataset import build_ipl_phase_moe_features
from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset
from app.ml.partnership_dataset import build_partnership_dataset
from app.ml.player_archetype_dataset import build_player_archetype_dataset
from app.ml.player_competition_dataset import build_player_competition_features
from app.ml.player_venue_phase_dataset import build_player_venue_phase_features
from train_contract22_rigorous import BOWLER_FEATURES, build_enriched
from train_phase_calibrated_sharp_range_v33 import MAX_RUN_CLASS, best_bands, male_source_files, nll, temperature_scale
from train_run_range_enriched_v3_batting_style import (
    BASE_FEATURES, MOE_EXTRA_FEATURES, VENUE_FEATURES, _load_batting_style_lookup, features,
)
from train_run_range_v4_batter_phase import BATTER_PHASE_FEATURES

VERSION = "run_range_v24_player_archetypes"
COMPETITION_PRIOR_FEATURES = [
    "batter_competition_runs_per_ball_shrunk", "batter_competition_boundary_rate_shrunk",
    "batter_competition_dismissal_rate_shrunk", "batter_competition_balls",
    "bowler_competition_economy_shrunk", "bowler_competition_wicket_rate_shrunk",
    "bowler_competition_balls",
]
PARTNERSHIP_FEATURES = ["partnership_runs", "partnership_balls", "partnership_run_rate"]
NEW_FEATURES = ["striker_archetype", "bowler_archetype"]
FEATURES = (
    BASE_FEATURES + MOE_EXTRA_FEATURES + VENUE_FEATURES + BOWLER_FEATURES
    + BATTER_PHASE_FEATURES + COMPETITION_PRIOR_FEATURES + PARTNERSHIP_FEATURES + NEW_FEATURES
)
CATEGORICAL = [
    "phase", "active_batter_state", "wickets_remaining_bucket",
    "state_regime", "chase_pressure", "venue_scoring_regime",
    "venue_par_source", "batting_team_venue_context", "phase_venue_regime",
    "striker_batting_style", "bowler_type", "striker_archetype", "bowler_archetype",
]
SHRINKAGE_BALLS = 150
K_CLUSTERS = 4

root = Path(__file__).resolve().parents[1]
output = root / "models/candidates" / VERSION
output.mkdir(parents=True, exist_ok=True)

ipl_files = {p.name for p in (root / "data/raw/cricsheet/ipl").glob("*.json")}
eligible = male_source_files(root)
base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
base = base[base["source_file"].isin(eligible)].copy()
base["match_date"] = pd.to_datetime(base["match_date"])

print("Building enriched MOE + venue features...", flush=True)
moe = build_ipl_phase_moe_features(root, canonical_identities=True, scopes=("ipl", "t20i"))
venue = build_ipl_venue_regime_dataset(root, scopes=("ipl", "t20i"))
print("Building bowler features...", flush=True)
enriched = build_enriched(root, eligible)
print("Building batter phase features...", flush=True)
batter_phase = build_player_venue_phase_features(root, scopes=("ipl", "t20i"))
print("Building competition-specific player features...", flush=True)
competition = build_player_competition_features(root, scopes=("ipl", "t20i"))
print("Building partnership rate features...", flush=True)
partnership = build_partnership_dataset(root, scopes=("ipl", "t20i"))
print("Building player archetype profile dataset (the new signal being tested)...", flush=True)
archetype = build_player_archetype_dataset(root)

keys = ["source_file", "innings", "over"]
moe_columns = [c for c in moe.columns if c not in ("match_date",) and c not in keys]
venue_columns = [c for c in venue.columns if c not in ("match_date",) and c not in keys]
data = base.merge(moe[keys + moe_columns], on=keys, how="inner", validate="one_to_one")
data = data.merge(venue[keys + venue_columns], on=keys, how="inner", validate="one_to_one")
data = data.merge(enriched[keys + BOWLER_FEATURES], on=keys, how="inner", validate="one_to_one")
data = data.merge(batter_phase[keys + BATTER_PHASE_FEATURES], on=keys, how="inner", validate="one_to_one")
comp_cols = [
    "batter_competition_balls", "batter_competition_runs_per_ball",
    "batter_competition_boundary_rate", "batter_competition_dismissal_rate",
    "bowler_competition_balls", "bowler_competition_economy", "bowler_competition_wicket_rate",
]
data = data.merge(competition[keys + comp_cols], on=keys, how="inner", validate="one_to_one")
data = data.merge(partnership[keys + PARTNERSHIP_FEATURES], on=keys, how="inner", validate="one_to_one")
archetype_cols = [
    "striker_powerplay_rate_shrunk", "striker_middle_rate_shrunk", "striker_death_rate_shrunk",
    "bowler_powerplay_economy_shrunk", "bowler_middle_economy_shrunk", "bowler_death_economy_shrunk",
]
data = data.merge(archetype[keys + archetype_cols], on=keys, how="inner", validate="one_to_one")
print(f"Merged rows: {len(data)} (base was {len(base)})")

style_lookup = _load_batting_style_lookup(root)
data["striker_batting_style"] = data["striker"].map(style_lookup)
data["is_ipl"] = data["source_file"].isin(ipl_files)

w = data["batter_competition_balls"] / (data["batter_competition_balls"] + SHRINKAGE_BALLS)
data["batter_competition_runs_per_ball_shrunk"] = w * data["batter_competition_runs_per_ball"] + (1 - w) * data["striker_prior_runs_per_ball"]
data["batter_competition_boundary_rate_shrunk"] = w * data["batter_competition_boundary_rate"] + (1 - w) * data["striker_prior_boundary_rate"]
data["batter_competition_dismissal_rate_shrunk"] = w * data["batter_competition_dismissal_rate"] + (1 - w) * data["striker_prior_dismissal_rate"]
wb = data["bowler_competition_balls"] / (data["bowler_competition_balls"] + SHRINKAGE_BALLS)
bowler_pooled_economy = data["bowl_hist_avg_runs_conceded"] / 6.0
data["bowler_competition_economy_shrunk"] = wb * data["bowler_competition_economy"] + (1 - wb) * bowler_pooled_economy
data["bowler_competition_wicket_rate_shrunk"] = wb * data["bowler_competition_wicket_rate"] + (1 - wb) * data["bowl_hist_wicket_rate"]

train = data[data["match_date"] <= "2023-12-31"].reset_index(drop=True)
calibration = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)
holdout_is_ipl = holdout["is_ipl"].to_numpy()

# Fit K-means centroids on TRAIN rows only (no cal/holdout leakage into
# the clustering itself), then assign every split's rows via nearest
# centroid (predict, not fit) using their own career-to-date profile.
batter_cols = ["striker_powerplay_rate_shrunk", "striker_middle_rate_shrunk", "striker_death_rate_shrunk"]
bowler_cols = ["bowler_powerplay_economy_shrunk", "bowler_middle_economy_shrunk", "bowler_death_economy_shrunk"]
batter_kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10).fit(train[batter_cols])
bowler_kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10).fit(train[bowler_cols])

for split in (train, calibration, holdout):
    split["striker_archetype"] = [f"batter_{c}" for c in batter_kmeans.predict(split[batter_cols])]
    split["bowler_archetype"] = [f"bowler_{c}" for c in bowler_kmeans.predict(split[bowler_cols])]

print("Batter archetype centroids (powerplay/middle/death runs-per-ball):")
print(pd.DataFrame(batter_kmeans.cluster_centers_, columns=batter_cols))
print("Bowler archetype centroids (powerplay/middle/death economy):")
print(pd.DataFrame(bowler_kmeans.cluster_centers_, columns=bowler_cols))

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

temperatures = {}
calibration_scaled = calibration_raw.copy()
holdout_scaled = holdout_raw.copy()
for phase in ("powerplay", "middle", "death"):
    calibration_mask = (calibration["phase"].astype(str) == phase).to_numpy()
    holdout_mask = (holdout["phase"].astype(str) == phase).to_numpy()
    result = minimize_scalar(
        lambda value: nll(temperature_scale(calibration_raw[calibration_mask], value), calibration_actual[calibration_mask]),
        bounds=(0.5, 3.0), method="bounded",
    )
    temperatures[phase] = float(result.x)
    calibration_scaled[calibration_mask] = temperature_scale(calibration_raw[calibration_mask], temperatures[phase])
    holdout_scaled[holdout_mask] = temperature_scale(holdout_raw[holdout_mask], temperatures[phase])

holdout_low = np.empty(len(holdout), dtype=int)
holdout_high = np.empty(len(holdout), dtype=int)
holdout_low[holdout_is_ipl], holdout_high[holdout_is_ipl] = best_bands(holdout_scaled[holdout_is_ipl], width=3)
holdout_low[~holdout_is_ipl], holdout_high[~holdout_is_ipl] = best_bands(holdout_scaled[~holdout_is_ipl], width=2)
holdout_hit = (holdout_actual >= holdout_low) & (holdout_actual <= holdout_high)
raw_calibration_nll = nll(calibration_raw, calibration_actual)
scaled_calibration_nll = nll(calibration_scaled, calibration_actual)

v11_blended, v11_ipl, v11_t20i = 0.3004, 0.3325, 0.2945  # corrected 2026-08-26 numbers
new_blended = float(np.mean(holdout_hit))
new_ipl = float(np.mean(holdout_hit[holdout_is_ipl]))
new_t20i = float(np.mean(holdout_hit[~holdout_is_ipl]))
beats_v11 = bool(new_blended > v11_blended and scaled_calibration_nll <= raw_calibration_nll)
beats_v11_on_ipl_specifically = bool(new_ipl > v11_ipl)

importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
importance_rank = {name: rank + 1 for rank, (name, _) in enumerate(importances)}

report = {
    "candidate_version": VERSION,
    "candidate_only": True,
    "production_changed": False,
    "run_model_changed": False,
    "note": (
        "run_range_v11_partnership_rate (currently live) + batter/bowler "
        "role archetype (K=4 K-means on career-to-date 3-phase scoring/"
        "economy profile, centroids fit on train split only). Following "
        "external research on IPL batting-archetype clustering. "
        "Spot-checked: Dhoni's death rate 1.71 vs powerplay 1.18 -- real "
        "finisher shape."
    ),
    "split": {"train_rows": len(train), "holdout_rows": len(holdout)},
    "reference_run_range_v11_partnership_rate_corrected": {
        "blended": v11_blended, "ipl": v11_ipl, "t20i": v11_t20i,
    },
    "holdout": {
        "blended_hit_rate": new_blended,
        "ipl_hit_rate": new_ipl,
        "t20i_hit_rate": new_t20i,
    },
    "beats_v11_blended_and_calibration_gate": beats_v11,
    "beats_v11_on_ipl_specifically": beats_v11_on_ipl_specifically,
    "archetype_feature_importance_rank": {name: importance_rank[name] for name in NEW_FEATURES},
    "total_features": len(FEATURES),
    "decision": "promote_candidate" if beats_v11 else "reject_keep_research",
}
joblib.dump(model, output / "sharp_range_model.pkl")
joblib.dump(temperatures, output / "phase_temperatures.pkl")
joblib.dump(FEATURES, output / "feature_cols.pkl")
joblib.dump(CATEGORICAL, output / "categorical_cols.pkl")
joblib.dump(batter_kmeans, output / "batter_kmeans.pkl")
joblib.dump(bowler_kmeans, output / "bowler_kmeans.pkl")
(output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest_files = ["sharp_range_model.pkl", "phase_temperatures.pkl", "feature_cols.pkl", "categorical_cols.pkl"]
manifest = {"candidate_version": VERSION, "artifacts": {name: _sha256(output / name) for name in manifest_files}}
(output / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
