"""How often does match_winner_v1 actually call the correct winner --
not just AUC (a ranking-quality metric), but real classification
accuracy at a 0.5 threshold. Reuses the exact same feature-building
pipeline as train_match_winner_v1.py (same merges, same split) but loads
the already-trained model/calibrator instead of retraining, and reports
accuracy overall, by phase, by IPL/T20I, and by confidence bucket (a
model that's only confident when it's actually more likely to be right
is doing its job even if overall accuracy looks modest near 50/50 early
in a match).

Also splits T20I further into full-member-international vs
associate-involved matches (2026-08-23): the blended "t20i" bucket is
heavily weighted toward associate/minor-nation fixtures -- 688 of 829
matches in the 2025+ holdout involve at least one associate nation, only
141 are genuine full-member internationals. Checking this directly
(same "don't trust a blended number" practice as the earlier IPL
run-range finding) found a real ~4.3-point gap: full-member
internationals score 74.0% accuracy vs associate-involved matches at
78.3% -- associate fixtures are often more lopsided (bigger skill gaps,
easier to call), while genuine contests between full-member sides are
closer and harder, similar in shape to why IPL (69.6%) is harder than
the blended T20I average.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from app.ml.ipl_phase_moe_dataset import build_ipl_phase_moe_features
from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset
from app.ml.match_winner_dataset import build_match_winner_dataset
from app.ml.partnership_dataset import build_partnership_dataset
from app.ml.recency_weighted_prior_dataset import (
    build_batter_phase_recency_dataset, build_recency_weighted_prior_dataset,
)
from app.ml.team_composition_dataset import build_team_composition_dataset
from app.ml.team_h2h_dataset import build_team_h2h_dataset
from app.ml.toss_dataset import build_toss_dataset
from app.ml.venue_recency_dataset import build_venue_recency_dataset
from train_contract22_rigorous import BASE_FEATURES, BATTER_FEATURES, BOWLER_FEATURES, MATCHUP_FEATURES, VENUE_FEATURES, build_enriched
from train_contract22_wicket_v2_batter_state import STATE_FEATURES
from train_match_winner_v1 import (
    BATTER_PHASE_RECENCY_FEATURES, CATEGORICAL, FEATURES, PARTNERSHIP_FEATURES,
    RECENCY_FEATURES, TEAM_COMPOSITION_FEATURES, TEAM_H2H_FEATURES,
    TOSS_FEATURES, VENUE_CONTEXT_FEATURES, VENUE_RECENCY_FEATURES, frame,
)
from train_phase_calibrated_sharp_range_v33 import male_source_files

root = Path(__file__).resolve().parents[1]
output = root / "models/candidates/match_winner_v1"

eligible = male_source_files(root)
base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
base = base[base["source_file"].isin(eligible)].copy()
base["match_date"] = pd.to_datetime(base["match_date"])

print("Rebuilding the same feature set as training (no retraining, just loading the model)...", flush=True)
winner = build_match_winner_dataset(root, scopes=("ipl", "t20i"))
enriched = build_enriched(root, eligible)
venue = build_ipl_venue_regime_dataset(root, scopes=("ipl", "t20i"))
moe = build_ipl_phase_moe_features(root, canonical_identities=True, scopes=("ipl", "t20i"))
recency = build_recency_weighted_prior_dataset(root, scopes=("ipl", "t20i"))
partnership = build_partnership_dataset(root, scopes=("ipl", "t20i"))
batter_phase_recency = build_batter_phase_recency_dataset(root, scopes=("ipl", "t20i"))
team_composition = build_team_composition_dataset(root, scopes=("ipl", "t20i"))
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
data = data.merge(batter_phase_recency[keys + BATTER_PHASE_RECENCY_FEATURES], on=keys, how="inner", validate="one_to_one")
data = data.merge(team_composition[keys + TEAM_COMPOSITION_FEATURES], on=keys, how="inner", validate="one_to_one")
data = data.merge(team_h2h[keys + TEAM_H2H_FEATURES], on=keys, how="left", validate="one_to_one")
data = data.merge(toss[keys + TOSS_FEATURES], on=keys, how="left", validate="one_to_one")
data = data.merge(venue_recency[keys + VENUE_RECENCY_FEATURES], on=keys, how="left", validate="one_to_one")

data["h2h_matches_played"] = data["h2h_matches_played"].fillna(0).astype(int)
data["h2h_batting_team_win_rate_shrunk"] = data["h2h_batting_team_win_rate_shrunk"].fillna(0.5)
data["toss_decision"] = data["toss_decision"].fillna("unknown")
data["batting_team_won_toss"] = data["batting_team_won_toss"].fillna(0).astype(int)

ipl_files = {p.name for p in (root / "data/raw/cricsheet/ipl").glob("*.json")}
data["is_ipl"] = data["source_file"].isin(ipl_files)

holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)
holdout_is_ipl = holdout["is_ipl"].to_numpy()
holdout_actual = holdout["batting_team_won"].to_numpy()

model = joblib.load(output / "match_winner_model.pkl")
calibrator = joblib.load(output / "match_winner_calibrator.pkl")

raw = model.predict_proba(frame(holdout, bowler_known=True))[:, 1]
clipped = np.clip(raw, 1e-6, 1 - 1e-6)
logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
proba = calibrator.predict_proba(logit)[:, 1]
predicted = (proba >= 0.5).astype(int)

overall_accuracy = float(accuracy_score(holdout_actual, predicted))

by_phase = {}
for phase_name in ("powerplay", "middle", "death"):
    mask = (holdout["phase"].astype(str) == phase_name).to_numpy()
    by_phase[phase_name] = {
        "rows": int(mask.sum()),
        "accuracy": float(accuracy_score(holdout_actual[mask], predicted[mask])),
    }

by_competition = {
    "ipl": {
        "rows": int(holdout_is_ipl.sum()),
        "accuracy": float(accuracy_score(holdout_actual[holdout_is_ipl], predicted[holdout_is_ipl])),
    },
    "t20i": {
        "rows": int((~holdout_is_ipl).sum()),
        "accuracy": float(accuracy_score(holdout_actual[~holdout_is_ipl], predicted[~holdout_is_ipl])),
    },
}

# T20I split further: does the blended t20i number above mask a gap
# between genuine full-member internationals and associate/minor-nation
# fixtures? See this module's docstring.
FULL_MEMBER_NATIONS = {
    "India", "Pakistan", "Australia", "England", "New Zealand",
    "South Africa", "West Indies", "Sri Lanka", "Bangladesh",
    "Zimbabwe", "Afghanistan", "Ireland",
}


def _is_full_member_international(source_file: str) -> bool:
    for scope in ("t20i", "ipl"):
        path = root / "data/raw/cricsheet" / scope / source_file
        if path.exists():
            teams = json.loads(path.read_text(encoding="utf-8"))["info"].get("teams", [])
            return len(teams) == 2 and all(t in FULL_MEMBER_NATIONS for t in teams)
    return False


t20i_source_files = holdout.loc[~holdout_is_ipl, "source_file"].unique()
fm_lookup = {sf: _is_full_member_international(sf) for sf in t20i_source_files}
is_fm_row = holdout["source_file"].map(fm_lookup).fillna(False).astype(bool).to_numpy()
t20i_mask = np.logical_not(holdout_is_ipl)
fm_row_mask = t20i_mask & is_fm_row
assoc_row_mask = t20i_mask & np.logical_not(is_fm_row)

t20i_split = {
    "full_member_international": {
        "matches": int(sum(1 for v in fm_lookup.values() if v)),
        "rows": int(fm_row_mask.sum()),
        "accuracy": (
            float(accuracy_score(holdout_actual[fm_row_mask], predicted[fm_row_mask]))
            if fm_row_mask.sum() else None
        ),
    },
    "associate_involved": {
        "matches": int(sum(1 for v in fm_lookup.values() if not v)),
        "rows": int(assoc_row_mask.sum()),
        "accuracy": (
            float(accuracy_score(holdout_actual[assoc_row_mask], predicted[assoc_row_mask]))
            if assoc_row_mask.sum() else None
        ),
    },
}

confident_mask = (proba >= 0.7) | (proba <= 0.3)
confident_accuracy = {
    "rows": int(confident_mask.sum()),
    "share_of_holdout": float(confident_mask.mean()),
    "accuracy": float(accuracy_score(holdout_actual[confident_mask], predicted[confident_mask])),
}
very_confident_mask = (proba >= 0.85) | (proba <= 0.15)
very_confident_accuracy = {
    "rows": int(very_confident_mask.sum()),
    "share_of_holdout": float(very_confident_mask.mean()),
    "accuracy": float(accuracy_score(holdout_actual[very_confident_mask], predicted[very_confident_mask])),
}

# Late-innings check flagged as untested in finding_match_winner_v1.md:
# does accuracy approach near-certainty in the final overs of a chase?
late_chase_mask = (holdout["is_chase"].to_numpy() == 1) & (holdout["over"].to_numpy() >= 18)
late_chase_accuracy = {
    "rows": int(late_chase_mask.sum()),
    "accuracy": (
        float(accuracy_score(holdout_actual[late_chase_mask], predicted[late_chase_mask]))
        if late_chase_mask.sum() > 0 else None
    ),
}

report = {
    "candidate_version": "match_winner_v1",
    "note": "Classification accuracy (correct winner called, 0.5 threshold) on the exact same 2025+ holdout as the AUC report -- not a retrain, same model/calibrator artifacts.",
    "holdout_rows": len(holdout),
    "overall_accuracy": overall_accuracy,
    "accuracy_by_phase": by_phase,
    "accuracy_by_competition": by_competition,
    "t20i_full_member_vs_associate": t20i_split,
    "confident_predictions_ge70pct_or_le30pct": confident_accuracy,
    "very_confident_predictions_ge85pct_or_le15pct": very_confident_accuracy,
    "late_chase_overs18to20_accuracy": late_chase_accuracy,
}
(output / "accuracy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
