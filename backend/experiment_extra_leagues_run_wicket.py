"""Does adding the other men's T20 leagues to the training rows help the run-range / wicket (Candidate v3 GBM) models?

Rebuilds the other-leagues over table if missing (data/candidates/v3/extra_leagues_overs.csv, not committed), then fits the
v3 recipe (train <= 2023, calibrate 2024, holdout 2025+) on T20I+IPL alone, + other leagues, + other leagues at weight 0.4.
Result 2026-10-11: see docs/finding_other_leagues_run_wicket.md. Usage: python experiment_extra_leagues_run_wicket.py
"""
import json
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from train_validate_candidate_v3 import V3_FEATURES, CATEGORICAL_FEATURES, _features, _models, _metrics
from train_lock_ranked_candidate_v3 import TOP_10

root = Path(__file__).resolve().parents[1]
EXTRA = ("bbl", "bpl", "cpl", "ilt", "lpl", "mlc", "msl", "ntb", "psl", "sat", "sma", "ssm")
extra_path = root / "data/candidates/v3/extra_leagues_overs.csv"
if not extra_path.exists():
    from app.ml.candidate_v3_dataset import build_verified_dataset

    build_verified_dataset(root, scopes=EXTRA).frame.to_csv(extra_path, index=False)
D = root / "data/candidates/v3"
base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
extra = pd.read_csv(extra_path)
for f in (base, extra):
    f["match_date"] = pd.to_datetime(f["match_date"])
t20i_files = {p.name for p in (root / "data/raw/cricsheet/t20i").glob("*.json")}
ipl_files = {p.name for p in (root / "data/raw/cricsheet/ipl").glob("*.json")}
teams = {}
for p in (root / "data/raw/cricsheet/t20i").glob("*.json"):
    teams[p.name] = tuple(json.loads(p.read_text())["info"].get("teams", ()))
base["is_t20i"] = base["source_file"].isin(t20i_files)
top10_files = {n for n, t in teams.items() if len(t) == 2 and set(t) <= set(TOP_10)}

def fit(train, cal, weights=None):
    runs, wk = _models()
    X = _features(train, V3_FEATURES)
    runs.fit(X, train["runs_in_over"], sample_weight=weights, categorical_feature=CATEGORICAL_FEATURES)
    wk.fit(X, train["wicket_in_over"], sample_weight=weights, categorical_feature=CATEGORICAL_FEATURES)
    p = wk.predict_proba(_features(cal, V3_FEATURES))[:, 1]
    cal_model = IsotonicRegression(out_of_bounds="clip").fit(p, cal["wicket_in_over"])
    return runs, wk, cal_model

tr_base = base[base["match_date"].dt.year <= 2023]
cal = base[base["match_date"].dt.year == 2024]
hold = base[base["match_date"].dt.year >= 2025]
tr_extra = extra[extra["match_date"].dt.year <= 2023]
print("train base", len(tr_base), "extra", len(tr_extra), "cal", len(cal), "holdout", len(hold), flush=True)
holds = {
    "T20I 2025+": hold[hold["is_t20i"]],
    "T20I top-10 sides 2025+": hold[hold["source_file"].isin(top10_files)],
    "IPL 2025+": hold[~hold["is_t20i"]],
}
variants = {
    "baseline (T20I+IPL)": (tr_base, None),
    "+ other leagues (equal weight)": (pd.concat([tr_base, tr_extra]), None),
    "+ other leagues (weight 0.4)": (pd.concat([tr_base, tr_extra]), np.r_[np.ones(len(tr_base)), 0.4 * np.ones(len(tr_extra))]),
}
results = {}
for name, (train, w) in variants.items():
    runs, wk, cm = fit(train, cal, w)
    results[name] = {}
    for hname, h in holds.items():
        m, _, _ = _metrics(h, runs, wk, cm, V3_FEATURES)
        results[name][hname] = m
    print(name, flush=True)
    for hname, m in results[name].items():
        print(f"   {hname:26s} rows {int(m['rows']):6d}  runs MAE {m['runs_mae']:.4f}  RMSE {m['runs_rmse']:.4f}  wicket Brier {m['wicket_brier']:.5f}  AUC {m['wicket_auc']:.4f}", flush=True)
(root / "data/candidates/v3/extra_leagues_experiment.json").write_text(json.dumps(results, indent=1))
