"""Batter evidence in the order the user set (2026-10-24): the batter's runs and strike rate in EACH phase (career), then the current phase and
his rate in it, then his strike rate in this innings, then the match situation. Features, all from before the match / before the over:
  * profile: striker's and non-striker's career strike rate in powerplay / middle / death, and the share of his career balls faced in each phase
    (his role: opener, anchor, finisher),
  * current-phase strike rate (already in the career set), * in-innings runs, balls and strike rate, * the gap: in-innings rate minus his rate in
    THIS phase, * the match situation (the v3 features: run rates, wickets, chase pressure, recent overs).
Variants (train <= 2023, holdout 2025+): v3 only; + career rates (the best so far); + phase profile and role; + in-innings rate; + the gap.
Match-clustered bootstrap for the change from "+ career rates". Usage: python experiment_batter_phase_profile.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiment_pitch_and_career_form import BIG, CAREER, D, PROFILE, QUIET, fit_predict, prepare
from train_validate_candidate_v3 import V3_FEATURES

INNINGS = ["striker_runs", "striker_balls", "partner_runs", "partner_balls", "striker_innings_sr", "partner_innings_sr"]
GAP = ["striker_gap_vs_phase", "partner_gap_vs_phase"]


def main() -> None:
    data = prepare()
    form = pd.read_csv(D / "batter_form_overs.csv")
    data = data.merge(form, on=["source_file", "innings", "over"], how="inner", validate="one_to_one")
    for who in ("striker", "partner"):
        balls = data[f"{who}_balls"]
        data[f"{who}_innings_sr"] = np.where(balls > 0, data[f"{who}_runs"] / balls.clip(lower=1), np.nan)
        data[f"{who}_gap_vs_phase"] = np.where(balls > 0, data[f"{who}_innings_sr"] - data[f"{who}_phase_sr"], 0.0)
    year = pd.to_datetime(data["match_date"]).dt.year
    train, hold = data[year <= 2023].reset_index(drop=True), data[year >= 2025].reset_index(drop=True)
    actual = hold["runs_in_over"].to_numpy()
    wicket = hold["wicket_in_over"].to_numpy() == 1
    variants = {
        "v3 situation only": V3_FEATURES,
        "+ career rates (overall, this phase)": V3_FEATURES + CAREER,
        "+ phase profile and role": V3_FEATURES + CAREER + PROFILE,
        "+ in-innings runs, balls, rate": V3_FEATURES + CAREER + PROFILE + INNINGS,
        "+ gap (in-innings minus this phase)": V3_FEATURES + CAREER + PROFILE + INNINGS + GAP,
    }
    preds = {name: fit_predict(train, hold, cols) for name, cols in variants.items()}
    print(f"train {len(train)} rows, holdout {len(hold)} rows from {hold['source_file'].nunique()} matches\n")
    print(f"{'variant':<40}{'RMSE':>7}{'corr':>7}{'bigAUC':>8}{'quietAUC':>9}{'wktAUC':>8}")
    for name, p in preds.items():
        print(f"{name:<40}{np.sqrt(((p['mean'] - actual) ** 2).mean()):7.3f}{np.corrcoef(p['mean'], actual)[0, 1]:7.3f}"
              f"{roc_auc_score(actual >= BIG, p['big']):8.4f}{roc_auc_score(actual <= QUIET, p['quiet']):9.4f}{roc_auc_score(wicket, p['wicket']):8.4f}")
    codes, uniq = pd.factorize(hold["source_file"].to_numpy())
    groups = [np.where(codes == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(7)
    boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(120)]
    ref = preds["+ career rates (overall, this phase)"]
    print("\nchange from '+ career rates', 95% interval over matches (negative RMSE / positive AUC = better):")
    for name, p in preds.items():
        if name.startswith("v3") or name.startswith("+ career"):
            continue
        d = {"RMSE": [], "big": [], "quiet": [], "wicket": []}
        for idx in boots:
            a, w = actual[idx], wicket[idx]
            d["RMSE"].append(np.sqrt(((p["mean"][idx] - a) ** 2).mean()) - np.sqrt(((ref["mean"][idx] - a) ** 2).mean()))
            d["big"].append(roc_auc_score(a >= BIG, p["big"][idx]) - roc_auc_score(a >= BIG, ref["big"][idx]))
            d["quiet"].append(roc_auc_score(a <= QUIET, p["quiet"][idx]) - roc_auc_score(a <= QUIET, ref["quiet"][idx]))
            d["wicket"].append(roc_auc_score(w, p["wicket"][idx]) - roc_auc_score(w, ref["wicket"][idx]))
        ci = lambda v: f"{np.mean(v):+.4f} [{np.percentile(v, 2.5):+.4f},{np.percentile(v, 97.5):+.4f}]"
        print(f"  {name}\n     RMSE {ci(d['RMSE'])} | big {ci(d['big'])} | quiet {ci(d['quiet'])} | wicket {ci(d['wicket'])}")


if __name__ == "__main__":
    main()
