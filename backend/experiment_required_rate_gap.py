"""Does comparing the current run rate with the required run rate directly help? (user 2026-10-24: "are you matching it current required RR".)
The model has current_run_rate and required_run_rate as separate inputs. Adds, for chases only (zero otherwise): the gap (required - current), the ratio,
the runs needed per wicket in hand, and the required rate against the recent (last 2 overs) rate. Same split/recipe as the other experiments.
Usage: python experiment_required_rate_gap.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiment_pitch_and_career_form import BIG, CAREER, QUIET, fit_predict, prepare
from train_validate_candidate_v3 import V3_FEATURES

GAP = ["rr_gap", "rr_ratio", "runs_per_wicket_in_hand", "rr_gap_recent"]


def main() -> None:
    data = prepare()
    chase = data["is_chase"] == 1
    recent_rate = data["recent_runs_per_ball"] * 6
    data["rr_gap"] = np.where(chase, data["required_run_rate"] - data["current_run_rate"], 0.0)
    data["rr_ratio"] = np.where(chase & (data["current_run_rate"] > 0), data["required_run_rate"] / data["current_run_rate"].clip(lower=0.5), 0.0)
    data["runs_per_wicket_in_hand"] = np.where(chase, data["runs_required"] / data["wickets_in_hand"].clip(lower=1), 0.0)
    data["rr_gap_recent"] = np.where(chase & (data["recent_legal_balls"] > 0), data["required_run_rate"] - recent_rate, 0.0)
    year = pd.to_datetime(data["match_date"]).dt.year
    train, hold = data[year <= 2023].reset_index(drop=True), data[year >= 2025].reset_index(drop=True)
    actual = hold["runs_in_over"].to_numpy()
    wicket = hold["wicket_in_over"].to_numpy() == 1
    chase_h = (hold["is_chase"] == 1).to_numpy()
    base = fit_predict(train, hold, V3_FEATURES + CAREER)
    gap = fit_predict(train, hold, V3_FEATURES + CAREER + GAP)
    print(f"holdout {len(hold)} overs, {chase_h.sum()} in chases\n")
    print(f"{'':<34}{'RMSE':>7}{'bigAUC':>8}{'quietAUC':>9}{'wktAUC':>8}")
    for label, mask in (("all overs", np.ones(len(hold), bool)), ("chase overs only", chase_h)):
        for name, p in (("career model", base), ("+ required-vs-current gap", gap)):
            a, w = actual[mask], wicket[mask]
            print(f"{label + ' | ' + name:<34}{np.sqrt(((p['mean'][mask] - a) ** 2).mean()):7.3f}{roc_auc_score(a >= BIG, p['big'][mask]):8.4f}"
                  f"{roc_auc_score(a <= QUIET, p['quiet'][mask]):9.4f}{roc_auc_score(w, p['wicket'][mask]):8.4f}")
    codes, uniq = pd.factorize(hold["source_file"].to_numpy())
    groups = [np.where(codes == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(7)
    d = {"RMSE": [], "big": [], "quiet": [], "wicket": []}
    for _ in range(120):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        a, w = actual[idx], wicket[idx]
        d["RMSE"].append(np.sqrt(((gap["mean"][idx] - a) ** 2).mean()) - np.sqrt(((base["mean"][idx] - a) ** 2).mean()))
        d["big"].append(roc_auc_score(a >= BIG, gap["big"][idx]) - roc_auc_score(a >= BIG, base["big"][idx]))
        d["quiet"].append(roc_auc_score(a <= QUIET, gap["quiet"][idx]) - roc_auc_score(a <= QUIET, base["quiet"][idx]))
        d["wicket"].append(roc_auc_score(w, gap["wicket"][idx]) - roc_auc_score(w, base["wicket"][idx]))
    ci = lambda v: f"{np.mean(v):+.4f} [{np.percentile(v, 2.5):+.4f},{np.percentile(v, 97.5):+.4f}]"
    print("\nchange from the career model (all overs), 95% interval over matches:")
    print(f"  RMSE {ci(d['RMSE'])} | big {ci(d['big'])} | quiet {ci(d['quiet'])} | wicket {ci(d['wicket'])}")


if __name__ == "__main__":
    main()
