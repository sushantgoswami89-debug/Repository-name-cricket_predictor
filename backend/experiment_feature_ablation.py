"""Which inputs of the next-over forecast carry weight and which could be deleted? (user 2026-10-24: "let us think what to add and what to delete".)
Takes the proposed set (v3 features + career strike rates) and removes ONE group at a time, refitting the same four models (mean runs, big, quiet,
wicket). A group whose removal changes nothing is a candidate to delete. Holdout 2025+, change from the full set with a match-clustered bootstrap
interval. Usage: python experiment_feature_ablation.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiment_pitch_and_career_form import BIG, CAREER, QUIET, VENUE, fit_predict, prepare
from train_validate_candidate_v3 import V3_FEATURES

GROUPS = {
    "recent form (last 2 overs: runs, dots, singles, boundaries, wickets)": ["recent_legal_balls", "recent_runs_per_ball", "recent_dot_rate", "recent_single_rate",
                                                                            "recent_boundary_rate", "recent_wicket_rate"],
    "chase pressure (is_chase, runs_required, required_run_rate)": ["is_chase", "runs_required", "required_run_rate"],
    "current run rate": ["current_run_rate"],
    "score so far": ["score_before_over"],
    "wickets (down, in hand)": ["wkts_down_before_over", "wickets_in_hand"],
    "phase label": ["phase"],
    "position (over, balls bowled, balls remaining)": ["over", "legal_balls_bowled", "balls_remaining"],
    "career strike rates": CAREER,
    "pitch / venue history": VENUE,
}


def main() -> None:
    data = prepare()
    year = pd.to_datetime(data["match_date"]).dt.year
    train, hold = data[year <= 2023].reset_index(drop=True), data[year >= 2025].reset_index(drop=True)
    full = V3_FEATURES + CAREER + VENUE
    actual = hold["runs_in_over"].to_numpy()
    wicket = hold["wicket_in_over"].to_numpy() == 1
    runs = {"all inputs (v3 + career + venue)": fit_predict(train, hold, full)}
    for name, cols in GROUPS.items():
        runs[f"without {name}"] = fit_predict(train, hold, [c for c in full if c not in cols])
        print("done", name, flush=True)
    codes, uniq = pd.factorize(hold["source_file"].to_numpy())
    groups = [np.where(codes == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(7)
    boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(100)]
    ref = runs["all inputs (v3 + career + venue)"]
    print(f"\nholdout {len(hold)} overs / {len(uniq)} matches\n")
    print(f"{'variant':<80}{'RMSE':>7}{'bigAUC':>8}{'quietAUC':>9}{'wktAUC':>8}")
    for name, p in runs.items():
        print(f"{name:<80}{np.sqrt(((p['mean'] - actual) ** 2).mean()):7.3f}{roc_auc_score(actual >= BIG, p['big']):8.4f}"
              f"{roc_auc_score(actual <= QUIET, p['quiet']):9.4f}{roc_auc_score(wicket, p['wicket']):8.4f}")
    print("\nchange when the group is removed, 95% interval over matches (RMSE up / AUC down = the group was helping):")
    for name, p in runs.items():
        if name.startswith("all"):
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
