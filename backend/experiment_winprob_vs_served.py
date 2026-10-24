"""Head to head on IDENTICAL rows: the SERVED win-probability model (match_winner_v1 + calibrator, scored by score_served_winner_holdout.py) against
the in-house model with team-phase coverage, pitch, Elo team strength and batter career rates (experiment_winprob_team_phase.py), plus an
average of the two. The in-house models are trained on <= 2024 matches only and scored on the 2025+ overs that the served model's feature pipeline also
covers. Reports Brier, log loss, AUC (overall, first innings, chases, by phase), a match-clustered bootstrap of the Brier change from the served model,
and calibration of the served model in chases. Usage: python experiment_winprob_vs_served.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from experiment_pitch_and_career_form import CAREER, D, VENUE, prepare
from experiment_team_phase_coverage import TEAM, add_team_phase
from experiment_winprob_team_phase import fit, labels_and_elo
from train_validate_candidate_v3 import V3_FEATURES

KEYS = ["source_file", "innings", "over"]


def main() -> None:
    data = prepare()
    data = data.merge(labels_and_elo(set(data["source_file"])), on=["source_file", "innings"], how="inner", validate="many_to_one")
    data = data.dropna(subset=["bat_won"]).reset_index(drop=True)
    year = pd.to_datetime(data["match_date"]).dt.year
    data = add_team_phase(data, year <= 2024)
    data[TEAM] = data[TEAM].fillna(0.0)
    year = pd.to_datetime(data["match_date"]).dt.year
    train, hold = data[year <= 2024].reset_index(drop=True), data[year >= 2025].reset_index(drop=True)
    elo = ["elo_diff", "elo_bat"]
    models = {
        "in-house: state only": V3_FEATURES,
        "in-house: + coverage, pitch, Elo": V3_FEATURES + TEAM + VENUE + elo,
        "in-house: all (+ batter career rates)": V3_FEATURES + TEAM + VENUE + elo + CAREER,
    }
    for name, cols in models.items():
        hold[name] = fit(train, hold, cols)
    served = pd.read_csv(D / "served_winprob_holdout.csv")
    both = hold.merge(served[KEYS + ["batting_team_won", "served_p"]], on=KEYS, how="inner", validate="one_to_one")
    agree = (both["bat_won"] == both["batting_team_won"]).mean()
    print(f"in-house holdout {len(hold)} rows | served holdout {len(served)} rows | identical rows {len(both)} from {both['source_file'].nunique()} matches "
          f"| labels agree on {agree:.1%}\n")
    both = both[both["bat_won"] == both["batting_team_won"]].reset_index(drop=True)
    y = both["bat_won"].to_numpy()
    preds = {"SERVED match_winner_v1": both["served_p"].to_numpy()}
    for name in models:
        preds[name] = both[name].to_numpy()
    preds["average: served + in-house all"] = (both["served_p"].to_numpy() + both["in-house: all (+ batter career rates)"].to_numpy()) / 2
    preds["average: served + in-house (no batters)"] = (both["served_p"].to_numpy() + both["in-house: + coverage, pitch, Elo"].to_numpy()) / 2
    inn1, chase, ph = (both["innings"] == 1).to_numpy(), (both["innings"] == 2).to_numpy(), both["phase"].to_numpy()
    print(f"{'model':<42}{'Brier':>8}{'logloss':>9}{'AUC':>7}{'inn1':>8}{'chase':>8}{'PP':>7}{'mid':>7}{'death':>7}")
    for name, p in preds.items():
        cells = [brier_score_loss(y, p), log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), roc_auc_score(y, p), brier_score_loss(y[inn1], p[inn1]),
                 brier_score_loss(y[chase], p[chase])] + [brier_score_loss(y[ph == k], p[ph == k]) for k in ("powerplay", "middle", "death")]
        print(f"{name:<42}{cells[0]:8.4f}{cells[1]:9.4f}{cells[2]:7.3f}{cells[3]:8.4f}{cells[4]:8.4f}{cells[5]:7.3f}{cells[6]:7.3f}{cells[7]:7.3f}")
    codes, uniq = pd.factorize(both["source_file"].to_numpy())
    groups = [np.where(codes == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(7)
    boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(150)]
    ref = preds["SERVED match_winner_v1"]
    print("\nBrier change from the SERVED model (negative = better than served), 95% interval over matches:")
    for name, p in preds.items():
        if name.startswith("SERVED"):
            continue
        d = {"all": [], "inn1": [], "chase": []}
        for idx in boots:
            yy = y[idx]
            d["all"].append(((p[idx] - yy) ** 2).mean() - ((ref[idx] - yy) ** 2).mean())
            for key, mask in (("inn1", inn1), ("chase", chase)):
                m = mask[idx]
                d[key].append(((p[idx][m] - yy[m]) ** 2).mean() - ((ref[idx][m] - yy[m]) ** 2).mean())
        ci = lambda v: f"{np.mean(v):+.4f} [{np.percentile(v, 2.5):+.4f},{np.percentile(v, 97.5):+.4f}]"
        print(f"  {name:<42} all {ci(d['all'])} | first innings {ci(d['inn1'])} | chases {ci(d['chase'])}")
    print("\nSERVED model reliability, chases:")
    p = ref[chase]
    bins = np.minimum((p * 10).astype(int), 9)
    for b in range(10):
        sel = bins == b
        if sel.sum() >= 30:
            print(f"  {b * 10}-{b * 10 + 10}%: said {p[sel].mean():.0%} actual {y[chase][sel].mean():.0%} (n={sel.sum()})")


if __name__ == "__main__":
    main()
