"""Recalibrate the SERVED win probability? (user 2026-10-24, after experiment_winprob_vs_served.py found the served model slightly cautious in chases: said
35% -> actual 40%, 75% -> 80%.) The served calibrator is ONE Platt fit on 2024 for every over of both innings. Candidates fitted on 2024 (the calibration
year, out of sample for the model) and scored on 2025+:
  * served as is;
  * Platt on the raw score, separately for the first innings and the chase;
  * isotonic on the served probability, separately for the first innings and the chase;
  * Platt with phase: [logit, chase, logit x chase, phase, logit x phase];
  * the same, fitted on 2024 AND the first half (by date) of 2025+, scored on the second half only (a drift check), against the served model on those rows.
Match-clustered bootstrap for the Brier change from the served model. Usage: python experiment_winprob_recalibration.py
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from experiment_pitch_and_career_form import D


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def design(frame: pd.DataFrame, source: str) -> np.ndarray:
    z = logit(frame[source].to_numpy())
    chase = (frame["innings"] == 2).to_numpy().astype(float)
    over = frame["over"].to_numpy()
    pp, mid = (over <= 6).astype(float), ((over > 6) & (over <= 15)).astype(float)
    return np.column_stack([z, chase, z * chase, pp, mid, z * pp, z * mid])


def recalibrators(fit: pd.DataFrame):
    """Returns {name: function(frame) -> probabilities}, all fitted on `fit`."""
    out = {"served as is": lambda f: f["served_p"].to_numpy()}
    platt = {}
    for chase in (False, True):
        part = fit[(fit["innings"] == 2) == chase]
        platt[chase] = LogisticRegression(C=1.0, max_iter=500).fit(logit(part[["served_raw_p"]].to_numpy()), part["batting_team_won"])
    iso = {}
    for chase in (False, True):
        part = fit[(fit["innings"] == 2) == chase]
        iso[chase] = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(part["served_p"], part["batting_team_won"])

    def per_innings(models, source, isotonic=False):
        def apply(f):
            p = np.zeros(len(f))
            for chase in (False, True):
                m = ((f["innings"] == 2) == chase).to_numpy()
                if m.any():
                    p[m] = models[chase].predict(f.loc[m, source]) if isotonic else models[chase].predict_proba(logit(f.loc[m, [source]].to_numpy()))[:, 1]
            return p
        return apply

    out["Platt per innings (on raw score)"] = per_innings(platt, "served_raw_p")
    out["isotonic per innings (on served probability)"] = per_innings(iso, "served_p", isotonic=True)
    phase_model = LogisticRegression(C=1.0, max_iter=1000).fit(design(fit, "served_raw_p"), fit["batting_team_won"])
    out["Platt with chase and phase"] = lambda f: phase_model.predict_proba(design(f, "served_raw_p"))[:, 1]
    return out


def report(title: str, frame: pd.DataFrame, preds: dict[str, np.ndarray]) -> None:
    y = frame["batting_team_won"].to_numpy()
    inn1, chase = (frame["innings"] == 1).to_numpy(), (frame["innings"] == 2).to_numpy()
    print(f"\n{title}: {len(frame)} overs from {frame['source_file'].nunique()} matches")
    print(f"{'model':<48}{'Brier':>8}{'logloss':>9}{'AUC':>7}{'inn1':>8}{'chase':>8}")
    for name, p in preds.items():
        print(f"{name:<48}{brier_score_loss(y, p):8.4f}{log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)):9.4f}{roc_auc_score(y, p):7.3f}"
              f"{brier_score_loss(y[inn1], p[inn1]):8.4f}{brier_score_loss(y[chase], p[chase]):8.4f}")
    codes, uniq = pd.factorize(frame["source_file"].to_numpy())
    groups = [np.where(codes == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(7)
    boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(150)]
    ref = preds["served as is"]
    for name, p in preds.items():
        if name == "served as is":
            continue
        d = {"all": [], "inn1": [], "chase": []}
        for idx in boots:
            yy = y[idx]
            d["all"].append(((p[idx] - yy) ** 2).mean() - ((ref[idx] - yy) ** 2).mean())
            for key, mask in (("inn1", inn1), ("chase", chase)):
                m = mask[idx]
                d[key].append(((p[idx][m] - yy[m]) ** 2).mean() - ((ref[idx][m] - yy[m]) ** 2).mean())
        ci = lambda v: f"{np.mean(v):+.4f} [{np.percentile(v, 2.5):+.4f},{np.percentile(v, 97.5):+.4f}]"
        print(f"   change vs served | {name:<44} all {ci(d['all'])} | inn1 {ci(d['inn1'])} | chase {ci(d['chase'])}")


def main() -> None:
    served = pd.read_csv(D / "served_winprob_2024plus.csv")
    served["year"] = pd.to_datetime(served["match_date"]).dt.year
    fit, hold = served[served["year"] == 2024].reset_index(drop=True), served[served["year"] >= 2025].reset_index(drop=True)
    print(f"2024 (fit): {len(fit)} overs | 2025+ (test): {len(hold)} overs")
    cal = recalibrators(fit)
    report("2025+ test, recalibrators fitted on 2024", hold, {name: fn(hold) for name, fn in cal.items()})
    # drift check: fit on 2024 + the earlier half (by date) of 2025+, test on the later half
    dates = hold.groupby("source_file")["match_date"].first().sort_values()
    early = set(dates.index[: len(dates) // 2])
    fit2 = pd.concat([fit, hold[hold["source_file"].isin(early)]]).reset_index(drop=True)
    late = hold[~hold["source_file"].isin(early)].reset_index(drop=True)
    cal2 = recalibrators(fit2)
    report("drift check: fitted on 2024 + earlier half of 2025+, tested on the later half", late, {name: fn(late) for name, fn in cal2.items()})
    chase = hold[hold["innings"] == 2]
    best = cal["Platt per innings (on raw score)"](hold)[(hold["innings"] == 2).to_numpy()]
    print("\nchase reliability, served vs Platt per innings (2025+):")
    for label, p in (("served", chase["served_p"].to_numpy()), ("Platt per innings", best)):
        y = chase["batting_team_won"].to_numpy()
        bins = np.minimum((p * 10).astype(int), 9)
        print("  " + label + ": " + " | ".join(f"{b * 10}-{b * 10 + 10}%: {p[bins == b].mean():.0%}/{y[bins == b].mean():.0%}" for b in range(10) if (bins == b).sum() >= 30))


if __name__ == "__main__":
    main()
