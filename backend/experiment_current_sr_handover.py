"""Should a batter's strike rate IN THIS INNINGS take over from his career strike rate once he has faced enough balls? (user 2026-10-24: "current SR
should be used after certain balls".)

experiment_pitch_and_career_form.py found the career strike rate helps; experiment_batter_form.py found the in-innings rate alone does not (it was
shrunk toward a typical rate, not toward the batter's own). Here the two are joined, all before the over so nothing leaks:
  * career only (the reference),
  * career + raw in-innings runs/balls (the model learns the hand-over itself),
  * blended: (innings runs + career rate x K) / (innings balls + K) -- the in-innings rate takes over smoothly, K balls of career weight (K = 10, 20, 40),
  * switch: the plain in-innings rate once the batter has faced N balls (N = 10, 20), the career rate before that.
Same split and recipe as the other experiments (train <= 2023, holdout 2025+). Reports error and AUC, and the change from "career only" with a
match-clustered bootstrap interval. Usage: python experiment_current_sr_handover.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiment_pitch_and_career_form import BIG, CAREER, D, QUIET, fit_predict, prepare
from train_validate_candidate_v3 import V3_FEATURES

RAW = ["striker_runs", "striker_balls", "partner_runs", "partner_balls"]


def add_handover(data: pd.DataFrame) -> pd.DataFrame:
    for who in ("striker", "partner"):
        runs, balls, career = data[f"{who}_runs"], data[f"{who}_balls"], data[f"{who}_career_sr"]
        for k in (10, 20, 40):
            data[f"{who}_blend{k}"] = (runs + career * k) / (balls + k)
        for n in (10, 20):
            data[f"{who}_switch{n}"] = np.where(balls >= n, runs / balls.clip(lower=1), career)
    for name in ("blend10", "blend20", "blend40", "switch10", "switch20"):
        data[f"pair_{name}"] = (data[f"striker_{name}"] + data[f"partner_{name}"]) / 2
    return data


def main() -> None:
    data = prepare()
    form = pd.read_csv(D / "batter_form_overs.csv")
    data = add_handover(data.merge(form, on=["source_file", "innings", "over"], how="inner", validate="one_to_one"))
    year = pd.to_datetime(data["match_date"]).dt.year
    train, hold = data[year <= 2023].reset_index(drop=True), data[year >= 2025].reset_index(drop=True)
    actual = hold["runs_in_over"].to_numpy()
    wicket = hold["wicket_in_over"].to_numpy() == 1
    variants = {"career only": V3_FEATURES + CAREER,
                "career + raw in-innings": V3_FEATURES + CAREER + RAW}
    for name in ("blend10", "blend20", "blend40", "switch10", "switch20"):
        variants[f"career + {name}"] = V3_FEATURES + CAREER + [f"striker_{name}", f"partner_{name}", f"pair_{name}"] + RAW[1::2]
    preds = {name: fit_predict(train, hold, cols) for name, cols in variants.items()}
    print(f"train {len(train)} rows, holdout {len(hold)} rows from {hold['source_file'].nunique()} matches\n")
    print(f"{'variant':<26}{'RMSE':>7}{'corr':>7}{'bigAUC':>8}{'quietAUC':>9}{'wktAUC':>8}")
    for name, p in preds.items():
        print(f"{name:<26}{np.sqrt(((p['mean'] - actual) ** 2).mean()):7.3f}{np.corrcoef(p['mean'], actual)[0, 1]:7.3f}"
              f"{roc_auc_score(actual >= BIG, p['big']):8.3f}{roc_auc_score(actual <= QUIET, p['quiet']):9.3f}{roc_auc_score(wicket, p['wicket']):8.3f}")
    codes, uniq = pd.factorize(hold["source_file"].to_numpy())
    groups = [np.where(codes == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(7)
    ref = preds["career only"]
    print("\nchange from 'career only', 95% interval over matches (negative RMSE / positive AUC = better):")
    for name, p in preds.items():
        if name == "career only":
            continue
        d_rmse, d_big = [], []
        for _ in range(150):
            idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            a = actual[idx]
            d_rmse.append(np.sqrt(((p["mean"][idx] - a) ** 2).mean()) - np.sqrt(((ref["mean"][idx] - a) ** 2).mean()))
            d_big.append(roc_auc_score(a >= BIG, p["big"][idx]) - roc_auc_score(a >= BIG, ref["big"][idx]))
        ci = lambda v: f"{np.mean(v):+.4f} [{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}]"
        print(f"  {name:<26} RMSE {ci(d_rmse)} | bigAUC {ci(d_big)}")
    # how many balls does a batter usually have faced when the over starts?
    balls = hold["striker_balls"]
    print(f"\nstriker has faced >=10 balls at the start of {np.mean(balls >= 10):.0%} of overs, >=20 in {np.mean(balls >= 20):.0%}")


if __name__ == "__main__":
    main()
