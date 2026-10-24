"""Current strike rate as a correction layer, weighted by RUNS scored in the innings instead of balls faced (user 2026-10-24: "give weightage after
each run"). Same stage 1 (career-form model, out-of-fold on the training years) and holdout as experiment_current_sr_layer.py. Layer variants,
for striker and non-striker, all computed before the over:
  * balls-weighted (the earlier layer): balls / (balls + 20) x (innings SR - career SR),
  * runs-weighted: runs / (runs + R) x (innings SR - career SR), R = 10, 20, 40,
  * excess runs: runs scored minus what the career rate predicts for the balls faced -- it moves with every run (and every dot ball).
Usage: python experiment_current_sr_runs_weight.py
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from experiment_pitch_and_career_form import BIG, CAREER, D, QUIET, fit_predict, prepare
from experiment_current_sr_layer import logit
from train_validate_candidate_v3 import V3_FEATURES


def layer_columns(data: pd.DataFrame) -> dict[str, list[str]]:
    sets = {}
    for who in ("striker", "partner"):
        runs, balls, career = data[f"{who}_runs"], data[f"{who}_balls"], data[f"{who}_career_sr"]
        gap = (runs / balls.clip(lower=1) - career) * (balls > 0)
        data[f"{who}_w_balls20"] = balls / (balls + 20) * gap
        for r in (10, 20, 40):
            data[f"{who}_w_runs{r}"] = runs / (runs + r) * gap
        data[f"{who}_excess"] = runs - career * balls
    for name in ("w_balls20", "w_runs10", "w_runs20", "w_runs40", "excess"):
        sets[name] = [f"striker_{name}", f"partner_{name}"]
    return sets


def main() -> None:
    data = prepare()
    form = pd.read_csv(D / "batter_form_overs.csv")
    data = data.merge(form, on=["source_file", "innings", "over"], how="inner", validate="one_to_one")
    sets = layer_columns(data)
    year = pd.to_datetime(data["match_date"]).dt.year
    train, hold = data[year <= 2023].reset_index(drop=True), data[year >= 2025].reset_index(drop=True)
    features = V3_FEATURES + CAREER
    print("stage 1 ...", flush=True)
    s1 = fit_predict(train, hold, features)
    oof = {k: np.zeros(len(train)) for k in s1}
    for fit_idx, val_idx in GroupKFold(n_splits=4).split(train, groups=train["source_file"]):
        part = fit_predict(train.iloc[fit_idx], train.iloc[val_idx], features)
        for k in oof:
            oof[k][val_idx] = part[k]
    y_tr, y_ho = train["runs_in_over"].to_numpy(), hold["runs_in_over"].to_numpy()
    wicket = hold["wicket_in_over"].to_numpy() == 1
    targets = {"big": y_tr >= BIG, "quiet": y_tr <= QUIET, "wicket": train["wicket_in_over"].to_numpy() == 1}
    resid = y_tr - oof["mean"]
    results = {"stage 1 (career form)": dict(s1)}
    for name, cols in sets.items():
        p = dict(s1)
        g = lgb.LGBMRegressor(objective="regression", n_estimators=150, learning_rate=0.03, num_leaves=8, min_child_samples=400, subsample=0.8,
                              colsample_bytree=0.9, random_state=42, verbose=-1)
        g.fit(np.column_stack([train[cols].to_numpy(), oof["mean"], train["over"]]), resid)
        p["mean"] = np.clip(s1["mean"] + g.predict(np.column_stack([hold[cols].to_numpy(), s1["mean"], hold["over"]])), 0, None)
        scale = np.abs(train[cols].to_numpy()).std()
        for k in targets:
            lr = LogisticRegression(C=1.0, max_iter=500).fit(np.column_stack([logit(oof[k]), train[cols].to_numpy() / scale]), targets[k])
            p[k] = lr.predict_proba(np.column_stack([logit(s1[k]), hold[cols].to_numpy() / scale]))[:, 1]
        results[f"layer: {name}"] = p
    print(f"\ntrain {len(train)}, holdout {len(hold)} rows / {hold['source_file'].nunique()} matches\n")
    print(f"{'variant':<28}{'RMSE':>7}{'bigAUC':>8}{'quietAUC':>9}{'wktAUC':>8}")
    for name, p in results.items():
        print(f"{name:<28}{np.sqrt(((p['mean'] - y_ho) ** 2).mean()):7.3f}{roc_auc_score(y_ho >= BIG, p['big']):8.4f}"
              f"{roc_auc_score(y_ho <= QUIET, p['quiet']):9.4f}{roc_auc_score(wicket, p['wicket']):8.4f}")
    codes, uniq = pd.factorize(hold["source_file"].to_numpy())
    groups = [np.where(codes == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(7)
    ref = results["stage 1 (career form)"]
    print("\nchange from stage 1, 95% interval over matches:")
    for name, p in results.items():
        if name.startswith("stage 1"):
            continue
        d_r, d_b = [], []
        for _ in range(120):
            idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            a = y_ho[idx]
            d_r.append(np.sqrt(((p["mean"][idx] - a) ** 2).mean()) - np.sqrt(((ref["mean"][idx] - a) ** 2).mean()))
            d_b.append(roc_auc_score(a >= BIG, p["big"][idx]) - roc_auc_score(a >= BIG, ref["big"][idx]))
        ci = lambda v: f"{np.mean(v):+.4f} [{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}]"
        print(f"  {name:<24} RMSE {ci(d_r)} | bigAUC {ci(d_b)}")


if __name__ == "__main__":
    main()
