"""Use the batters' strike rate IN THIS INNINGS as a correction layer on top of the career-form forecast, not as a hand-over (user 2026-10-24:
"instead of handover use as a layer").

Stage 1 is the career-form model from experiment_pitch_and_career_form.py (v3 features + career strike rates). Its out-of-fold predictions on
the training years (folds split by match) are what stage 2 learns from, so the layer is fitted on honest errors. Stage 2 sees only what the
batters are doing NOW relative to what they usually do: for striker and non-striker, w x (innings strike rate - career strike rate) with
w = balls / (balls + 20) so a batter with few balls counts little, plus the balls faced.
  * mean runs: (a) ridge on the stage-1 residual, (b) a small GBM on the residual,
  * big / quiet / wicket chance: logistic regression on [logit(stage-1 chance), the layer features].
Holdout 2025+, change from stage 1 with a match-clustered bootstrap interval. Usage: python experiment_current_sr_layer.py
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from experiment_pitch_and_career_form import BIG, CAREER, D, QUIET, fit_predict, prepare
from train_validate_candidate_v3 import V3_FEATURES

W_BALLS = 20.0
LAYER = ["dev_striker", "dev_partner", "dev_pair"]
LAYER_GBM = LAYER + ["striker_balls", "partner_balls", "stage1_mean", "over"]


def add_layer(data: pd.DataFrame) -> pd.DataFrame:
    for who in ("striker", "partner"):
        balls = data[f"{who}_balls"]
        innings_sr = data[f"{who}_runs"] / balls.clip(lower=1)
        data[f"dev_{who}"] = balls / (balls + W_BALLS) * (innings_sr - data[f"{who}_career_sr"]) * (balls > 0)
    data["dev_pair"] = (data["dev_striker"] + data["dev_partner"]) / 2
    return data


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def main() -> None:
    data = prepare()
    form = pd.read_csv(D / "batter_form_overs.csv")
    data = add_layer(data.merge(form, on=["source_file", "innings", "over"], how="inner", validate="one_to_one"))
    year = pd.to_datetime(data["match_date"]).dt.year
    train, hold = data[year <= 2023].reset_index(drop=True), data[year >= 2025].reset_index(drop=True)
    features = V3_FEATURES + CAREER

    print("stage 1 on holdout ...", flush=True)
    s1_hold = fit_predict(train, hold, features)
    print("stage 1 out-of-fold on training years (4 folds by match) ...", flush=True)
    oof = {k: np.zeros(len(train)) for k in ("mean", "big", "quiet", "wicket")}
    for fit_idx, val_idx in GroupKFold(n_splits=4).split(train, groups=train["source_file"]):
        part = fit_predict(train.iloc[fit_idx], train.iloc[val_idx], features)
        for k in oof:
            oof[k][val_idx] = part[k]

    train["stage1_mean"], hold["stage1_mean"] = oof["mean"], s1_hold["mean"]
    y_tr, y_ho = train["runs_in_over"].to_numpy(), hold["runs_in_over"].to_numpy()
    resid = y_tr - oof["mean"]
    results = {"stage 1 (career form)": dict(s1_hold)}
    results["+ layer: ridge on residual"] = dict(s1_hold)
    results["+ layer: ridge on residual"]["mean"] = np.clip(s1_hold["mean"] + Ridge(alpha=10.0).fit(train[LAYER[:2]], resid).predict(hold[LAYER[:2]]), 0, None)
    gbm = lgb.LGBMRegressor(objective="regression", n_estimators=150, learning_rate=0.03, num_leaves=8, min_child_samples=400,
                            subsample=0.8, colsample_bytree=0.9, random_state=42, verbose=-1).fit(train[LAYER_GBM], resid)
    results["+ layer: small GBM on residual"] = dict(s1_hold)
    results["+ layer: small GBM on residual"]["mean"] = np.clip(s1_hold["mean"] + gbm.predict(hold[LAYER_GBM]), 0, None)
    layered = dict(s1_hold)
    targets_tr = {"big": y_tr >= BIG, "quiet": y_tr <= QUIET, "wicket": train["wicket_in_over"].to_numpy() == 1}
    for k in ("big", "quiet", "wicket"):
        Xtr = np.column_stack([logit(oof[k]), train[LAYER[:2]].to_numpy()])
        Xho = np.column_stack([logit(s1_hold[k]), hold[LAYER[:2]].to_numpy()])
        lr = LogisticRegression(C=1.0, max_iter=500).fit(Xtr, targets_tr[k])
        layered[k] = lr.predict_proba(Xho)[:, 1]
        print(f"  {k}: layer weights on [stage-1 logit, striker dev, partner dev] = {np.round(lr.coef_[0], 3)}")
    layered["mean"] = results["+ layer: small GBM on residual"]["mean"]
    results["+ layer: all (GBM mean + logistic chances)"] = layered

    wicket = hold["wicket_in_over"].to_numpy() == 1
    print(f"\ntrain {len(train)} rows, holdout {len(hold)} rows from {hold['source_file'].nunique()} matches\n")
    print(f"{'variant':<44}{'RMSE':>7}{'corr':>7}{'bigAUC':>8}{'quietAUC':>9}{'wktAUC':>8}")
    for name, p in results.items():
        print(f"{name:<44}{np.sqrt(((p['mean'] - y_ho) ** 2).mean()):7.3f}{np.corrcoef(p['mean'], y_ho)[0, 1]:7.3f}"
              f"{roc_auc_score(y_ho >= BIG, p['big']):8.3f}{roc_auc_score(y_ho <= QUIET, p['quiet']):9.3f}{roc_auc_score(wicket, p['wicket']):8.3f}")
    codes, uniq = pd.factorize(hold["source_file"].to_numpy())
    groups = [np.where(codes == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(7)
    ref = results["stage 1 (career form)"]
    print("\nchange from stage 1, 95% interval over matches (negative RMSE / positive AUC = better):")
    for name, p in results.items():
        if name.startswith("stage 1"):
            continue
        d_rmse, d_big, d_quiet, d_wkt = [], [], [], []
        for _ in range(150):
            idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            a, w = y_ho[idx], wicket[idx]
            d_rmse.append(np.sqrt(((p["mean"][idx] - a) ** 2).mean()) - np.sqrt(((ref["mean"][idx] - a) ** 2).mean()))
            d_big.append(roc_auc_score(a >= BIG, p["big"][idx]) - roc_auc_score(a >= BIG, ref["big"][idx]))
            d_quiet.append(roc_auc_score(a <= QUIET, p["quiet"][idx]) - roc_auc_score(a <= QUIET, ref["quiet"][idx]))
            d_wkt.append(roc_auc_score(w, p["wicket"][idx]) - roc_auc_score(w, ref["wicket"][idx]))
        ci = lambda v: f"{np.mean(v):+.4f} [{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}]"
        print(f"  {name}\n     RMSE {ci(d_rmse)} | bigAUC {ci(d_big)} | quietAUC {ci(d_quiet)} | wicketAUC {ci(d_wkt)}")


if __name__ == "__main__":
    main()
