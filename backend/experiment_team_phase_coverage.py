"""Team performance split by phase, how far the team has "covered itself" against a normal path, then the batters' strike rates patched over it
(user 2026-10-24: "divide a team performance in phases ... how many team has covered itself in those phases like current RR, score, wickets in hand,
pitch ... same for chases ... and then SR, current SR and patching it over").

Norm tables come from the TRAINING years only (<= 2023). Per over, before the over is bowled:
  * expected remaining runs from this state (over, wickets down, chase or not) -- the resources left, a Duckworth-Lewis-style idea -- and the projected total;
  * ahead of / behind par: score now minus the usual score at this over, and projected total against the ground's usual first-innings total;
  * per phase (powerplay 1-6, middle 7-15, death 16-20): runs and wickets so far in the CURRENT phase against the phase norm for the overs gone, and the
    same for the PREVIOUS phase (how the team did in the phase just finished);
  * chases: expected remaining runs divided by runs needed (coverage > 1 = comfortable), the gap, and the first innings total against the ground's usual
    (the pitch signal: a big total on a ground that usually scores less means a good pitch or a strong side).
Variants (holdout 2025+): stage 1 = the career-form model; A = + these team-phase inputs as features; B = a correction LAYER fitted on stage 1's
out-of-fold errors using the team-phase inputs; C = layer with the team-phase inputs AND the batters' strike-rate gaps (career vs this innings).
Usage: python experiment_team_phase_coverage.py
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from experiment_current_sr_layer import logit
from experiment_pitch_and_career_form import BIG, CAREER, D, QUIET, fit_predict, prepare
from train_validate_candidate_v3 import V3_FEATURES

PHASE_OF = lambda o: 0 if o <= 6 else 1 if o <= 15 else 2
PHASE_START = {0: 1, 1: 7, 2: 16}
PHASE_OVERS = {0: 6, 1: 9, 2: 5}
TEAM = ["par_remaining", "proj_total", "ahead_of_par", "proj_vs_venue", "cur_phase_vs_norm", "prev_phase_vs_norm", "cur_phase_wkts", "prev_phase_wkts",
        "chase_coverage", "chase_gap", "first_total_vs_venue"]


def add_team_phase(data: pd.DataFrame, fit_rows: pd.Series) -> pd.DataFrame:
    data = data.copy()
    keys = ["source_file", "innings"]
    final = (data["score_before_over"] + data["runs_in_over"]).groupby([data[k] for k in keys]).transform("max")
    data["remaining"] = final - data["score_before_over"]
    ref = data[fit_rows]
    # expected remaining runs by (chase, over, wickets down), shrunk toward the (chase, over) mean
    by_over = ref.groupby(["is_chase", "over"])["remaining"].agg(["mean"]).rename(columns={"mean": "over_mean"})
    cell = ref.groupby(["is_chase", "over", "wkts_down_before_over"])["remaining"].agg(["sum", "count"])
    cell = cell.join(by_over, on=["is_chase", "over"])
    cell["exp_remaining"] = (cell["sum"] + 50 * cell["over_mean"]) / (cell["count"] + 50)
    data = data.join(cell["exp_remaining"], on=["is_chase", "over", "wkts_down_before_over"])
    data["par_remaining"] = data["exp_remaining"].fillna(data.join(by_over, on=["is_chase", "over"])["over_mean"])
    par_score = ref.groupby(["is_chase", "over"])["score_before_over"].mean().rename("par_score")
    data = data.join(par_score, on=["is_chase", "over"])
    data["proj_total"] = data["score_before_over"] + data["par_remaining"]
    data["ahead_of_par"] = data["score_before_over"] - data["par_score"]
    data["proj_vs_venue"] = data["proj_total"] - data["venue_first_avg"]
    # phase-by-phase
    norm = {}
    for ph in range(3):
        sel = ref[ref["over"].map(PHASE_OF) == ph]
        norm[(ph, 0)] = sel[sel.is_chase == 0]["runs_in_over"].mean()
        norm[(ph, 1)] = sel[sel.is_chase == 1]["runs_in_over"].mean()
    start_score, start_wkts = {}, {}
    for ph in (1, 2):
        sub = data[data["over"] == PHASE_START[ph]].set_index(keys)
        start_score[ph], start_wkts[ph] = sub["score_before_over"], sub["wkts_down_before_over"]
    idx = pd.MultiIndex.from_frame(data[keys])
    phase = data["over"].map(PHASE_OF)
    s_start = np.where(phase == 0, 0.0, np.where(phase == 1, pd.Series(idx.map(start_score[1]), index=data.index), pd.Series(idx.map(start_score[2]), index=data.index)))
    w_start = np.where(phase == 0, 0.0, np.where(phase == 1, pd.Series(idx.map(start_wkts[1]), index=data.index), pd.Series(idx.map(start_wkts[2]), index=data.index)))
    s_start = pd.Series(s_start, index=data.index).astype(float).fillna(data["score_before_over"])
    w_start = pd.Series(w_start, index=data.index).astype(float).fillna(data["wkts_down_before_over"])
    elapsed = data["over"] - phase.map(PHASE_START)
    norm_cur = pd.Series([norm[(p, c)] for p, c in zip(phase, data["is_chase"])], index=data.index)
    data["cur_phase_vs_norm"] = (data["score_before_over"] - s_start) - elapsed * norm_cur
    data["cur_phase_wkts"] = data["wkts_down_before_over"] - w_start
    # previous phase (how the phase just finished went): middle rows look at the powerplay, death rows at the middle
    prev_runs = np.where(phase == 1, s_start, np.where(phase == 2, s_start - pd.Series(idx.map(start_score[1]), index=data.index).astype(float), 0.0))
    prev_wk = np.where(phase == 1, w_start, np.where(phase == 2, w_start - pd.Series(idx.map(start_wkts[1]), index=data.index).astype(float), 0.0))
    prev_phase = (phase - 1).clip(lower=0)
    prev_norm = pd.Series([norm[(p, c)] * PHASE_OVERS[p] for p, c in zip(prev_phase, data["is_chase"])], index=data.index)
    data["prev_phase_vs_norm"] = np.where(phase == 0, 0.0, prev_runs - prev_norm)
    data["prev_phase_wkts"] = np.where(phase == 0, 0.0, prev_wk)
    chase = data["is_chase"] == 1
    data["chase_coverage"] = np.where(chase & (data["runs_required"] > 0), data["par_remaining"] / data["runs_required"].clip(lower=1), 0.0)
    data["chase_gap"] = np.where(chase, data["par_remaining"] - data["runs_required"], 0.0)
    first_total = data["score_before_over"] + data["runs_required"] - 1
    data["first_total_vs_venue"] = np.where(chase, first_total - data["venue_first_avg"], 0.0)
    return data


def main() -> None:
    data = prepare()
    form = pd.read_csv(D / "batter_form_overs.csv")
    data = data.merge(form, on=["source_file", "innings", "over"], how="inner", validate="one_to_one")
    year = pd.to_datetime(data["match_date"]).dt.year
    data = add_team_phase(data, year <= 2023)
    for who in ("striker", "partner"):
        balls = data[f"{who}_balls"]
        data[f"dev_{who}"] = balls / (balls + 20) * (data[f"{who}_runs"] / balls.clip(lower=1) - data[f"{who}_career_sr"]) * (balls > 0)
    BAT = ["dev_striker", "dev_partner"]
    data[TEAM] = data[TEAM].fillna(0.0)
    train, hold = data[year <= 2023].reset_index(drop=True), data[year >= 2025].reset_index(drop=True)
    features = V3_FEATURES + CAREER
    print("stage 1 ...", flush=True)
    s1 = fit_predict(train, hold, features)
    featA = fit_predict(train, hold, features + TEAM)
    oof = {k: np.zeros(len(train)) for k in s1}
    for fit_idx, val_idx in GroupKFold(n_splits=4).split(train, groups=train["source_file"]):
        part = fit_predict(train.iloc[fit_idx], train.iloc[val_idx], features)
        for k in oof:
            oof[k][val_idx] = part[k]
    y_tr, y_ho = train["runs_in_over"].to_numpy(), hold["runs_in_over"].to_numpy()
    wicket = hold["wicket_in_over"].to_numpy() == 1
    targets = {"big": y_tr >= BIG, "quiet": y_tr <= QUIET, "wicket": train["wicket_in_over"].to_numpy() == 1}

    def layer(cols):
        p = dict(s1)
        g = lgb.LGBMRegressor(objective="regression", n_estimators=200, learning_rate=0.03, num_leaves=12, min_child_samples=300, subsample=0.8,
                              colsample_bytree=0.9, random_state=42, verbose=-1)
        g.fit(np.column_stack([train[cols].to_numpy(), oof["mean"], train["over"]]), y_tr - oof["mean"])
        p["mean"] = np.clip(s1["mean"] + g.predict(np.column_stack([hold[cols].to_numpy(), s1["mean"], hold["over"]])), 0, None)
        mu, sd = train[cols].mean().to_numpy(), train[cols].std().replace(0, 1).to_numpy()
        for k in targets:
            lr = LogisticRegression(C=0.5, max_iter=800).fit(np.column_stack([logit(oof[k]), (train[cols].to_numpy() - mu) / sd]), targets[k])
            p[k] = lr.predict_proba(np.column_stack([logit(s1[k]), (hold[cols].to_numpy() - mu) / sd]))[:, 1]
        return p

    results = {"stage 1 (career form)": s1, "A: + team-phase inputs as features": featA, "B: layer, team-phase inputs": layer(TEAM),
               "C: layer, team-phase + batter strike-rate gaps": layer(TEAM + BAT)}
    chase_h = (hold["is_chase"] == 1).to_numpy()
    print(f"\ntrain {len(train)}, holdout {len(hold)} overs / {hold['source_file'].nunique()} matches ({chase_h.sum()} in chases)\n")
    print(f"{'variant':<48}{'RMSE':>7}{'chaseRMSE':>10}{'bigAUC':>8}{'quietAUC':>9}{'wktAUC':>8}")
    for name, p in results.items():
        print(f"{name:<48}{np.sqrt(((p['mean'] - y_ho) ** 2).mean()):7.3f}{np.sqrt(((p['mean'][chase_h] - y_ho[chase_h]) ** 2).mean()):10.3f}"
              f"{roc_auc_score(y_ho >= BIG, p['big']):8.4f}{roc_auc_score(y_ho <= QUIET, p['quiet']):9.4f}{roc_auc_score(wicket, p['wicket']):8.4f}")
    codes, uniq = pd.factorize(hold["source_file"].to_numpy())
    groups = [np.where(codes == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(7)
    boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(120)]
    print("\nchange from stage 1, 95% interval over matches (negative RMSE / positive AUC = better):")
    for name, p in results.items():
        if name.startswith("stage 1"):
            continue
        d = {"RMSE": [], "big": [], "quiet": [], "wicket": []}
        for idx in boots:
            a, w = y_ho[idx], wicket[idx]
            d["RMSE"].append(np.sqrt(((p["mean"][idx] - a) ** 2).mean()) - np.sqrt(((s1["mean"][idx] - a) ** 2).mean()))
            d["big"].append(roc_auc_score(a >= BIG, p["big"][idx]) - roc_auc_score(a >= BIG, s1["big"][idx]))
            d["quiet"].append(roc_auc_score(a <= QUIET, p["quiet"][idx]) - roc_auc_score(a <= QUIET, s1["quiet"][idx]))
            d["wicket"].append(roc_auc_score(w, p["wicket"][idx]) - roc_auc_score(w, s1["wicket"][idx]))
        ci = lambda v: f"{np.mean(v):+.4f} [{np.percentile(v, 2.5):+.4f},{np.percentile(v, 97.5):+.4f}]"
        print(f"  {name}\n     RMSE {ci(d['RMSE'])} | big {ci(d['big'])} | quiet {ci(d['quiet'])} | wicket {ci(d['wicket'])}")


if __name__ == "__main__":
    main()
