"""Do the pitch (venue history) and the batters' CAREER strike rates help the next-over forecast? (user 2026-10-24: "check strike rates, RR, current
RR, pitch and match phase for better prediction".)

Already in the v3 features: current run rate, required run rate, phase, recent-balls rates. Already tested and useless: the batters' strike rate in
THIS innings (experiment_batter_form.py). New here, all computed only from matches played BEFORE the match (no leakage):
  * venue: runs per over at the ground (overall and in this over's phase), average first-innings total, shrunk toward the global figure;
    and the team's current run rate relative to the venue norm.
  * career form: the striker's and non-striker's career strike rate (overall and in this over's phase), shrunk toward the typical T20 rate.
Variants (train <= 2023, holdout 2025+, same LightGBM recipe as the live over outlook, mean objective + big/quiet/wicket classifiers):
base, + venue, + career strike rates, + both. Reports error, AUC, Brier and a match-clustered bootstrap interval on the difference from base.
Usage: python experiment_pitch_and_career_form.py
"""
import json
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from train_validate_candidate_v3 import V3_FEATURES, CATEGORICAL_FEATURES, _features

root = Path(__file__).resolve().parents[1]
D = root / "data/candidates/v3"
CACHE = D / "pitch_career_overs.csv"
PRIOR_RPB = 1.25            # typical T20 runs per ball
BAT_PRIOR_BALLS = 60.0      # career strike rate shrinks toward the typical rate until a batter has faced ~60 balls
VENUE_PRIOR_OVERS = 120.0   # venue figures shrink toward the global figure until ~6 matches of overs exist
BIG, QUIET = 10, 4
KW = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, max_depth=7, min_child_samples=80, subsample=0.85, colsample_bytree=0.9,
          random_state=42, verbose=-1)


def _phase(over: int) -> str:
    return "powerplay" if over <= 6 else "middle" if over <= 15 else "death"


def build() -> pd.DataFrame:
    matches = []
    for scope in ("ipl", "t20i"):
        for path in sorted((root / "data/raw/cricsheet" / scope).glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            info = raw["info"]
            matches.append((str(info["dates"][0]), path.name, str(info.get("venue", "")), raw))
    matches.sort(key=lambda m: (m[0], m[1]))

    bat = defaultdict(lambda: [0, 0])                    # batter -> [runs, balls]
    bat_phase = defaultdict(lambda: [0, 0])              # (batter, phase) -> [runs, balls]
    ven = defaultdict(lambda: [0.0, 0])                  # venue -> [runs, overs]
    ven_phase = defaultdict(lambda: [0.0, 0])            # (venue, phase) -> [runs, overs]
    ven_first = defaultdict(lambda: [0.0, 0])            # venue -> [first-innings total, innings]
    glob = {"runs": 0.0, "overs": 0, "phase": defaultdict(lambda: [0.0, 0]), "first": [0.0, 0]}
    rows = []

    def rate(pair, prior_rate, prior_n):
        return (pair[0] + prior_rate * prior_n) / (pair[1] + prior_n)

    for date, name, venue, raw in matches:
        gl_rpo = glob["runs"] / glob["overs"] if glob["overs"] else 7.8
        gl_first = glob["first"][0] / glob["first"][1] if glob["first"][1] else 150.0
        pending_bat, pending_bat_phase = [], []
        pending_ven = [0.0, 0]
        pending_ven_phase = defaultdict(lambda: [0.0, 0])
        first_total = None
        for index, innings in enumerate(raw.get("innings", []), start=1):
            total = 0
            for over in innings.get("overs", []):
                deliveries = over.get("deliveries", [])
                if not deliveries:
                    continue
                over_no = int(over["over"]) + 1
                phase = _phase(over_no)
                striker, partner = deliveries[0].get("batter"), deliveries[0].get("non_striker")
                gl_phase = glob["phase"][phase][0] / glob["phase"][phase][1] if glob["phase"][phase][1] else gl_rpo
                v_rpo = (ven[venue][0] + gl_rpo * VENUE_PRIOR_OVERS) / (ven[venue][1] + VENUE_PRIOR_OVERS)
                vp = ven_phase[(venue, phase)]
                v_phase_rpo = (vp[0] + gl_phase * VENUE_PRIOR_OVERS) / (vp[1] + VENUE_PRIOR_OVERS)
                vf = ven_first[venue]
                v_first = (vf[0] + gl_first * 6) / (vf[1] + 6)
                row = {"source_file": name, "innings": index, "over": over_no,
                       "venue_rpo": v_rpo, "venue_phase_rpo": v_phase_rpo, "venue_first_avg": v_first, "venue_overs_seen": ven[venue][1]}
                for who, player in (("striker", striker), ("partner", partner)):
                    career = bat[player]
                    in_phase = bat_phase[(player, phase)]
                    row[f"{who}_career_sr"] = rate(career, PRIOR_RPB, BAT_PRIOR_BALLS)
                    row[f"{who}_phase_sr"] = rate(in_phase, rate(career, PRIOR_RPB, BAT_PRIOR_BALLS), BAT_PRIOR_BALLS)
                    row[f"{who}_career_balls"] = career[1]
                rows.append(row)
                runs = sum(int(d["runs"]["total"]) for d in deliveries)
                total += runs
                pending_ven[0] += runs
                pending_ven[1] += 1
                pending_ven_phase[phase][0] += runs
                pending_ven_phase[phase][1] += 1
                for d in deliveries:
                    player = d.get("batter")
                    extras = d.get("extras", {})
                    pending_bat.append((player, int(d["runs"]["batter"]), 0 if "wides" in extras else 1))
                    pending_bat_phase.append(((player, phase), int(d["runs"]["batter"]), 0 if "wides" in extras else 1))
            if index == 1:
                first_total = total
        # update the history only after the whole match (so nothing from this match leaks into its own rows)
        for player, r, b in pending_bat:
            bat[player][0] += r
            bat[player][1] += b
        for key, r, b in pending_bat_phase:
            bat_phase[key][0] += r
            bat_phase[key][1] += b
        ven[venue][0] += pending_ven[0]
        ven[venue][1] += pending_ven[1]
        glob["runs"] += pending_ven[0]
        glob["overs"] += pending_ven[1]
        for phase, (r, n) in pending_ven_phase.items():
            ven_phase[(venue, phase)][0] += r
            ven_phase[(venue, phase)][1] += n
            glob["phase"][phase][0] += r
            glob["phase"][phase][1] += n
        if first_total is not None:
            ven_first[venue][0] += first_total
            ven_first[venue][1] += 1
            glob["first"][0] += first_total
            glob["first"][1] += 1
    return pd.DataFrame(rows)


VENUE = ["venue_rpo", "venue_phase_rpo", "venue_first_avg", "venue_overs_seen", "crr_vs_venue", "crr_vs_venue_phase"]
CAREER = ["striker_career_sr", "partner_career_sr", "striker_phase_sr", "partner_phase_sr", "pair_phase_sr", "striker_career_balls", "partner_career_balls"]


def prepare() -> pd.DataFrame:
    if CACHE.exists():
        extra = pd.read_csv(CACHE)
    else:
        extra = build()
        extra.to_csv(CACHE, index=False)
    frame = pd.read_csv(D / "verified_training_overs.csv")
    data = frame.merge(extra, on=["source_file", "innings", "over"], how="inner", validate="one_to_one")
    print(f"verified rows {len(frame)}, with venue/career features {len(data)}")
    data["pair_phase_sr"] = (data["striker_phase_sr"] + data["partner_phase_sr"]) / 2
    data["crr_vs_venue"] = np.where(data["legal_balls_bowled"] > 0, data["current_run_rate"] - data["venue_rpo"], 0.0)
    data["crr_vs_venue_phase"] = np.where(data["legal_balls_bowled"] > 0, data["current_run_rate"] - data["venue_phase_rpo"], 0.0)
    return data


def fit_predict(train, hold, features):
    Xt, Xh = _features(train, features), _features(hold, features)
    out = {"mean": np.clip(lgb.LGBMRegressor(objective="regression", **KW).fit(Xt, train["runs_in_over"], categorical_feature=CATEGORICAL_FEATURES).predict(Xh), 0, None)}
    for name, target in (("big", train["runs_in_over"] >= BIG), ("quiet", train["runs_in_over"] <= QUIET), ("wicket", train["wicket_in_over"] == 1)):
        out[name] = lgb.LGBMClassifier(**KW).fit(Xt, target.astype(int), categorical_feature=CATEGORICAL_FEATURES).predict_proba(Xh)[:, 1]
    return out


def main() -> None:
    data = prepare()
    year = pd.to_datetime(data["match_date"]).dt.year
    train, hold = data[year <= 2023].reset_index(drop=True), data[year >= 2025].reset_index(drop=True)
    actual = hold["runs_in_over"].to_numpy()
    targets = {"big": actual >= BIG, "quiet": actual <= QUIET, "wicket": hold["wicket_in_over"].to_numpy() == 1}
    variants = {"base (live features)": V3_FEATURES, "+ pitch/venue": V3_FEATURES + VENUE, "+ career strike rates": V3_FEATURES + CAREER,
                "+ both": V3_FEATURES + VENUE + CAREER}
    preds = {name: fit_predict(train, hold, cols) for name, cols in variants.items()}
    print(f"train {len(train)} rows, holdout {len(hold)} rows from {hold['source_file'].nunique()} matches\n")
    print(f"{'variant':<24}{'MAE':>7}{'RMSE':>7}{'corr':>7}{'bigAUC':>8}{'quietAUC':>9}{'wktAUC':>8}")
    for name, p in preds.items():
        print(f"{name:<24}{np.abs(p['mean'] - actual).mean():7.3f}{np.sqrt(((p['mean'] - actual) ** 2).mean()):7.3f}{np.corrcoef(p['mean'], actual)[0, 1]:7.3f}"
              f"{roc_auc_score(targets['big'], p['big']):8.3f}{roc_auc_score(targets['quiet'], p['quiet']):9.3f}{roc_auc_score(targets['wicket'], p['wicket']):8.3f}")
    # match-clustered bootstrap of the change from base
    rng = np.random.default_rng(7)
    matches = hold["source_file"].to_numpy()
    codes, uniq = pd.factorize(matches)
    groups = [np.where(codes == i)[0] for i in range(len(uniq))]
    base = preds["base (live features)"]
    print("\nchange from base, 95% interval over matches (negative RMSE / positive AUC = better):")
    for name, p in preds.items():
        if name.startswith("base"):
            continue
        d_rmse, d_big, d_wkt = [], [], []
        for _ in range(200):
            idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            a = actual[idx]
            d_rmse.append(np.sqrt(((p["mean"][idx] - a) ** 2).mean()) - np.sqrt(((base["mean"][idx] - a) ** 2).mean()))
            d_big.append(roc_auc_score(a >= BIG, p["big"][idx]) - roc_auc_score(a >= BIG, base["big"][idx]))
            w = hold["wicket_in_over"].to_numpy()[idx] == 1
            d_wkt.append(roc_auc_score(w, p["wicket"][idx]) - roc_auc_score(w, base["wicket"][idx]))
        ci = lambda v: f"{np.mean(v):+.4f} [{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}]"
        print(f"  {name:<24} RMSE {ci(d_rmse)} | bigAUC {ci(d_big)} | wicketAUC {ci(d_wkt)}")


if __name__ == "__main__":
    main()
