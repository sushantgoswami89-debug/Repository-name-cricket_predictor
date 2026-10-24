"""Do the team-phase "coverage" inputs, the pitch, the batters' career rates and a team-strength rating improve the WIN PROBABILITY? (user
2026-10-24: "run it on win probability".) A clean in-house baseline on the verified men's T20 overs (not the served model, whose feature pipeline is
much heavier): label = the batting side wins the match (ties / no results dropped); trained on <= 2024, tested on 2025+.
Groups added one at a time to the match-state baseline, then together:
  * state (baseline): over, score, wickets, balls, run rates, chase pressure, last-2-overs form;
  * team-phase coverage (experiment_team_phase_coverage.py): expected remaining runs, ahead of par, phase vs norm, chase coverage/gap, first total vs ground;
  * pitch: the ground's usual runs per over, per phase, and first-innings average;
  * team strength: an Elo rating of each side from earlier matches (batting minus bowling) -- what Cricbuzz's early overs seem to use;
  * batters: career strike rates of the two batters at the crease.
Reports Brier, log loss, AUC overall, first innings, chases and by phase; a match-clustered bootstrap interval on the Brier change from baseline; and a
reliability table for the combined model. Usage: python experiment_winprob_team_phase.py
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from experiment_pitch_and_career_form import CAREER, VENUE, prepare
from experiment_team_phase_coverage import TEAM, add_team_phase
from train_validate_candidate_v3 import CATEGORICAL_FEATURES, V3_FEATURES, _features

root = Path(__file__).resolve().parents[1]
ELO_K, ELO_START = 24.0, 1500.0
KW = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, max_depth=7, min_child_samples=100, subsample=0.85, colsample_bytree=0.9,
          reg_lambda=1.0, random_state=42, verbose=-1)


def labels_and_elo(files: set[str]) -> pd.DataFrame:
    """Per (source_file, innings): did the batting side win, and the Elo gap (batting - bowling) from matches BEFORE this one."""
    matches = []
    for scope in ("ipl", "t20i"):
        for path in sorted((root / "data/raw/cricsheet" / scope).glob("*.json")):
            if path.name not in files:
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            info = raw["info"]
            teams = list(info.get("teams", []))
            bats = [i.get("team") for i in raw.get("innings", [])]
            matches.append((str(info["dates"][0]), path.name, teams, bats, info.get("outcome", {}).get("winner")))
    matches.sort(key=lambda m: (m[0], m[1]))
    rating: dict[str, float] = {}
    rows = []
    for date, name, teams, bats, winner in matches:
        if len(teams) != 2:
            continue
        a, b = teams
        ra, rb = rating.get(a, ELO_START), rating.get(b, ELO_START)
        for index, bat in enumerate(bats[:2], start=1):
            other = b if bat == a else a
            r_bat, r_other = (ra, rb) if bat == a else (rb, ra)
            rows.append({"source_file": name, "innings": index, "bat_won": float(bat == winner) if winner else np.nan,
                         "elo_diff": r_bat - r_other, "elo_bat": r_bat})
        if winner in (a, b):
            expected_a = 1 / (1 + 10 ** (-(ra - rb) / 400))
            score_a = 1.0 if winner == a else 0.0
            rating[a] = ra + ELO_K * (score_a - expected_a)
            rating[b] = rb - ELO_K * (score_a - expected_a)
    return pd.DataFrame(rows)


def fit(train: pd.DataFrame, hold: pd.DataFrame, cols: list[str]) -> np.ndarray:
    cats = [c for c in CATEGORICAL_FEATURES if c in cols]
    model = lgb.LGBMClassifier(**KW).fit(_features(train, cols), train["bat_won"].astype(int), categorical_feature=cats)
    return model.predict_proba(_features(hold, cols))[:, 1]


def main() -> None:
    data = prepare()
    lab = labels_and_elo(set(data["source_file"]))
    data = data.merge(lab, on=["source_file", "innings"], how="inner", validate="many_to_one")
    data = data.dropna(subset=["bat_won"]).reset_index(drop=True)
    year = pd.to_datetime(data["match_date"]).dt.year
    data = add_team_phase(data, year <= 2024)
    data[TEAM] = data[TEAM].fillna(0.0)
    year = pd.to_datetime(data["match_date"]).dt.year
    train, hold = data[year <= 2024].reset_index(drop=True), data[year >= 2025].reset_index(drop=True)
    y = hold["bat_won"].to_numpy()
    print(f"labelled rows {len(data)} | train {len(train)} | holdout {len(hold)} from {hold['source_file'].nunique()} matches; "
          f"batting side wins {y.mean():.1%} of holdout rows\n")
    state = V3_FEATURES
    variants = {
        "baseline: match state": state,
        "+ team-phase coverage": state + TEAM,
        "+ pitch": state + VENUE,
        "+ team strength (Elo)": state + ["elo_diff", "elo_bat"],
        "+ batters' career rates": state + CAREER,
        "all together": state + TEAM + VENUE + ["elo_diff", "elo_bat"] + CAREER,
    }
    preds = {name: fit(train, hold, cols) for name, cols in variants.items()}
    inn1, chase = (hold["innings"] == 1).to_numpy(), (hold["innings"] == 2).to_numpy()
    print(f"{'variant':<28}{'Brier':>8}{'logloss':>9}{'AUC':>7}{'inn1 Brier':>12}{'chase Brier':>13}{'PP':>7}{'mid':>7}{'death':>7}")
    ph = hold["phase"].to_numpy()
    for name, p in preds.items():
        cells = [brier_score_loss(y, p), log_loss(y, p), roc_auc_score(y, p), brier_score_loss(y[inn1], p[inn1]), brier_score_loss(y[chase], p[chase])]
        cells += [brier_score_loss(y[ph == k], p[ph == k]) for k in ("powerplay", "middle", "death")]
        print(f"{name:<28}{cells[0]:8.4f}{cells[1]:9.4f}{cells[2]:7.3f}{cells[3]:12.4f}{cells[4]:13.4f}{cells[5]:7.3f}{cells[6]:7.3f}{cells[7]:7.3f}")
    codes, uniq = pd.factorize(hold["source_file"].to_numpy())
    groups = [np.where(codes == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(7)
    boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(120)]
    ref = preds["baseline: match state"]
    print("\nBrier change from baseline (negative = better), 95% interval over matches:")
    for name, p in preds.items():
        if name.startswith("baseline"):
            continue
        d_all, d_inn1, d_chase = [], [], []
        for idx in boots:
            yy = y[idx]
            d_all.append(((p[idx] - yy) ** 2).mean() - ((ref[idx] - yy) ** 2).mean())
            m1, m2 = inn1[idx], chase[idx]
            d_inn1.append(((p[idx][m1] - yy[m1]) ** 2).mean() - ((ref[idx][m1] - yy[m1]) ** 2).mean())
            d_chase.append(((p[idx][m2] - yy[m2]) ** 2).mean() - ((ref[idx][m2] - yy[m2]) ** 2).mean())
        ci = lambda v: f"{np.mean(v):+.4f} [{np.percentile(v, 2.5):+.4f},{np.percentile(v, 97.5):+.4f}]"
        print(f"  {name:<28} all {ci(d_all)} | first innings {ci(d_inn1)} | chases {ci(d_chase)}")
    best = preds["all together"]
    print("\nreliability of 'all together' (predicted bucket -> actual win rate), chases and first innings:")
    for label, mask in (("chases", chase), ("first innings", inn1)):
        bins = np.minimum((best[mask] * 10).astype(int), 9)
        line = []
        for b in range(10):
            sel = bins == b
            if sel.sum() >= 30:
                line.append(f"{b * 10}-{b * 10 + 10}%: said {best[mask][sel].mean():.0%} actual {y[mask][sel].mean():.0%} (n={sel.sum()})")
        print(f"  {label}:\n    " + "\n    ".join(line))


if __name__ == "__main__":
    main()
