"""Does the Monte Carlo win-probability model anticipate dramatic chase
reversals earlier than the GBM (2026-08-25, user-requested)? Extends the
2026-08-23 finding (median lead over the scoreboard is 0 overs for the
GBM -- it confirms fast but doesn't lead) to compare both models directly.

Scoped to IPL only -- the Monte Carlo model has no calibrated serving
path for T20I (see docs/finding_monte_carlo_win_probability_ipl_chase_win.md).

Same "dramatic-recovery chase" definition as 2026-08-23: peak required
run rate >= 11 after over 10, chasing team still won. Real 2025+ holdout.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from app.simulation.monte_carlo_win_probability import simulate_chase_win_probability
from train_match_winner_v1 import build_dataset, frame

root = Path(__file__).resolve().parents[1]

print("Rebuilding match_winner_v1's exact feature frame...", flush=True)
data = build_dataset(root)
holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)
chase = holdout[(holdout["is_chase"] == 1) & (holdout["is_ipl"])].reset_index(drop=True)
print(f"  IPL chase rows: {len(chase)}")

won_by_match = chase.groupby("source_file")["batting_team_won"].first()
peak_rrr_after_10 = chase[chase["over"] > 10].groupby("source_file")["required_run_rate"].max()
dramatic_matches = sorted(set(
    m for m in peak_rrr_after_10.index
    if peak_rrr_after_10[m] >= 11 and won_by_match.get(m, 0) == 1
))
print(f"  Dramatic-recovery IPL matches found: {len(dramatic_matches)}")

print("Loading GBM model + calibrator...", flush=True)
gbm_model = joblib.load(root / "models/candidates/match_winner_v1/match_winner_model.pkl")
gbm_calibrator = joblib.load(root / "models/candidates/match_winner_v1/match_winner_calibrator.pkl")
gbm_raw = gbm_model.predict_proba(frame(chase, bowler_known=True))[:, 1]
clipped = np.clip(gbm_raw, 1e-6, 1 - 1e-6)
logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
chase = chase.assign(gbm_win_probability=gbm_calibrator.predict_proba(logit)[:, 1])

print("Loading Monte Carlo outcome table + IPL calibrator...", flush=True)
mc_dir = root / "models/candidates/win_probability_monte_carlo_v1"
table_arr = np.load(mc_dir / "outcome_table.npy")
ipl_calibrator = joblib.load(mc_dir / "ipl_platt_calibrator.pkl")

print(f"Simulating Monte Carlo win probability for {len(dramatic_matches)} matches' overs...", flush=True)
mc_proba = np.full(len(chase), np.nan)
rng = np.random.default_rng(42)
mask = chase["source_file"].isin(dramatic_matches).to_numpy()
for i in np.nonzero(mask)[0]:
    row = chase.iloc[i]
    target = int(row["score_before_over"] + row["runs_required"])
    raw = simulate_chase_win_probability(
        table_arr, score=int(row["score_before_over"]), wickets_in_hand=int(row["wickets_in_hand"]),
        legal_balls_bowled=int(row["legal_balls_bowled"]),
        total_legal_balls=int(row["legal_balls_bowled"] + row["balls_remaining"]),
        target=target, n_sims=2000, rng=rng,
    )
    clipped_r = min(max(raw, 1e-6), 1 - 1e-6)
    logit_r = np.log(clipped_r / (1 - clipped_r))
    mc_proba[i] = ipl_calibrator.predict_proba(np.array([[logit_r]]))[0][1]
chase["mc_win_probability"] = mc_proba


def _reversal_over(sub, prob_col):
    sub = sub.sort_values("over")
    overs = sub["over"].to_numpy()
    probs = sub[prob_col].to_numpy()
    for i in range(len(overs)):
        if probs[i] > 0.5 and np.all(probs[i:] > 0.5):
            return int(overs[i])
    return None


def _par_crossing_over(sub):
    sub = sub.sort_values("over")
    overs = sub["over"].to_numpy()
    ahead = (sub["current_run_rate"] >= sub["required_run_rate"]).to_numpy()
    for i in range(len(overs)):
        if ahead[i] and np.all(ahead[i:]):
            return int(overs[i])
    return None


results = []
for match_id in dramatic_matches:
    sub = chase[chase["source_file"] == match_id]
    par_over = _par_crossing_over(sub)
    gbm_over = _reversal_over(sub, "gbm_win_probability")
    mc_over = _reversal_over(sub, "mc_win_probability")
    results.append({
        "match_id": match_id, "par_crossing_over": par_over,
        "gbm_reversal_over": gbm_over, "mc_reversal_over": mc_over,
        "gbm_lead_over_scoreboard": (par_over - gbm_over) if (par_over is not None and gbm_over is not None) else None,
        "mc_lead_over_scoreboard": (par_over - mc_over) if (par_over is not None and mc_over is not None) else None,
        "mc_lead_over_gbm": (gbm_over - mc_over) if (gbm_over is not None and mc_over is not None) else None,
    })

gbm_leads = [r["gbm_lead_over_scoreboard"] for r in results if r["gbm_lead_over_scoreboard"] is not None]
mc_leads = [r["mc_lead_over_scoreboard"] for r in results if r["mc_lead_over_scoreboard"] is not None]
mc_vs_gbm = [r["mc_lead_over_gbm"] for r in results if r["mc_lead_over_gbm"] is not None]

report = {
    "dramatic_recovery_ipl_matches": len(dramatic_matches),
    "gbm_vs_scoreboard": {
        "matches_with_valid_reversal": len(gbm_leads),
        "mean_lead_overs": float(np.mean(gbm_leads)) if gbm_leads else None,
        "median_lead_overs": float(np.median(gbm_leads)) if gbm_leads else None,
    },
    "monte_carlo_vs_scoreboard": {
        "matches_with_valid_reversal": len(mc_leads),
        "mean_lead_overs": float(np.mean(mc_leads)) if mc_leads else None,
        "median_lead_overs": float(np.median(mc_leads)) if mc_leads else None,
    },
    "monte_carlo_vs_gbm_direct": {
        "matches_compared": len(mc_vs_gbm),
        "mean_mc_lead_over_gbm": float(np.mean(mc_vs_gbm)) if mc_vs_gbm else None,
        "median_mc_lead_over_gbm": float(np.median(mc_vs_gbm)) if mc_vs_gbm else None,
        "mc_earlier_count": sum(1 for v in mc_vs_gbm if v > 0),
        "same_over_count": sum(1 for v in mc_vs_gbm if v == 0),
        "gbm_earlier_count": sum(1 for v in mc_vs_gbm if v < 0),
    },
    "per_match": results,
}
output = root / "models/candidates/win_probability_monte_carlo_v1"
(output / "reversal_timing_comparison.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "per_match"}, indent=2))
