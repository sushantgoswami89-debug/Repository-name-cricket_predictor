"""WinViz-style Monte Carlo win-probability vs. the live match_winner_v1
classifier, on the identical real 2025+ chase holdout (2026-08-24).

Direct follow-up to the closed LSTM investigation
(`docs/finding_wicket_lstm_sequence_model_not_promoted.md`) -- a
genuinely different mechanism this time (forward simulation, not another
trained classifier), per the 2026-08-23 strategic roadmap's #2 option.

Reuses `train_match_winner_v1.build_dataset()` unchanged (same merges,
same leakage-safe label) so the classifier's predictions on this script's
row subset are guaranteed identical to what's actually live -- not
re-derived or approximated.

Scoped to CHASES ONLY (innings 2, `is_chase == 1`): a forward simulation
of "the rest of the match" is a clean, single-stage problem for a chase
(simulate to a fixed target) but would need a two-stage simulation for
innings-1 (simulate the rest of innings 1, THEN a hypothetical innings 2)
-- out of scope for this first test, same "start with the cleanest
testable case" practice used for toss/LSTM in this project.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from app.simulation.monte_carlo_win_probability import (
    build_outcome_array, build_outcome_table, simulate_chase_win_probability,
)
from train_match_winner_v1 import build_dataset, frame
from train_phase_calibrated_sharp_range_v33 import male_source_files

N_SIMS = 2000


def _metrics(actual: np.ndarray, proba: np.ndarray) -> dict:
    predicted = (proba >= 0.5).astype(int)
    if len(np.unique(actual)) < 2:
        return {"rows": int(len(actual)), "event_rate": float(actual.mean())}
    return {
        "rows": int(len(actual)),
        "event_rate": float(actual.mean()),
        "auc": float(roc_auc_score(actual, proba)),
        "brier": float(brier_score_loss(actual, proba)),
        "accuracy": float(accuracy_score(actual, predicted)),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "models/candidates/win_probability_monte_carlo_v1"
    output.mkdir(parents=True, exist_ok=True)

    print("Rebuilding match_winner_v1's exact feature frame (no retraining)...", flush=True)
    data = build_dataset(root)
    calibration = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
    cal_chase = calibration[calibration["is_chase"] == 1].reset_index(drop=True)
    holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)
    chase = holdout[holdout["is_chase"] == 1].reset_index(drop=True)
    print(f"  holdout={len(holdout)} rows, chase subset={len(chase)} rows; "
          f"calibration chase subset={len(cal_chase)} rows")

    print("Loading live match_winner_v1 model + calibrator...", flush=True)
    model = joblib.load(root / "models/candidates/match_winner_v1/match_winner_model.pkl")
    calibrator = joblib.load(root / "models/candidates/match_winner_v1/match_winner_calibrator.pkl")
    raw = model.predict_proba(frame(chase, bowler_known=True))[:, 1]
    clipped = np.clip(raw, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    gbm_proba = calibrator.predict_proba(logit)[:, 1]

    print("Building team-agnostic outcome table from real train<=2023 deliveries...", flush=True)
    eligible = male_source_files(root)
    t0 = time.time()
    table = build_outcome_table(root, eligible, before_date="2024-01-01")
    table_arr = build_outcome_array(table)
    print(f"  {len(table)} (phase, wickets_in_hand, pressure) buckets in {time.time() - t0:.1f}s")

    def run_mc(frame_df, label: str) -> np.ndarray:
        proba = np.zeros(len(frame_df), dtype=np.float64)
        rng = np.random.default_rng(42)
        t0 = time.time()
        for i, row in enumerate(frame_df.itertuples(index=False)):
            target = int(row.score_before_over + row.runs_required)
            proba[i] = simulate_chase_win_probability(
                table_arr,
                score=int(row.score_before_over),
                wickets_in_hand=int(row.wickets_in_hand),
                legal_balls_bowled=int(row.legal_balls_bowled),
                total_legal_balls=int(row.legal_balls_bowled + row.balls_remaining),
                target=target,
                n_sims=N_SIMS,
                rng=rng,
            )
            if (i + 1) % 5000 == 0:
                print(f"  [{label}] {i + 1}/{len(frame_df)} rows simulated ({time.time() - t0:.1f}s elapsed)")
        print(f"  [{label}] done in {time.time() - t0:.1f}s")
        return proba

    cal_cache = output / "mc_cal_proba_raw.npy"
    holdout_cache = output / "mc_holdout_proba_raw.npy"
    if cal_cache.exists() and holdout_cache.exists():
        print("Loading cached raw MC probabilities from a prior run...", flush=True)
        mc_cal_proba = np.load(cal_cache)
        mc_proba_raw = np.load(holdout_cache)
    else:
        print(f"Running Monte Carlo simulation ({N_SIMS} sims/row) on calibration chase snapshots (for Platt fit)...", flush=True)
        mc_cal_proba = run_mc(cal_chase, "calibration")
        print(f"Running Monte Carlo simulation ({N_SIMS} sims/row) on {len(chase)} real holdout chase snapshots...", flush=True)
        mc_proba_raw = run_mc(chase, "holdout")
        np.save(cal_cache, mc_cal_proba)
        np.save(holdout_cache, mc_proba_raw)

    def platt_fit_apply(cal_proba, cal_actual, target_proba):
        clipped_cal = np.clip(cal_proba, 1e-6, 1 - 1e-6)
        logit_cal = np.log(clipped_cal / (1 - clipped_cal)).reshape(-1, 1)
        fitted = LogisticRegression(C=1.0, random_state=42).fit(logit_cal, cal_actual)
        clipped_target = np.clip(target_proba, 1e-6, 1 - 1e-6)
        logit_target = np.log(clipped_target / (1 - clipped_target)).reshape(-1, 1)
        return fitted.predict_proba(logit_target)[:, 1]

    # Same Platt-calibration convention as every other model in this
    # project -- skipping this the first time around was an apples-to-
    # oranges comparison against the GBM's calibrated Brier/accuracy.
    cal_actual = cal_chase["batting_team_won"].to_numpy()
    mc_proba = platt_fit_apply(mc_cal_proba, cal_actual, mc_proba_raw)

    actual = chase["batting_team_won"].to_numpy()
    is_ipl = chase["is_ipl"].to_numpy()
    cal_is_ipl = cal_chase["is_ipl"].to_numpy()

    # Per-competition Platt calibration -- same "don't trust a single
    # blended correction, split by competition" practice this project has
    # applied repeatedly (IPL band-width fix, T20I associate/full-member
    # split, etc.). Global calibration left IPL's Brier/accuracy well
    # below its own AUC advantage over the GBM; test whether an
    # IPL-specific calibrator closes that gap.
    mc_proba_per_competition = np.zeros_like(mc_proba)
    mc_proba_per_competition[is_ipl] = platt_fit_apply(
        mc_cal_proba[cal_is_ipl], cal_actual[cal_is_ipl], mc_proba_raw[is_ipl]
    )
    mc_proba_per_competition[~is_ipl] = platt_fit_apply(
        mc_cal_proba[~cal_is_ipl], cal_actual[~cal_is_ipl], mc_proba_raw[~is_ipl]
    )

    def split(proba: np.ndarray) -> dict:
        return {
            "overall": _metrics(actual, proba),
            "ipl": _metrics(actual[is_ipl], proba[is_ipl]),
            "t20i": _metrics(actual[~is_ipl], proba[~is_ipl]),
        }

    report = {
        "candidate_version": "win_probability_monte_carlo_v1",
        "candidate_only": True,
        "production_changed": False,
        "note": (
            "Team-agnostic WinViz-style Monte Carlo (phase/wickets_in_hand/"
            "pressure-conditioned outcome distribution, 2000 sims/row) vs. "
            "the live match_winner_v1 LightGBM classifier, on the IDENTICAL "
            "real 2025+ chase-only holdout subset (same build_dataset() "
            "call, same rows, guaranteed apples-to-apples). Chases only "
            "(is_chase==1) -- see module docstring for why. MC probabilities "
            "are Platt-calibrated on the 2024 calibration chase subset, same "
            "convention as every other model here -- an earlier pass skipped "
            "this and understated MC's Brier/accuracy. Also reports a "
            "per-competition (IPL-only / T20I-only) Platt calibration, since "
            "the global one left IPL's Brier/accuracy well below what its "
            "own AUC advantage over the GBM would suggest is possible."
        ),
        "n_sims_per_row": N_SIMS,
        "chase_rows": len(chase),
        "calibration_chase_rows": len(cal_chase),
        "monte_carlo_raw_uncalibrated": split(mc_proba_raw),
        "monte_carlo_platt_calibrated_global": split(mc_proba),
        "monte_carlo_platt_calibrated_per_competition": split(mc_proba_per_competition),
        "live_match_winner_v1_gbm_same_rows": split(gbm_proba),
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
