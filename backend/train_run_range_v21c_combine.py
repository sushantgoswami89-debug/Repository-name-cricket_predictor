"""Part 3/3: combine the GBM-only (v21a) and NN-only (v21b) run-range
probability distributions into a single blend. Win condition: does the
ensemble beat the GBM alone on the real 2025+ holdout, at the exact same
corrected width-aware banding used for every other run-range test today
-- not whether the NN alone wins, and not a different evaluation method
that would make this incomparable to the v11 baseline.

Combination method: a single scalar blend weight alpha (blended =
alpha*gbm + (1-alpha)*nn, a convex combination of two valid probability
simplices), fit by minimizing NLL on the calibration set only (never the
holdout) -- deliberately the simplest possible combiner (1 free
parameter) given this is a first test of whether the NN carries any
independent signal at all for this target. Then the exact same per-phase
temperature scaling and width-aware best_bands banding as
train_run_range_v11_partnership_rate and every other test today.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from train_phase_calibrated_sharp_range_v33 import best_bands, nll, temperature_scale

VERSION = "run_range_v21_nn_gbm_ensemble"
root = Path(__file__).resolve().parents[1]
output = root / "models/candidates" / VERSION

g = np.load(output / "gbm_only.npz", allow_pickle=True)
m = np.load(output / "nn_only.npz", allow_pickle=True)
cal_actual, holdout_actual = g["cal_actual"], g["holdout_actual"]
holdout_is_ipl = g["holdout_is_ipl"].astype(bool)
cal_phase, holdout_phase = g["cal_phase"], g["holdout_phase"]
gbm_cal, gbm_holdout = g["gbm_cal"], g["gbm_holdout"]
nn_cal, nn_holdout = m["nn_cal"], m["nn_holdout"]


def blend_nll(alpha: float) -> float:
    blended = alpha * gbm_cal + (1 - alpha) * nn_cal
    return nll(blended, cal_actual)


result = minimize_scalar(blend_nll, bounds=(0.0, 1.0), method="bounded")
alpha = float(result.x)
print(f"Optimal blend alpha (GBM weight) = {alpha:.4f}", flush=True)

blended_cal = alpha * gbm_cal + (1 - alpha) * nn_cal
blended_holdout = alpha * gbm_holdout + (1 - alpha) * nn_holdout
gbm_alone_cal_nll = nll(gbm_cal, cal_actual)
blend_cal_nll = nll(blended_cal, cal_actual)


def evaluate(cal_raw, holdout_raw):
    temperatures = {}
    calibration_scaled = cal_raw.copy()
    holdout_scaled = holdout_raw.copy()
    for phase in ("powerplay", "middle", "death"):
        cal_mask = cal_phase == phase
        hold_mask = holdout_phase == phase
        r = minimize_scalar(
            lambda value: nll(temperature_scale(cal_raw[cal_mask], value), cal_actual[cal_mask]),
            bounds=(0.5, 3.0), method="bounded",
        )
        temperatures[phase] = float(r.x)
        calibration_scaled[cal_mask] = temperature_scale(cal_raw[cal_mask], temperatures[phase])
        holdout_scaled[hold_mask] = temperature_scale(holdout_raw[hold_mask], temperatures[phase])

    holdout_low = np.empty(len(holdout_actual), dtype=int)
    holdout_high = np.empty(len(holdout_actual), dtype=int)
    holdout_low[holdout_is_ipl], holdout_high[holdout_is_ipl] = best_bands(holdout_scaled[holdout_is_ipl], width=3)
    holdout_low[~holdout_is_ipl], holdout_high[~holdout_is_ipl] = best_bands(holdout_scaled[~holdout_is_ipl], width=2)
    hit = (holdout_actual >= holdout_low) & (holdout_actual <= holdout_high)
    return {
        "blended": float(np.mean(hit)),
        "ipl": float(np.mean(hit[holdout_is_ipl])),
        "t20i": float(np.mean(hit[~holdout_is_ipl])),
    }, temperatures


gbm_alone_hit, _ = evaluate(gbm_cal, gbm_holdout)
nn_alone_hit, _ = evaluate(nn_cal, nn_holdout)
ensemble_hit, ensemble_temperatures = evaluate(blended_cal, blended_holdout)

v11_blended, v11_ipl, v11_t20i = 0.3004, 0.3325, 0.2945  # corrected 2026-08-26 numbers
beats_v11 = bool(ensemble_hit["blended"] > v11_blended and blend_cal_nll <= gbm_alone_cal_nll)
beats_v11_on_ipl = bool(ensemble_hit["ipl"] > v11_ipl)

report = {
    "candidate_version": VERSION,
    "candidate_only": True,
    "production_changed": False,
    "run_model_changed": False,
    "note": (
        "Architecture test, not a new feature: does a GBM+NN ensemble (the "
        "same idea that gave a real win on the wicket target) beat the "
        "GBM alone for run-range, using the exact same feature set as the "
        "live run_range_v11_partnership_rate? Single scalar blend weight "
        "fit on calibration only."
    ),
    "blend_alpha_gbm_weight": alpha,
    "reference_run_range_v11_partnership_rate_corrected": {
        "blended": v11_blended, "ipl": v11_ipl, "t20i": v11_t20i,
    },
    "gbm_alone_holdout": gbm_alone_hit,
    "nn_alone_holdout": nn_alone_hit,
    "ensemble_holdout": ensemble_hit,
    "beats_v11_blended_and_calibration_gate": beats_v11,
    "beats_v11_on_ipl_specifically": beats_v11_on_ipl,
    "decision": "promote_candidate" if beats_v11 else "reject_keep_research",
}
(output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
