"""Build the production artifacts for the IPL-chase Monte Carlo win-
probability route (2026-08-24) -- see
`docs/finding_monte_carlo_win_probability_ipl_chase_win.md` for the full
validated result (beats the live GBM on all 3 metrics for IPL chases,
real 2025+ holdout).

Reuses the cached raw calibration-set probabilities from
`evaluate_win_probability_monte_carlo_v1.py`'s research run (same rng
seed, same outcome table construction) rather than re-simulating --
saves ~2.5 minutes, and the calibrator fit only depends on those cached
values plus the real labels, both already reproduced/verified in that
script's own IPL-only report section.

Writes: `outcome_table.npy` (dense simulation lookup table),
`ipl_platt_calibrator.pkl` (fit on 2024 calibration IPL-chase rows only,
matching this project's standing per-competition-calibration practice),
`ARTIFACT_MANIFEST.json` (sha256 integrity, same convention as every
other live runtime here).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from app.simulation.monte_carlo_win_probability import build_outcome_array, build_outcome_table
from train_match_winner_v1 import build_dataset
from train_phase_calibrated_sharp_range_v33 import male_source_files

VERSION = "win_probability_monte_carlo_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "models/candidates" / VERSION
    output.mkdir(parents=True, exist_ok=True)

    cal_cache = output / "mc_cal_proba_raw.npy"
    if not cal_cache.exists():
        raise FileNotFoundError(
            f"{cal_cache} not found -- run evaluate_win_probability_monte_carlo_v1.py "
            "first to produce the cached calibration-set raw probabilities."
        )
    mc_cal_proba = np.load(cal_cache)

    print("Rebuilding match_winner_v1's exact feature frame (for real labels + is_ipl mask only)...", flush=True)
    data = build_dataset(root)
    calibration = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
    cal_chase = calibration[calibration["is_chase"] == 1].reset_index(drop=True)
    if len(cal_chase) != len(mc_cal_proba):
        raise ValueError(
            f"Cached calibration probabilities ({len(mc_cal_proba)} rows) don't match "
            f"the rebuilt calibration chase frame ({len(cal_chase)} rows) -- rerun "
            "evaluate_win_probability_monte_carlo_v1.py to regenerate the cache."
        )

    cal_is_ipl = cal_chase["is_ipl"].to_numpy()
    cal_actual = cal_chase["batting_team_won"].to_numpy()

    print(f"Fitting IPL-only Platt calibrator on {int(cal_is_ipl.sum())} calibration chase rows...", flush=True)
    ipl_raw = mc_cal_proba[cal_is_ipl]
    ipl_actual = cal_actual[cal_is_ipl]
    clipped = np.clip(ipl_raw, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    ipl_calibrator = LogisticRegression(C=1.0, random_state=42).fit(logit, ipl_actual)

    print("Building the outcome table (train<=2023 deliveries)...", flush=True)
    eligible = male_source_files(root)
    table = build_outcome_table(root, eligible, before_date="2024-01-01")
    table_arr = build_outcome_array(table)

    np.save(output / "outcome_table.npy", table_arr)
    joblib.dump(ipl_calibrator, output / "ipl_platt_calibrator.pkl")

    manifest = {
        "candidate_version": VERSION,
        "artifacts": {
            "outcome_table.npy": _sha256(output / "outcome_table.npy"),
            "ipl_platt_calibrator.pkl": _sha256(output / "ipl_platt_calibrator.pkl"),
        },
    }
    (output / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote production artifacts to {output}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
