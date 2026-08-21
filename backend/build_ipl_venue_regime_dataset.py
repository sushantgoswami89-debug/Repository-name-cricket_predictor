"""Build the leakage-safe IPL venue scoring-regime dataset."""

from __future__ import annotations

from pathlib import Path

from app.ml.ipl_venue_regime_dataset import build_ipl_venue_regime_dataset

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "data/candidates/ipl_venue_regime/features.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = build_ipl_venue_regime_dataset(root)
    frame.to_csv(output, index=False)
    print(frame["venue_scoring_regime"].value_counts().to_string())
