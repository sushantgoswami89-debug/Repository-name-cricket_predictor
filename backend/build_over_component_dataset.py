"""Build the verified Candidate v3.4 over-component dataset."""

from __future__ import annotations

from pathlib import Path

from app.ml.over_component_dataset import build_over_component_dataset

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "data/candidates/v3.4/over_components.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = build_over_component_dataset(root)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} verified over-component rows to {output}")
