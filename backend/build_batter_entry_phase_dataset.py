"""Build Candidate v3.5 batter entry-phase response features."""

from pathlib import Path

from app.ml.batter_entry_phase_dataset import build_batter_entry_phase_dataset

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "data/candidates/v3.5/batter_entry_phase_features.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = build_batter_entry_phase_dataset(root)
    frame.to_csv(output, index=False)
    print({"rows": len(frame), "output": str(output)})
