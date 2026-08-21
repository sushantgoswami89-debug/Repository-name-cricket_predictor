"""Build the candidate-only IPL phase mixture feature dataset."""

from pathlib import Path

from app.ml.ipl_phase_moe_dataset import build_ipl_phase_moe_features

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "data/candidates/ipl_phase_moe_v1/features.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = build_ipl_phase_moe_features(root)
    frame.to_csv(output, index=False)
    print(f"rows={len(frame)} matches={frame['source_file'].nunique()}")
