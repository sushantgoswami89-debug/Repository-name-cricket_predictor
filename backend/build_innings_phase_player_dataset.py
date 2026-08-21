"""Build Candidate v3.8 player innings-role and phase profiles."""

from pathlib import Path

from app.ml.innings_phase_player_dataset import build_innings_phase_player_dataset

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "data/candidates/v3.8/innings_phase_player_features.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = build_innings_phase_player_dataset(root)
    frame.to_csv(output, index=False)
    print({"rows": len(frame), "output": str(output)})
