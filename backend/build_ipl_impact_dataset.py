"""Build IPL impact-player state features."""

from pathlib import Path

from app.ml.ipl_impact_dataset import build_ipl_impact_dataset

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "data/candidates/ipl_v1/impact_player_features.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = build_ipl_impact_dataset(root)
    frame.to_csv(output, index=False)
    print({"rows": len(frame), "output": str(output)})
