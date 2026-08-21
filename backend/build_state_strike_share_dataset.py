"""Build Candidate v3.6 state and strike-share features."""

from pathlib import Path

from app.ml.state_strike_share_dataset import build_state_strike_share_dataset

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "data/candidates/v3.6/state_strike_share_features.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = build_state_strike_share_dataset(root)
    frame.to_csv(output, index=False)
    print({"rows": len(frame), "output": str(output)})
