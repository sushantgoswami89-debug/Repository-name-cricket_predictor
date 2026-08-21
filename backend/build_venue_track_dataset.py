"""Build Candidate v3.7 venue and similar-track features."""

from pathlib import Path

from app.ml.venue_track_dataset import build_venue_track_dataset

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "data/candidates/v3.7/venue_track_features.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = build_venue_track_dataset(root)
    frame.to_csv(output, index=False)
    print({"rows": len(frame), "output": str(output)})
