"""Build Candidate v3.3 sequence and player-response features."""

from __future__ import annotations

from pathlib import Path

from app.ml.sequence_response_dataset import build_sequence_response_dataset

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "data/candidates/v3.3/sequence_response_features.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = build_sequence_response_dataset(root)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} verified sequence-response rows to {output}")
