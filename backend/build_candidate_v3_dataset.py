"""Build and verify the Candidate v3 training dataset."""

from __future__ import annotations

import json
from pathlib import Path

from app.ml.candidate_v3_dataset import write_verified_dataset

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    report = write_verified_dataset(root, root / "data/candidates/v3")
    print(json.dumps(report, indent=2))
