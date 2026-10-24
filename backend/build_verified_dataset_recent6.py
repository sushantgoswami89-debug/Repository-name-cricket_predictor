"""Build a recent-window=6 variant of the verified Candidate v3 dataset,
to test whether a faster-reacting momentum window (1 over instead of the
live 2-over/12-ball default) improves real run-range accuracy --
docs/live_field_test_2026-09-15.md's "run-range momentum lag" finding.

Writes to data/candidates/v3/verified_training_overs_recent6.csv,
separate from the real verified_training_overs.csv this doesn't touch.

Usage:
    .venv/bin/python3 backend/build_verified_dataset_recent6.py
"""

from pathlib import Path

from app.ml.candidate_v3_dataset import build_verified_dataset

root = Path(__file__).resolve().parents[1]
output_dir = root / "data/candidates/v3"
output_dir.mkdir(parents=True, exist_ok=True)

result = build_verified_dataset(root, recent_window=6)
result.frame.to_csv(output_dir / "verified_training_overs_recent6.csv", index=False)
print(f"Wrote {len(result.frame)} rows to verified_training_overs_recent6.csv")
print(result.report)
