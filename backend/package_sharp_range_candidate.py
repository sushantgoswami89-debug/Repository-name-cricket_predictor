"""Checksum-package the validated two-run Sharp Range candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(project_root: Path) -> dict[str, object]:
    candidate = project_root / "models/candidates/v3.2_sharp_range"
    validation = json.loads(
        (candidate / "validation_report.json").read_text(encoding="utf-8")
    )
    if not validation["ready_for_shadow"]:
        raise RuntimeError("Sharp Range candidate has not passed every gate.")
    required = [
        "sharp_range_model.pkl",
        "feature_cols.pkl",
        "cat_cols.pkl",
        "validation_report.json",
    ]
    manifest = {
        "candidate": "cricketbaba_v3.2_two_run_sharp_range",
        "status": "ready_for_live_shadow",
        "created_at": "2026-07-22",
        "production_promoted": False,
        "range_width_runs": 2,
        "confidence_changed": False,
        "safety_range_changed": False,
        "all_offline_gates_passed": True,
        "artifacts": {name: _sha256(candidate / name) for name in required},
    }
    (candidate / "READY_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(package(root), indent=2))
