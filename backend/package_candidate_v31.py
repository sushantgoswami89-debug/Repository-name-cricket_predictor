"""Create a checksum manifest for the validated Candidate v3.1 layer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(project_root: Path) -> dict[str, object]:
    candidate = project_root / "models/candidates/v3.1_historical_analogue"
    validation = json.loads(
        (candidate / "validation_report.json").read_text(encoding="utf-8")
    )
    if not validation["promotion_recommended"]:
        raise RuntimeError("Candidate v3.1 cannot be packaged because a gate failed.")
    required = [
        "historical_analogue_engine.pkl",
        "confidence_calibrator.pkl",
        "analogue_config.json",
        "validation_report.json",
    ]
    locked_base = project_root / "models/locked/cricketbaba_candidate_v3"
    manifest = {
        "candidate": "cricketbaba_v3.1_historical_analogue",
        "status": "ready_for_live_shadow",
        "created_at": "2026-07-22",
        "production_promoted": False,
        "base_model": "models/locked/cricketbaba_candidate_v3",
        "base_locked_manifest_sha256": _sha256(locked_base / "LOCKED_MANIFEST.json"),
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
