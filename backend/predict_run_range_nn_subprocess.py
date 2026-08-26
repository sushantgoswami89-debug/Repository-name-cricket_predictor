"""Standalone NN-only inference CLI for the run-range ensemble
(2026-08-26). Runs in its OWN process specifically so torch never
coexists with lightgbm in the live PredictionEngine process (confirmed
hang/crash if it does -- see
docs/finding_run_range_nn_gbm_ensemble_real_win.md). Reads one JSON
feature dict from stdin, writes {"probabilities": [p0, p1, ..., pN]} to
stdout (one probability per run-count class). Invoked via subprocess by
app/ml/run_range_ensemble_runtime.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "models/candidates/run_range_v21_nn_gbm_ensemble"


class Net(nn.Module):
    def __init__(self, numeric_dim: int, cat_dims: dict[str, int], cat_order: list[str], num_classes: int):
        super().__init__()
        self.cat_order = cat_order
        self.embs = nn.ModuleList([nn.Embedding(cat_dims[c], 4) for c in cat_order])
        in_dim = numeric_dim + 4 * len(cat_order)
        self.body = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, num, cats):
        embs = [e(cats[:, i]) for i, e in enumerate(self.embs)]
        return self.body(torch.cat([num] + embs, dim=1))


def main() -> None:
    row = json.loads(sys.stdin.read())
    metadata = json.loads((ARTIFACT_DIR / "nn_metadata.json").read_text(encoding="utf-8"))
    numeric, categorical = metadata["numeric"], metadata["categorical"]
    cat_maps, cat_dims = metadata["cat_maps"], metadata["cat_dims"]
    num_mean, num_std = metadata["num_mean"], metadata["num_std"]
    num_classes = metadata["num_classes"]

    num_vec = np.array([[(float(row.get(c, 0.0)) - num_mean[c]) / (num_std[c] or 1.0) for c in numeric]], dtype=np.float32)
    cat_vec = np.array([[cat_maps[c].get(str(row.get(c, "__NA__")), 0) for c in categorical]], dtype=np.int64)

    net = Net(len(numeric), cat_dims, categorical, num_classes)
    net.load_state_dict(torch.load(ARTIFACT_DIR / "nn_model.pt", map_location="cpu"))
    net.eval()
    with torch.no_grad():
        logit = net(torch.from_numpy(num_vec), torch.from_numpy(cat_vec)).numpy()
    prob = np.exp(logit - logit.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    print(json.dumps({"probabilities": prob[0].tolist()}))


if __name__ == "__main__":
    main()
