"""Standalone NN-only inference CLI for the wicket ensemble (2026-08-25).
Runs in its OWN process specifically so torch never coexists with
lightgbm in the live PredictionEngine process (confirmed hang/crash if
it does -- see docs/finding_wicket_nn_gbm_ensemble_real_win.md). Reads
one JSON feature dict from stdin, writes {"probability": p} to stdout.
Invoked via subprocess by app/ml/wicket_ensemble_runtime.py.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "models/candidates/wicket_v19_nn_gbm_ensemble"


class Net(nn.Module):
    def __init__(self, numeric_dim: int, cat_dims: dict[str, int], cat_order: list[str]):
        super().__init__()
        self.cat_order = cat_order
        self.embs = nn.ModuleList([nn.Embedding(cat_dims[c], 4) for c in cat_order])
        in_dim = numeric_dim + 4 * len(cat_order)
        self.body = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
                                   nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, num, cats):
        embs = [e(cats[:, i]) for i, e in enumerate(self.embs)]
        return self.body(torch.cat([num] + embs, dim=1)).squeeze(-1)


def main() -> None:
    row = json.loads(sys.stdin.read())
    metadata = json.loads((ARTIFACT_DIR / "nn_metadata.json").read_text(encoding="utf-8"))
    numeric, categorical = metadata["numeric"], metadata["categorical"]
    cat_maps, cat_dims = metadata["cat_maps"], metadata["cat_dims"]
    num_mean, num_std = metadata["num_mean"], metadata["num_std"]

    num_vec = np.array([[(float(row.get(c, 0.0)) - num_mean[c]) / (num_std[c] or 1.0) for c in numeric]], dtype=np.float32)
    cat_vec = np.array([[cat_maps[c].get(str(row.get(c, "__NA__")), 0) for c in categorical]], dtype=np.int64)

    net = Net(len(numeric), cat_dims, categorical)
    net.load_state_dict(torch.load(ARTIFACT_DIR / "nn_model.pt", map_location="cpu"))
    net.eval()
    with torch.no_grad():
        logit = net(torch.from_numpy(num_vec), torch.from_numpy(cat_vec)).item()
    probability = 1 / (1 + np.exp(-logit))
    print(json.dumps({"probability": float(probability)}))


if __name__ == "__main__":
    main()
