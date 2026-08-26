"""Part 2/3 of the run-range NN+GBM ensemble test -- torch only, no
lightgbm (see train_run_range_v21a_gbm_only.py for why these run as
separate processes, and for the full rationale). Loads the cached
train/calibration/holdout feature frames v21a wrote to parquet -- exact
same features/rows as the GBM, so this is a fair architecture comparison,
not a different dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from train_phase_calibrated_sharp_range_v33 import MAX_RUN_CLASS

VERSION = "run_range_v21_nn_gbm_ensemble"
root = Path(__file__).resolve().parents[1]
output = root / "models/candidates" / VERSION

train = pd.read_pickle(output / "train_cache.pkl")
cal = pd.read_pickle(output / "calibration_cache.pkl")
holdout = pd.read_pickle(output / "holdout_cache.pkl")

CAT = [
    "phase", "active_batter_state", "wickets_remaining_bucket",
    "state_regime", "chase_pressure", "venue_scoring_regime",
    "venue_par_source", "batting_team_venue_context", "phase_venue_regime",
    "striker_batting_style", "bowler_type",
]
NUMERIC = [c for c in train.columns if c not in CAT + ["runs_in_over", "phase", "is_ipl"]]

cat_maps = {c: {v: i + 1 for i, v in enumerate(train[c].fillna("__NA__").astype(str).unique())} for c in CAT}
cat_dims = {c: len(m) + 1 for c, m in cat_maps.items()}
num_mean = train[NUMERIC].mean()
num_std = train[NUMERIC].std().replace(0, 1)


def to_tensors(d):
    num = torch.tensor(((d[NUMERIC].fillna(0) - num_mean) / num_std).to_numpy(), dtype=torch.float32)
    cats = torch.tensor(
        np.stack([d[c].fillna("__NA__").astype(str).map(cat_maps[c]).fillna(0).to_numpy() for c in CAT], axis=1),
        dtype=torch.long,
    )
    return num, cats


NUM_CLASSES = MAX_RUN_CLASS + 1


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(cat_dims[c], 4) for c in CAT])
        in_dim = len(NUMERIC) + 4 * len(CAT)
        self.body = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, num, cats):
        embs = [e(cats[:, i]) for i, e in enumerate(self.embs)]
        return self.body(torch.cat([num] + embs, dim=1))


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
net = Net().to(device)
train_num, train_cat = to_tensors(train)
cal_num, cal_cat = to_tensors(cal)
holdout_num, holdout_cat = to_tensors(holdout)
train_y = torch.tensor(np.minimum(train["runs_in_over"].to_numpy(), MAX_RUN_CLASS), dtype=torch.long)
cal_y = np.minimum(cal["runs_in_over"].to_numpy(), MAX_RUN_CLASS)

opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
crit = nn.CrossEntropyLoss()
n = len(train)
batch = 2048
best_nll, best_state, patience = float("inf"), None, 0
print("Training NN...", flush=True)
for epoch in range(1, 26):
    net.train()
    perm = torch.randperm(n)
    for i in range(0, n, batch):
        idx = perm[i:i + batch]
        opt.zero_grad()
        out = net(train_num[idx].to(device), train_cat[idx].to(device))
        loss = crit(out, train_y[idx].to(device))
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        cal_logit = net(cal_num.to(device), cal_cat.to(device)).cpu().numpy()
    cal_prob = np.exp(cal_logit - cal_logit.max(axis=1, keepdims=True))
    cal_prob /= cal_prob.sum(axis=1, keepdims=True)
    cal_nll = float(-np.log(np.clip(cal_prob[np.arange(len(cal_y)), cal_y], 1e-9, 1.0)).mean())
    print(f"  epoch {epoch} cal_nll={cal_nll:.4f}", flush=True)
    if cal_nll < best_nll:
        best_nll, best_state, patience = cal_nll, {k: v.clone() for k, v in net.state_dict().items()}, 0
    else:
        patience += 1
        if patience >= 4:
            break
net.load_state_dict(best_state)
net.eval()
with torch.no_grad():
    nn_cal_logit = net(cal_num.to(device), cal_cat.to(device)).cpu().numpy()
    nn_holdout_logit = net(holdout_num.to(device), holdout_cat.to(device)).cpu().numpy()


def softmax(logit):
    p = np.exp(logit - logit.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


nn_cal = softmax(nn_cal_logit)
nn_holdout = softmax(nn_holdout_logit)

np.savez(output / "nn_only.npz", nn_cal=nn_cal, nn_holdout=nn_holdout)
print("Saved nn_only.npz")
