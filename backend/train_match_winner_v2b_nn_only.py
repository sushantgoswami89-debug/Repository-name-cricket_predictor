"""Part 2/3 of the match_winner NN+GBM ensemble test -- torch only, no
lightgbm (see train_match_winner_v2a_gbm_only.py for why these run as
separate processes). Loads the cached train/calibration/holdout feature
frames v2a wrote -- exact same features/rows as the GBM, fair
architecture comparison, not a different dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

VERSION = "match_winner_v2_nn_gbm_ensemble"
root = Path(__file__).resolve().parents[1]
output = root / "models/candidates" / VERSION

train = pd.read_pickle(output / "train_cache.pkl")
cal = pd.read_pickle(output / "calibration_cache.pkl")
holdout = pd.read_pickle(output / "holdout_cache.pkl")

CAT = ["phase", "striker_batting_style", "bowler_type", "venue_scoring_regime",
       "venue_par_source", "active_batter_state", "toss_decision", "batting_team_venue_context"]
NUMERIC = [c for c in train.columns if c not in CAT + ["batting_team_won", "phase", "is_ipl"]]

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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(cat_dims[c], 4) for c in CAT])
        in_dim = len(NUMERIC) + 4 * len(CAT)
        self.body = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
                                   nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, num, cats):
        embs = [e(cats[:, i]) for i, e in enumerate(self.embs)]
        return self.body(torch.cat([num] + embs, dim=1)).squeeze(-1)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
net = Net().to(device)
train_num, train_cat = to_tensors(train)
cal_num, cal_cat = to_tensors(cal)
holdout_num, holdout_cat = to_tensors(holdout)
train_y = torch.tensor(train["batting_team_won"].to_numpy(), dtype=torch.float32)
pos_weight = torch.tensor([(len(train_y) - train_y.sum()) / train_y.sum()])

opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
n = len(train)
batch = 2048
best_auc, best_state, patience = -1, None, 0
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
    cal_auc = roc_auc_score(cal["batting_team_won"], cal_logit)
    print(f"  epoch {epoch} cal_auc={cal_auc:.4f}", flush=True)
    if cal_auc > best_auc:
        best_auc, best_state, patience = cal_auc, {k: v.clone() for k, v in net.state_dict().items()}, 0
    else:
        patience += 1
        if patience >= 4:
            break
net.load_state_dict(best_state)
net.eval()
with torch.no_grad():
    nn_cal_logit = net(cal_num.to(device), cal_cat.to(device)).cpu().numpy()
    nn_holdout_logit = net(holdout_num.to(device), holdout_cat.to(device)).cpu().numpy()
nn_cal = 1 / (1 + np.exp(-nn_cal_logit))
nn_holdout = 1 / (1 + np.exp(-nn_holdout_logit))

np.savez(output / "nn_only.npz", nn_cal=nn_cal, nn_holdout=nn_holdout)
print("Saved nn_only.npz")
