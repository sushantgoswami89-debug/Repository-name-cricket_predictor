"""Part 2/3 of the NN+GBM ensemble test -- torch only, no lightgbm (see
train_wicket_v19a_gbm_only.py for why these run as separate processes)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from app.ml.cascade_features import WICKET_CATEGORICAL, WICKET_FEATURES, build_wicket_base_dataset

NUMERIC = [c for c in WICKET_FEATURES if c not in WICKET_CATEGORICAL]
CAT = WICKET_CATEGORICAL
root = Path(__file__).resolve().parents[1]
output = root / "models/candidates/wicket_v19_nn_gbm_ensemble"
output.mkdir(parents=True, exist_ok=True)

print("Building dataset...", flush=True)
data = build_wicket_base_dataset(root)
train = data[data["match_date"] <= "2023-12-31"].reset_index(drop=True)
cal = data[data["match_date"].dt.year == 2024].reset_index(drop=True)
holdout = data[data["match_date"] >= "2025-01-01"].reset_index(drop=True)

cat_maps = {c: {v: i + 1 for i, v in enumerate(train[c].fillna("__NA__").astype(str).unique())} for c in CAT}
cat_dims = {c: len(m) + 1 for c, m in cat_maps.items()}
num_mean = train[NUMERIC].mean()
num_std = train[NUMERIC].std().replace(0, 1)

def to_tensors(d):
    num = torch.tensor(((d[NUMERIC].fillna(0) - num_mean) / num_std).to_numpy(), dtype=torch.float32)
    cats = torch.tensor(np.stack([d[c].fillna("__NA__").astype(str).map(cat_maps[c]).fillna(0).to_numpy() for c in CAT], axis=1), dtype=torch.long)
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
train_y = torch.tensor(train["wicket_in_over"].to_numpy(), dtype=torch.float32)
pos_weight = torch.tensor([(len(train_y) - train_y.sum()) / train_y.sum()])

opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
n = len(train)
batch = 1024
best_auc, best_state, patience = -1, None, 0
print("Training NN...", flush=True)
for epoch in range(1, 21):
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
    cal_auc = roc_auc_score(cal["wicket_in_over"], cal_logit)
    print(f"  epoch {epoch} cal_auc={cal_auc:.4f}")
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
