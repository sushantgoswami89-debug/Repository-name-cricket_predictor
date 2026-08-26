"""Does a genuinely bigger/deeper NN improve the run-range GBM+NN ensemble
further, beyond the v21 result (+0.18-0.20pp over v11 on all cuts)? User
explicitly asked to push the NN architecture further ("this is still
low"). Same features, same GBM (reused from v21a, not retrained), same
downstream calibration/banding pipeline -- only the NN body changes:
wider (256->128->64 vs v21's 128->64), larger embeddings (8 vs 4),
LR scheduling (ReduceLROnPlateau vs fixed), more training budget (up to
60 epochs / patience 6 vs 25 / patience 4), light label smoothing.

Reuses the train/calibration/holdout caches and the already-trained GBM
outputs from models/candidates/run_range_v21_nn_gbm_ensemble/ -- no need
to rebuild the dataset or retrain the GBM a second time."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from torch import nn

from train_phase_calibrated_sharp_range_v33 import MAX_RUN_CLASS, best_bands, nll, temperature_scale

root = Path(__file__).resolve().parent.parent
v21_dir = root / "models/candidates/run_range_v21_nn_gbm_ensemble"
output = root / "models/candidates/run_range_v22_bigger_nn"
output.mkdir(parents=True, exist_ok=True)

train = pd.read_pickle(v21_dir / "train_cache.pkl")
cal = pd.read_pickle(v21_dir / "calibration_cache.pkl")
holdout = pd.read_pickle(v21_dir / "holdout_cache.pkl")
g = np.load(v21_dir / "gbm_only.npz", allow_pickle=True)
gbm_cal, gbm_holdout = g["gbm_cal"], g["gbm_holdout"]
cal_actual, holdout_actual = g["cal_actual"], g["holdout_actual"]
holdout_is_ipl = g["holdout_is_ipl"].astype(bool)
cal_phase, holdout_phase = g["cal_phase"], g["holdout_phase"]

CAT = [
    "phase", "active_batter_state", "wickets_remaining_bucket",
    "state_regime", "chase_pressure", "venue_scoring_regime",
    "venue_par_source", "batting_team_venue_context", "phase_venue_regime",
    "striker_batting_style", "bowler_type",
]
NUMERIC = [c for c in train.columns if c not in CAT + ["runs_in_over", "is_ipl"]]
NUM_CLASSES = MAX_RUN_CLASS + 1
EMB_DIM = 8

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


class BiggerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(cat_dims[c], EMB_DIM) for c in CAT])
        in_dim = len(NUMERIC) + EMB_DIM * len(CAT)
        self.body = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, num, cats):
        embs = [e(cats[:, i]) for i, e in enumerate(self.embs)]
        return self.body(torch.cat([num] + embs, dim=1))


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
net = BiggerNet().to(device)
train_num, train_cat = to_tensors(train)
cal_num, cal_cat = to_tensors(cal)
holdout_num, holdout_cat = to_tensors(holdout)
train_y = torch.tensor(np.minimum(train["runs_in_over"].to_numpy(), MAX_RUN_CLASS), dtype=torch.long)
cal_y = np.minimum(cal["runs_in_over"].to_numpy(), MAX_RUN_CLASS)

opt = torch.optim.Adam(net.parameters(), lr=1.5e-3, weight_decay=2e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
crit = nn.CrossEntropyLoss(label_smoothing=0.02)
n = len(train)
batch = 2048
best_nll, best_state, patience = float("inf"), None, 0
print("Training bigger NN...", flush=True)
for epoch in range(1, 61):
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
    scheduler.step(cal_nll)
    print(f"  epoch {epoch} cal_nll={cal_nll:.4f} lr={opt.param_groups[0]['lr']:.2e}", flush=True)
    if cal_nll < best_nll - 1e-5:
        best_nll, best_state, patience = cal_nll, {k: v.clone() for k, v in net.state_dict().items()}, 0
    else:
        patience += 1
        if patience >= 6:
            break
net.load_state_dict(best_state)
net.eval()
with torch.no_grad():
    nn_cal_logit = net(cal_num.to(device), cal_cat.to(device)).cpu().numpy()
    nn_holdout_logit = net(holdout_num.to(device), holdout_cat.to(device)).cpu().numpy()


def softmax(logit):
    p = np.exp(logit - logit.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


nn_cal, nn_holdout = softmax(nn_cal_logit), softmax(nn_holdout_logit)


def blend_nll(alpha: float) -> float:
    return nll(alpha * gbm_cal + (1 - alpha) * nn_cal, cal_actual)


result = minimize_scalar(blend_nll, bounds=(0.0, 1.0), method="bounded")
alpha = float(result.x)
blended_cal = alpha * gbm_cal + (1 - alpha) * nn_cal
blended_holdout = alpha * gbm_holdout + (1 - alpha) * nn_holdout


def evaluate(cal_raw, holdout_raw):
    temperatures = {}
    calibration_scaled = cal_raw.copy()
    holdout_scaled = holdout_raw.copy()
    for phase in ("powerplay", "middle", "death"):
        cal_mask = cal_phase == phase
        hold_mask = holdout_phase == phase
        r = minimize_scalar(
            lambda value: nll(temperature_scale(cal_raw[cal_mask], value), cal_actual[cal_mask]),
            bounds=(0.5, 3.0), method="bounded",
        )
        temperatures[phase] = float(r.x)
        calibration_scaled[cal_mask] = temperature_scale(cal_raw[cal_mask], temperatures[phase])
        holdout_scaled[hold_mask] = temperature_scale(holdout_raw[hold_mask], temperatures[phase])

    holdout_low = np.empty(len(holdout_actual), dtype=int)
    holdout_high = np.empty(len(holdout_actual), dtype=int)
    holdout_low[holdout_is_ipl], holdout_high[holdout_is_ipl] = best_bands(holdout_scaled[holdout_is_ipl], width=3)
    holdout_low[~holdout_is_ipl], holdout_high[~holdout_is_ipl] = best_bands(holdout_scaled[~holdout_is_ipl], width=2)
    hit = (holdout_actual >= holdout_low) & (holdout_actual <= holdout_high)
    return {
        "blended": float(np.mean(hit)),
        "ipl": float(np.mean(hit[holdout_is_ipl])),
        "t20i": float(np.mean(hit[~holdout_is_ipl])),
    }


nn_alone_hit = evaluate(nn_cal, nn_holdout)
ensemble_hit = evaluate(blended_cal, blended_holdout)

v11 = {"blended": 0.3004, "ipl": 0.3325, "t20i": 0.2945}
v21_ensemble = {"blended": 0.30225, "ipl": 0.33429, "t20i": 0.29645}  # from the smaller-NN ensemble

report = {
    "candidate_version": "run_range_v22_bigger_nn",
    "note": (
        "Bigger NN (256->128->64 vs v21's 128->64, embed dim 8 vs 4, "
        "LR scheduling, label smoothing, longer training budget) blended "
        "with the SAME GBM as v21 -- isolates whether NN capacity was the "
        "limiting factor in the v21 ensemble result."
    ),
    "blend_alpha_gbm_weight": alpha,
    "reference_v11_live": v11,
    "reference_v21_smaller_nn_ensemble": v21_ensemble,
    "nn_alone_holdout": nn_alone_hit,
    "ensemble_holdout": ensemble_hit,
    "beats_v21_ensemble": bool(ensemble_hit["blended"] > v21_ensemble["blended"]),
}
(output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
