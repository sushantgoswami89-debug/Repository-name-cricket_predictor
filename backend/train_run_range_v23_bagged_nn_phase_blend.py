"""Two more real levers for the run-range ensemble, after "bigger single
NN" failed to improve on v21 (finding_run_range_bigger_nn_no_further_gain.md):

1. **Bag 3 NNs** (same winning smaller architecture as v21 -- 128->64,
   embed dim 4 -- since that beat the bigger variant), different random
   seeds, average their softmax outputs. Variance reduction via
   ensembling multiple training runs is a different lever than raising
   one network's capacity, and a standard way to squeeze more out of an
   architecture that already works rather than one that doesn't.
2. **Per-phase blend weight** instead of one global scalar alpha --
   powerplay/middle/death already get separate temperature calibration
   in this pipeline; there's no reason the GBM/NN mix should be forced
   the same in all three phases. Fits one alpha per phase on the
   calibration set.

Reuses the train/calibration/holdout caches and the already-trained GBM
outputs from models/candidates/run_range_v21_nn_gbm_ensemble/ -- no need
to rebuild the dataset or retrain the GBM again."""
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
output = root / "models/candidates/run_range_v23_bagged_nn_phase_blend"
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
        self.body = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, num, cats):
        embs = [e(cats[:, i]) for i, e in enumerate(self.embs)]
        return self.body(torch.cat([num] + embs, dim=1))


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
train_num, train_cat = to_tensors(train)
cal_num, cal_cat = to_tensors(cal)
holdout_num, holdout_cat = to_tensors(holdout)
train_y = torch.tensor(np.minimum(train["runs_in_over"].to_numpy(), MAX_RUN_CLASS), dtype=torch.long)
cal_y = np.minimum(cal["runs_in_over"].to_numpy(), MAX_RUN_CLASS)


def softmax(logit):
    p = np.exp(logit - logit.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def train_one(seed: int) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    net = Net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.CrossEntropyLoss()
    n = len(train)
    batch = 2048
    best_nll, best_state, patience = float("inf"), None, 0
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
        cal_prob = softmax(cal_logit)
        cal_nll_value = float(-np.log(np.clip(cal_prob[np.arange(len(cal_y)), cal_y], 1e-9, 1.0)).mean())
        if cal_nll_value < best_nll:
            best_nll, best_state, patience = cal_nll_value, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 4:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        c_logit = net(cal_num.to(device), cal_cat.to(device)).cpu().numpy()
        h_logit = net(holdout_num.to(device), holdout_cat.to(device)).cpu().numpy()
    print(f"  seed {seed}: best_cal_nll={best_nll:.4f}", flush=True)
    return softmax(c_logit), softmax(h_logit)


print("Training bagged NNs (3 seeds)...", flush=True)
seeds = [42, 123, 7]
cal_preds, holdout_preds = [], []
for seed in seeds:
    c, h = train_one(seed)
    cal_preds.append(c)
    holdout_preds.append(h)
nn_cal = np.mean(cal_preds, axis=0)
nn_holdout = np.mean(holdout_preds, axis=0)


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

# Global scalar blend (same method as v21) for comparison.
def blend_nll_global(alpha: float) -> float:
    return nll(alpha * gbm_cal + (1 - alpha) * nn_cal, cal_actual)


global_alpha = float(minimize_scalar(blend_nll_global, bounds=(0.0, 1.0), method="bounded").x)
global_blend_cal = global_alpha * gbm_cal + (1 - global_alpha) * nn_cal
global_blend_holdout = global_alpha * gbm_holdout + (1 - global_alpha) * nn_holdout
global_hit = evaluate(global_blend_cal, global_blend_holdout)

# Per-phase blend weight.
phase_alphas: dict[str, float] = {}
blended_cal = gbm_cal.copy()
blended_holdout = gbm_holdout.copy()
for phase in ("powerplay", "middle", "death"):
    cal_mask = cal_phase == phase
    hold_mask = holdout_phase == phase

    def phase_blend_nll(alpha: float, mask=cal_mask) -> float:
        return nll(alpha * gbm_cal[mask] + (1 - alpha) * nn_cal[mask], cal_actual[mask])

    a = float(minimize_scalar(phase_blend_nll, bounds=(0.0, 1.0), method="bounded").x)
    phase_alphas[phase] = a
    blended_cal[cal_mask] = a * gbm_cal[cal_mask] + (1 - a) * nn_cal[cal_mask]
    blended_holdout[hold_mask] = a * gbm_holdout[hold_mask] + (1 - a) * nn_holdout[hold_mask]

phase_blend_hit = evaluate(blended_cal, blended_holdout)

v11 = {"blended": 0.3004, "ipl": 0.3325, "t20i": 0.2945}
v21_ensemble = {"blended": 0.30225, "ipl": 0.33429, "t20i": 0.29645}

report = {
    "candidate_version": "run_range_v23_bagged_nn_phase_blend",
    "note": (
        "3-seed bagged NN (same smaller architecture that beat the bigger "
        "single NN in v22) + per-phase blend weight instead of one global "
        "scalar alpha."
    ),
    "seeds": seeds,
    "reference_v11_live": v11,
    "reference_v21_single_nn_global_blend": v21_ensemble,
    "bagged_nn_alone_holdout": nn_alone_hit,
    "bagged_nn_global_blend_holdout": global_hit,
    "global_alpha": global_alpha,
    "bagged_nn_per_phase_blend_holdout": phase_blend_hit,
    "phase_alphas": phase_alphas,
    "beats_v21_with_global_blend": bool(global_hit["blended"] > v21_ensemble["blended"]),
    "beats_v21_with_phase_blend": bool(phase_blend_hit["blended"] > v21_ensemble["blended"]),
}
(output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
