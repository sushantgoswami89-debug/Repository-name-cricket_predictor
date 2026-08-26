"""Build production artifacts for the run-range GBM+NN ensemble
(2026-08-26) -- see docs/finding_run_range_nn_gbm_ensemble_real_win.md for
the validated result (beats the live run_range_v11_partnership_rate GBM
alone on all three cuts: blended +0.18pp, IPL +0.18pp, T20I +0.20pp).

Reuses the train/calibration feature caches already written by
train_run_range_v21a_gbm_only.py (same artifact directory,
models/candidates/run_range_v21_nn_gbm_ensemble/) -- no need to rebuild
the ~10 dataset joins a third time.

Saves: gbm_model.pkl, feature_cols.pkl, categorical_cols.pkl, nn_model.pt,
nn_metadata.json, phase_temperatures.pkl (fit on the BLENDED distribution,
not the GBM alone), blend_config.json (the scalar blend alpha),
ARTIFACT_MANIFEST.json.

Run as three parts (env RUN_RANGE_ENSEMBLE_PART=gbm|nn|combine) since
torch and lightgbm cannot coexist in one process on this machine."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PART = os.environ.get("RUN_RANGE_ENSEMBLE_PART", "gbm")
project_root = Path(__file__).resolve().parent.parent
output = project_root / "models/candidates/run_range_v21_nn_gbm_ensemble"
output.mkdir(parents=True, exist_ok=True)

CAT = [
    "phase", "active_batter_state", "wickets_remaining_bucket",
    "state_regime", "chase_pressure", "venue_scoring_regime",
    "venue_par_source", "batting_team_venue_context", "phase_venue_regime",
    "striker_batting_style", "bowler_type",
]


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if PART == "gbm":
    import joblib
    import lightgbm as lgb
    from train_phase_calibrated_sharp_range_v33 import MAX_RUN_CLASS

    train = pd.read_pickle(output / "train_cache.pkl")
    cal = pd.read_pickle(output / "calibration_cache.pkl")
    FEATURES = [c for c in train.columns if c not in ("runs_in_over", "is_ipl")]

    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=MAX_RUN_CLASS + 1,
        n_estimators=60, learning_rate=0.08, num_leaves=25, max_depth=6,
        min_child_samples=100, subsample=0.85, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=42, verbose=-1,
    )
    train_actual = np.minimum(train["runs_in_over"].to_numpy(), MAX_RUN_CLASS)

    def frame(d):
        r = d[FEATURES].copy()
        for c in CAT:
            r[c] = r[c].fillna("__UNKNOWN__").astype("category")
        return r

    print("Training GBM...", flush=True)
    model.fit(frame(train), train_actual, categorical_feature=CAT)
    joblib.dump(model, output / "gbm_model.pkl")
    joblib.dump(FEATURES, output / "feature_cols.pkl")
    joblib.dump(CAT, output / "categorical_cols.pkl")

    cal_actual = np.minimum(cal["runs_in_over"].to_numpy(), MAX_RUN_CLASS)
    np.save(output / "gbm_cal_raw.npy", model.predict_proba(frame(cal)))
    np.save(output / "cal_actual.npy", cal_actual)
    np.save(output / "cal_phase.npy", cal["phase"].astype(str).to_numpy())
    print("Saved gbm_model.pkl + feature_cols.pkl + categorical_cols.pkl + gbm_cal_raw.npy")

elif PART == "nn":
    import torch
    from torch import nn
    from train_phase_calibrated_sharp_range_v33 import MAX_RUN_CLASS

    train = pd.read_pickle(output / "train_cache.pkl")
    cal = pd.read_pickle(output / "calibration_cache.pkl")
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
    net = Net().to(device)
    train_num, train_cat = to_tensors(train)
    cal_num, cal_cat = to_tensors(cal)
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
    torch.save(net.state_dict(), output / "nn_model.pt")
    metadata = {
        "numeric": NUMERIC, "categorical": CAT,
        "cat_maps": cat_maps, "cat_dims": cat_dims,
        "num_mean": num_mean.to_dict(), "num_std": num_std.to_dict(),
        "num_classes": NUM_CLASSES,
    }
    (output / "nn_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    net.eval()
    with torch.no_grad():
        nn_cal_logit = net(cal_num.to(device), cal_cat.to(device)).cpu().numpy()
    nn_prob = np.exp(nn_cal_logit - nn_cal_logit.max(axis=1, keepdims=True))
    nn_prob /= nn_prob.sum(axis=1, keepdims=True)
    np.save(output / "nn_cal_raw.npy", nn_prob)
    print("Saved nn_model.pt + nn_metadata.json + nn_cal_raw.npy")

elif PART == "combine":
    import joblib
    from scipy.optimize import minimize_scalar
    from train_phase_calibrated_sharp_range_v33 import nll, temperature_scale

    cal_actual = np.load(output / "cal_actual.npy")
    cal_phase = np.load(output / "cal_phase.npy", allow_pickle=True)
    gbm_cal_raw = np.load(output / "gbm_cal_raw.npy")
    nn_cal_raw = np.load(output / "nn_cal_raw.npy")

    def blend_nll(alpha: float) -> float:
        return nll(alpha * gbm_cal_raw + (1 - alpha) * nn_cal_raw, cal_actual)

    result = minimize_scalar(blend_nll, bounds=(0.0, 1.0), method="bounded")
    alpha = float(result.x)
    blended_cal = alpha * gbm_cal_raw + (1 - alpha) * nn_cal_raw

    temperatures = {}
    for phase in ("powerplay", "middle", "death"):
        mask = cal_phase == phase
        r = minimize_scalar(
            lambda value: nll(temperature_scale(blended_cal[mask], value), cal_actual[mask]),
            bounds=(0.5, 3.0), method="bounded",
        )
        temperatures[phase] = float(r.x)

    joblib.dump(temperatures, output / "phase_temperatures.pkl")
    (output / "blend_config.json").write_text(json.dumps({"alpha_gbm_weight": alpha}, indent=2), encoding="utf-8")

    manifest_files = [
        "gbm_model.pkl", "feature_cols.pkl", "categorical_cols.pkl",
        "nn_model.pt", "nn_metadata.json",
        "phase_temperatures.pkl", "blend_config.json",
    ]
    manifest = {
        "candidate_version": "run_range_v21_nn_gbm_ensemble",
        "artifacts": {name: _sha256(output / name) for name in manifest_files},
    }
    (output / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"blend alpha (GBM weight) = {alpha:.4f}, phase temperatures = {temperatures}")
    print("Saved phase_temperatures.pkl, blend_config.json, ARTIFACT_MANIFEST.json")

else:
    raise ValueError(f"Unknown RUN_RANGE_ENSEMBLE_PART={PART!r}, expected gbm|nn|combine")
