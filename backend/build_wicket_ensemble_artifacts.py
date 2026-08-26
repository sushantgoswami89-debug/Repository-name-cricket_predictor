"""Build production artifacts for the wicket GBM+NN ensemble (2026-08-25)
-- see docs/finding_wicket_nn_gbm_ensemble_real_win.md for the validated
result. Saves: gbm_model.pkl, gbm_calibrator.pkl, nn_model.pt,
nn_metadata.json (preprocessing needed for live inference), nn_calibrator.pkl,
stacker.pkl, ARTIFACT_MANIFEST.json. Run as two halves (GBM section then
NN section) is unnecessary here since this script itself never imports
torch and lightgbm together in a way that trains both -- it imports both
at the top, which IS the exact combination that hangs/dies on this
machine. So: run with the GBM section only first (env WICKET_ENSEMBLE_PART=gbm),
then NN section (WICKET_ENSEMBLE_PART=nn), then combine (WICKET_ENSEMBLE_PART=combine)."""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np

PART = os.environ.get("WICKET_ENSEMBLE_PART", "gbm")
root = Path(__file__).resolve().parents[1]
output = root / "models/candidates/wicket_v19_nn_gbm_ensemble"
output.mkdir(parents=True, exist_ok=True)


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
    from app.ml.cascade_features import WICKET_CATEGORICAL, WICKET_FEATURES, build_wicket_base_dataset

    CAT = WICKET_CATEGORICAL
    print("Building dataset...", flush=True)
    data = build_wicket_base_dataset(root)
    train = data[data["match_date"] <= "2023-12-31"].reset_index(drop=True)
    cal = data[data["match_date"].dt.year == 2024].reset_index(drop=True)

    def frame(d):
        r = d[WICKET_FEATURES].copy()
        for c in CAT:
            r[c] = r[c].fillna("__UNKNOWN__").astype("category")
        return r

    print("Training GBM...", flush=True)
    gbm = lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.03, max_depth=5,
                              num_leaves=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    gbm.fit(frame(train), train["wicket_in_over"], categorical_feature=CAT)
    joblib.dump(gbm, output / "gbm_model.pkl")
    joblib.dump(WICKET_FEATURES, output / "feature_cols.pkl")
    joblib.dump(CAT, output / "categorical_cols.pkl")
    np.save(output / "gbm_cal_raw.npy", gbm.predict_proba(frame(cal))[:, 1])
    np.save(output / "cal_y.npy", cal["wicket_in_over"].to_numpy())
    print("Saved gbm_model.pkl + feature_cols.pkl + categorical_cols.pkl")

elif PART == "nn":
    import torch
    from torch import nn
    from app.ml.cascade_features import WICKET_CATEGORICAL, WICKET_FEATURES, build_wicket_base_dataset

    NUMERIC = [c for c in WICKET_FEATURES if c not in WICKET_CATEGORICAL]
    CAT = WICKET_CATEGORICAL
    print("Building dataset...", flush=True)
    data = build_wicket_base_dataset(root)
    train = data[data["match_date"] <= "2023-12-31"].reset_index(drop=True)
    cal = data[data["match_date"].dt.year == 2024].reset_index(drop=True)

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
    train_y = torch.tensor(train["wicket_in_over"].to_numpy(), dtype=torch.float32)
    pos_weight = torch.tensor([(len(train_y) - train_y.sum()) / train_y.sum()])

    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    n = len(train)
    batch = 1024
    best_auc, best_state, patience = -1, None, 0
    from sklearn.metrics import roc_auc_score
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
    torch.save(net.state_dict(), output / "nn_model.pt")
    metadata = {
        "numeric": NUMERIC, "categorical": CAT,
        "cat_maps": cat_maps, "cat_dims": cat_dims,
        "num_mean": num_mean.to_dict(), "num_std": num_std.to_dict(),
        "hidden": 128,
    }
    (output / "nn_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    net.eval()
    with torch.no_grad():
        nn_cal_logit = net(cal_num.to(device), cal_cat.to(device)).cpu().numpy()
    np.save(output / "nn_cal_raw.npy", 1 / (1 + np.exp(-nn_cal_logit)))
    print("Saved nn_model.pt + nn_metadata.json")

elif PART == "combine":
    import joblib
    from sklearn.linear_model import LogisticRegression

    cal_y = np.load(output / "cal_y.npy")
    gbm_cal_raw = np.load(output / "gbm_cal_raw.npy")
    nn_cal_raw = np.load(output / "nn_cal_raw.npy")

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    gbm_calibrator = LogisticRegression(C=1.0, random_state=42).fit(logit(gbm_cal_raw).reshape(-1, 1), cal_y)
    nn_calibrator = LogisticRegression(C=1.0, random_state=42).fit(logit(nn_cal_raw).reshape(-1, 1), cal_y)
    gbm_cal_platt = gbm_calibrator.predict_proba(logit(gbm_cal_raw).reshape(-1, 1))[:, 1]
    nn_cal_platt = nn_calibrator.predict_proba(logit(nn_cal_raw).reshape(-1, 1))[:, 1]
    stacker = LogisticRegression(C=1.0, random_state=42).fit(
        np.column_stack([logit(gbm_cal_platt), logit(nn_cal_platt)]), cal_y)

    joblib.dump(gbm_calibrator, output / "gbm_calibrator.pkl")
    joblib.dump(nn_calibrator, output / "nn_calibrator.pkl")
    joblib.dump(stacker, output / "stacker.pkl")

    manifest_files = [
        "gbm_model.pkl", "feature_cols.pkl", "categorical_cols.pkl",
        "nn_model.pt", "nn_metadata.json",
        "gbm_calibrator.pkl", "nn_calibrator.pkl", "stacker.pkl",
    ]
    manifest = {"candidate_version": "wicket_v19_nn_gbm_ensemble",
                "artifacts": {name: _sha256(output / name) for name in manifest_files}}
    (output / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"stacker coefficients [gbm, nn]: {stacker.coef_}, intercept: {stacker.intercept_}")
    print("Saved gbm_calibrator.pkl, nn_calibrator.pkl, stacker.pkl, ARTIFACT_MANIFEST.json")

else:
    raise ValueError(f"Unknown WICKET_ENSEMBLE_PART={PART!r}, expected gbm|nn|combine")
