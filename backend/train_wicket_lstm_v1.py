"""LSTM-over-ball-sequences prototype for the wicket target (2026-08-24).

Standing thread: six-plus real single-feature additions to
`contract22_wicket_v16_team_composition` (currently live) have all been
rejected in a row, and feature importance is diffuse (32-33/61-63 features
needed for 80% of importance) -- evidence of a real ceiling on the current
flat/single-row GBM architecture, not "haven't found the right feature yet."
Proposed lever: model structure itself. This is the first real test --
a plain LSTM over the raw ball-by-ball sequence of each innings so far
(minimal per-ball features, no player identity, no engineered rolling
windows -- deliberately isolates whether sequence ORDER carries signal a
flat/aggregated view misses), predicting `wicket_in_over` for the upcoming
over. Same leakage-safe target, same eligible-file filter, same
train(<=2023)/calibration(2024)/holdout(2025+) split, same IPL/T20I split
convention as every other wicket candidate in this project.

Live baseline for direct comparison (`contract22_wicket_v16_team_composition`
validation_report.json, reproduced here, not re-derived from a stale
number): AUC 0.6126 blended known-bowler / 0.6114 unknown-bowler; IPL 0.6151
known / 0.6131 unknown; T20I 0.6113 known / 0.6113 unknown. This LSTM has no
bowler-identity feature at all, so the fairest comparison is the
bowler-unknown cut -- reported anyway against both for honesty.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from app.ml.wicket_sequence_dataset import N_FEATURES, build_wicket_sequences
from train_phase_calibrated_sharp_range_v33 import male_source_files

VERSION = "wicket_lstm_v1"
MAX_LEN = 120
HIDDEN = 64
BATCH_SIZE = 256
MAX_EPOCHS = 30
PATIENCE = 4
LR = 1e-3

REFERENCE_V16 = {
    "auc_known_blended": 0.6126, "auc_unknown_blended": 0.6114,
    "auc_known_ipl": 0.6151, "auc_unknown_ipl": 0.6131,
    "auc_known_t20i": 0.6113, "auc_unknown_t20i": 0.6113,
}


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[np.ndarray], labels: np.ndarray):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


def collate(batch):
    seqs, labels = zip(*batch)
    lengths = [max(1, len(s)) for s in seqs]
    max_len = max(lengths)
    padded = torch.zeros(len(seqs), max_len, N_FEATURES, dtype=torch.float32)
    for i, s in enumerate(seqs):
        if len(s) == 0:
            continue
        padded[i, : len(s)] = torch.from_numpy(s)
    lengths_t = torch.tensor(lengths, dtype=torch.int64)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    order = torch.argsort(lengths_t, descending=True)
    return padded[order], lengths_t[order], labels_t[order]


class WicketLSTM(nn.Module):
    def __init__(self, input_size: int = N_FEATURES, hidden: int = HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
        )

    def forward(self, padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (h_n, _) = self.lstm(packed)
        return self.head(h_n[-1]).squeeze(-1)


def split_frame(root: Path) -> pd.DataFrame:
    eligible = male_source_files(root)
    base = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    base = base[base["source_file"].isin(eligible)].copy()
    base["match_date"] = pd.to_datetime(base["match_date"])
    ipl_files = {p.name for p in (root / "data/raw/cricsheet/ipl").glob("*.json")}
    base["is_ipl"] = base["source_file"].isin(ipl_files)
    return base


def to_dataset(frame: pd.DataFrame, sequences: dict) -> SequenceDataset:
    seqs, labels = [], []
    missing = 0
    for row in frame.itertuples(index=False):
        key = (row.source_file, row.innings, row.over)
        seq = sequences.get(key)
        if seq is None:
            missing += 1
            continue
        seqs.append(seq)
        labels.append(float(row.wicket_in_over))
    if missing:
        print(f"  warning: {missing} rows had no matching sequence (skipped)")
    return SequenceDataset(seqs, np.asarray(labels, dtype=np.float32))


@torch.no_grad()
def predict_proba(model: WicketLSTM, dataset: SequenceDataset, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=False, collate_fn=collate)
    all_probs, all_labels = [], []
    for padded, lengths, labels in loader:
        logits = model(padded.to(device), lengths)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def _metrics(actual: np.ndarray, proba: np.ndarray) -> dict:
    if len(np.unique(actual)) < 2:
        return {"rows": int(len(actual)), "event_rate": float(actual.mean())}
    return {
        "rows": int(len(actual)),
        "event_rate": float(actual.mean()),
        "auc": float(roc_auc_score(actual, proba)),
        "brier": float(brier_score_loss(actual, proba)),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "models/candidates" / VERSION
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading labels + splitting...")
    base = split_frame(root)
    train_f = base[base["match_date"] <= "2023-12-31"].reset_index(drop=True)
    cal_f = base[base["match_date"].dt.year == 2024].reset_index(drop=True)
    holdout_f = base[base["match_date"] >= "2025-01-01"].reset_index(drop=True)
    print(f"  train={len(train_f)} cal={len(cal_f)} holdout={len(holdout_f)}")

    print("Building raw ball sequences from Cricsheet JSON...")
    t0 = time.time()
    sequences = build_wicket_sequences(root, max_len=MAX_LEN)
    print(f"  {len(sequences)} sequences built in {time.time() - t0:.1f}s")

    print("Assembling datasets...")
    train_ds = to_dataset(train_f, sequences)
    cal_ds = to_dataset(cal_f, sequences)
    holdout_ds = to_dataset(holdout_f, sequences)

    pos = train_ds.labels.sum()
    neg = len(train_ds.labels) - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    model = WicketLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    best_auc = -1.0
    best_state = None
    epochs_since_improve = 0

    print("Training...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for padded, lengths, labels in train_loader:
            optimizer.zero_grad()
            logits = model(padded.to(device), lengths)
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        train_loss = total_loss / len(train_ds)

        cal_probs, cal_labels = predict_proba(model, cal_ds, device)
        cal_auc = roc_auc_score(cal_labels, cal_probs)
        print(f"  epoch {epoch}: train_loss={train_loss:.4f} cal_auc={cal_auc:.4f} ({time.time() - t0:.1f}s)")

        if cal_auc > best_auc:
            best_auc = cal_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= PATIENCE:
                print(f"  early stopping at epoch {epoch} (best cal_auc={best_auc:.4f})")
                break

    model.load_state_dict(best_state)

    print("Evaluating on real 2025+ holdout...")
    cal_probs, cal_labels = predict_proba(model, cal_ds, device)
    holdout_probs, holdout_labels = predict_proba(model, holdout_ds, device)

    clipped = np.clip(cal_probs, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    platt = LogisticRegression(C=1.0, random_state=42).fit(logit, cal_labels)
    holdout_clipped = np.clip(holdout_probs, 1e-6, 1 - 1e-6)
    holdout_logit = np.log(holdout_clipped / (1 - holdout_clipped)).reshape(-1, 1)
    platt_proba = platt.predict_proba(holdout_logit)[:, 1]

    # is_ipl must align with exactly the rows that survived to_dataset (in case of skips)
    kept_mask = np.array([
        (row.source_file, row.innings, row.over) in sequences
        for row in holdout_f.itertuples(index=False)
    ])
    holdout_is_ipl = holdout_f["is_ipl"].to_numpy()[kept_mask]

    result = {
        "auc": float(roc_auc_score(holdout_labels, holdout_probs)),
        "brier_platt": float(brier_score_loss(holdout_labels, platt_proba)),
        "brier_uncalibrated": float(brier_score_loss(holdout_labels, holdout_probs)),
        "event_rate": float(holdout_labels.mean()),
        "rows": int(len(holdout_labels)),
        "ipl_platt": _metrics(holdout_labels[holdout_is_ipl], platt_proba[holdout_is_ipl]),
        "t20i_platt": _metrics(holdout_labels[~holdout_is_ipl], platt_proba[~holdout_is_ipl]),
    }

    report = {
        "candidate_version": VERSION,
        "candidate_only": True,
        "production_changed": False,
        "note": (
            "Plain single-layer LSTM over raw per-ball sequence (9 minimal "
            "features/ball, no player identity, no engineered rolling "
            "windows, hidden=64) predicting wicket_in_over. Tests whether "
            "sequence order itself carries signal beyond the live GBM's "
            "flat/aggregated features. No bowler-identity feature at all -- "
            "compare primarily against the live model's bowler_unknown cut."
        ),
        "architecture": {
            "hidden_size": HIDDEN, "max_seq_len": MAX_LEN,
            "features_per_ball": N_FEATURES, "best_epoch_cal_auc": best_auc,
        },
        "split": {"train_rows": len(train_ds), "cal_rows": len(cal_ds), "holdout_rows": len(holdout_ds)},
        "reference_contract22_wicket_v16_team_composition": REFERENCE_V16,
        "holdout_result": result,
    }

    torch.save(model.state_dict(), output / "wicket_lstm.pt")
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
