"""Fit an isotonic-regression calibrator for the raw wicket model's output
probabilities, using the tuning sample (seed 123, distinct from every
reporting sample used in this session) to avoid overfitting the numbers
we report. Saves models/wkt_calibrator.pkl."""

import random
from pathlib import Path

import joblib
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from app.data_ingestion.json_loader import JsonLoader
from app.data_ingestion.match_parser import MatchParser
from app.ml.feature_builder import FeatureBuilder
from app.ml.historical_feature_store import match_exclude_key
from app.ml.model_repository import ModelRepository
from app.models.match_context import MatchContext
from app.services.match_replay import MatchReplay

N_MATCHES = 20
random.seed(123)
match_dir = Path("../data/raw/cricsheet/ipl")
sample = random.sample(sorted(match_dir.glob("*.json")), N_MATCHES)

repository = ModelRepository()
builder = FeatureBuilder()
feature_order = repository.get_feature_columns()
cat_cols = repository.get_categorical_columns()

probs = []
labels = []
for path in sample:
    try:
        match = MatchParser().parse(JsonLoader().load(str(path)))
    except Exception:
        continue
    exclude_key = match_exclude_key(match)
    for innings in match.innings:
        for frame in MatchReplay().frames(innings):
            context = MatchContext(
                team1=match.info.teams[0], team2=match.info.teams[1],
                venue=match.info.venue or "Unknown", format=match.info.match_type,
                live=frame.state, metadata={"exclude_match_key": exclude_key},
            )
            features = builder.build(context)
            ordered = {name: features[name] for name in feature_order}
            df = pd.DataFrame([ordered])
            for col in cat_cols:
                if col in df.columns:
                    df[col] = df[col].astype("category")
            wkt_prob = float(repository.get_wicket_model().predict_proba(df)[0][1])
            probs.append(wkt_prob)
            labels.append(1 if frame.actual_wickets > 0 else 0)

calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
calibrator.fit(probs, labels)

out_path = Path("../models/wkt_calibrator.pkl")
joblib.dump(calibrator, out_path)
print(f"Saved calibrator to {out_path} (fit on n={len(labels)})")

# Quick sanity: calibrated Brier on the SAME fitting sample (expect improvement)
calibrated = calibrator.predict(probs)
from sklearn.metrics import brier_score_loss
print("Brier before calibration (fit sample):", brier_score_loss(labels, probs))
print("Brier after calibration  (fit sample):", brier_score_loss(labels, calibrated))
