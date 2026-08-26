# Cascade features (run-range ↔ wicket): real idea, closed, no signal either direction

Date: 2026-08-25. Closes the standing thread flagged 2026-08-23 ("feed
run-range's own expected-runs prediction as an input feature to the
wicket model, and/or wicket's probability into run-range, testing
whether the two already-separate, already-fault-isolated engines'
outputs carry residual signal for each other") -- never decided at the
time ("let me think"), picked up now per the user's own "any thoughts
for model that needs update" prompt, given the Monte Carlo win-
probability work is now blocked on live-match availability for months
(IPL 2026 season already concluded in May; next season ~March 2027).

**Explicitly not a proposal to merge or restructure engines** -- run-
range/wicket/win-probability stay separate, exactly as they are; this
only adds one new scalar feature to each model's existing input.

## Construction

Built `app/ml/cascade_features.py`: each engine's own predicted value
for a given over (run-range's expected runs; wicket's wicket_in_over
probability), generated **leakage-safely** -- a model's raw prediction on
its own training rows is an inflated, non-generalizing signal, so this
uses 5-fold `GroupKFold` (grouped by `source_file`, so no single match's
overs span both a training and validation fold -- same-match rows are
too correlated for a naive row-level split) over the train population,
with real single-pass out-of-sample predictions for calibration/holdout
(already genuinely unseen once a model is fit on the full training
period).

Both base dataset builders (`build_wicket_base_dataset`,
`build_run_range_base_dataset`) exactly replicate the merges from the
live `contract22_wicket_v16_team_composition.py` and
`train_run_range_v11_partnership_rate.py` scripts respectively (this
project's own convention: each new candidate rebuilds from the shared
`app/ml/*.py` dataset functions rather than importing another script's
top-level state) -- verified both produce the identical row count
(172,631) as their live counterparts before trusting anything downstream.

## Result -- real, symmetric negative

**`contract22_wicket_v18_cascade_features`** (run-range's expected-runs
cascaded into wicket): AUC flat-to-worse on almost every cut vs. live
`v16` -- known blended 0.6123 vs 0.6126, known IPL 0.6155 vs 0.6151 (a
negligible +0.0004), known T20I 0.6108 vs 0.6113, **unknown blended
0.6103 vs 0.6114, unknown IPL 0.6104 vs 0.6131** (the harder, more
common real live-serving case, and the clearest real regression).

**`run_range_v16_cascade_features`** (wicket's probability cascaded into
run-range): hit rate worse on every cut vs. live `v11` -- blended 28.56%
vs 28.80%, IPL 24.18% vs 25.16% (a real, nearly 1-point drop), T20I
29.36% vs 29.45%. `decision: reject_keep_research` (script's own
mechanical gate).

**The informative wrinkle**: on the wicket side, the cascade feature
ranked **#1 of 62 features** by importance -- the model leans on it
heavily -- yet real holdout AUC didn't improve and got measurably worse
on the harder unknown-bowler IPL cut. Same "high importance does not
imply helpful" pattern this project already found once before (venue
recency, 2026-08-23, `finding_venue_recency_ipl_regression.md`): the
model isn't ignoring the feature, it's substituting it for something it
already had a better-tuned internal representation of, net loss.

## Verdict

**Not promoted, either direction.** Closed cleanly with a real, honest
negative on both sides of the reciprocal test -- the two engines' own
outputs don't carry meaningful complementary signal for each other,
beyond what the shared match-state features (wickets in hand, run rate,
phase, recency) already give each model directly. No code/model changes
to either live model. New infrastructure kept (research-only):
`app/ml/cascade_features.py`,
`backend/train_contract22_wicket_v18_cascade_features.py`,
`backend/train_run_range_v16_cascade_features.py`,
`models/candidates/contract22_wicket_v18_cascade_features/`,
`models/candidates/run_range_v16_cascade_features/`.

This closes the last standing untested single-feature idea from the
2026-08-23/24 sessions. Combined with the earlier LSTM sequence-model
closure (`finding_wicket_lstm_sequence_model_not_promoted.md`), both of
this session's "structural" levers (sequence modeling, cross-engine
cascading) are now exhausted for the wicket/run-range targets on this
data. Remaining open threads: the still-blocked live-pipeline validation
(no live IPL/T20I match this whole project), the Monte Carlo win-
probability live comparison (blocked on the same live-match gap, now
~7 months out for IPL specifically), and the bigger, not-yet-attempted
full-feature neural-net-vs-GBM architecture comparison flagged in the
LSTM finding doc.
