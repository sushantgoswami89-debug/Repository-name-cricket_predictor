# Situation-aware blend weighting for the wicket ensemble: real signal, no net improvement

Date: 2026-08-25. Direct follow-up to the wicket GBM+NN ensemble
(`docs/finding_wicket_nn_gbm_ensemble_real_win.md`), which blends both
models with one fixed, globally-learned weight (0.614 GBM / 0.423 NN).
User pushed back on the hard-coded weight: "it should be situation
specific, we should define a logic and boundaries." Four real attempts,
all using the already-trained GBM/NN predictions (no retraining needed,
just re-fitting the blend), same real 2025+ holdout.

## v19d: blunt situational features (phase, wickets-in-hand interactions)

Added phase/wickets-in-hand/is_ipl and their interactions with each
model's logit (11 parameters vs. the flat blend's 2). **Result: worse**
than the flat blend on blended and T20I, roughly flat on IPL. Read: too
many parameters for what the 2024 calibration set (~24.7k rows) can
reliably estimate -- picks up noise, not real situational structure.

## v19e: retrospective match-character diagnostic (not a live signal)

Split the real holdout into close-finish / blowout / swing (even at over
10, decisively not close by the end) / moderate matches, using the
*final* result. Genuinely revealing: NN beat GBM specifically in close
finishes (0.604 vs 0.602 AUC) for the first time all session; GBM won
clearly in swing/moderate matches; ensemble was slightly worse than GBM
alone in swing/moderate. But these categories are defined by the match
outcome -- not knowable live -- so this is a diagnostic explaining *why*
a flat blend is a reasonable compromise, not a usable routing signal.

## v19f: live chase-equation volatility/momentum

A live-available proxy for v19e's insight: rolling volatility/momentum
of (current_run_rate - required_run_rate) over the trailing 5 overs.
**No signal** -- blended AUC 0.61363 vs flat's 0.61365, coefficients on
volatility/momentum and their interactions all near-zero. Only meaningful
for chases (current_run_rate - 0 for innings 1), diluting relevance.

## v19g: live scoring/wicket-clustering volatility

A more universal live volatility concept: rolling std of runs-per-over
and wickets-in-last-5-overs (applies to every innings, not just chases).
**Also no signal, and slightly worse than v19f** -- blended AUC 0.6131 vs
flat's 0.6137, fitted weights drifted toward 50/50 (0.54/0.52) without
gaining anything back for the extra flexibility.

## Verdict

**Closed -- the flat, globally-fixed blend weight (0.614 GBM / 0.423 NN)
remains the best real option found.** Four attempts at "situation-aware"
weighting, two different mechanisms (blunt interaction features; two
different live volatility formulations), zero net improvement -- every
attempt either matched or underperformed the flat blend. The one genuine
insight (v19e's close-finish/swing/blowout split) is real but not
operationalizable without knowing the future, and the live proxies tested
for it don't capture what makes those categories distinct.

No live model changes -- `contract22_wicket_v16_team_composition` remains
the served wicket prediction; the flat-weighted ensemble continues
running as the logged shadow, unchanged by this investigation. New
infrastructure kept (research-only): `train_wicket_v19d_situational_stacker.py`,
`train_wicket_v19e_match_character.py`,
`train_wicket_v19f_live_volatility_stacker.py`,
`train_wicket_v19g_scoring_volatility_stacker.py`, and their JSON reports
under `models/candidates/wicket_v19_nn_gbm_ensemble/`.
