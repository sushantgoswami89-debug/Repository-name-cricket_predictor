# Phase-specific bowler recency: real IPL signal, but doesn't survive the standard competition-split fix — not promoted

Date: 2026-08-23. Follow-up to `finding_recency_weighted_form.md` and
`finding_partnership_scoring_rate.md`: bowler recency features ranked
notably lower than batter recency features in the `v9` promotion
(`bowler_recency_economy` 28th of 49, vs `striker_recency_balls` 4th).
Hypothesis: bowlers are often phase specialists (strong at death, weak in
the powerplay, or vice versa) -- a phase-agnostic recency average blurs
together very different situations that the existing flat
`bowl_phase_avg_runs` feature already knows to keep separate, just
without recency weighting. This combines both levers.

## What was built

`app/ml/recency_weighted_prior_dataset.py` gained
`build_bowler_phase_recency_dataset` (a separate function, not touching
the already-promoted `build_recency_weighted_prior_dataset` used by live
`v9`/`v10`, to keep those exactly reproducible) and
`compute_final_recency_state` now also returns a third
`bowler_phase_form` dict (keyed `"{bowler_id}|{phase}"`). Live snapshot
builder (`build_recency_weighted_live_snapshots.py`) updated and re-run
-- confirmed the existing batter/bowler snapshot counts (7,061/5,205)
are unchanged, so this didn't disturb what `v9`/`v10` actually serve.

## Tested against the live baseline (v10): a real split, not a clean win

`train_contract22_wicket_v11_bowler_phase_recency.py`, on top of
currently-live `contract22_wicket_v10_partnership_rate`:

| | AUC known | AUC unknown |
|---|---:|---:|
| Live `v10`, IPL | 0.6104 | 0.6059 |
| `v11`, IPL | **0.6129** | **0.6075** |
| Live `v10`, T20I | 0.6113 | 0.6110 |
| `v11`, T20I | **0.6108** | **0.6100** |

IPL improves meaningfully (+0.0025 known, +0.0016 unknown). T20I gets
*worse* on both cuts. Blended is roughly a wash (known flat, unknown
-0.0006). This is the same shape of competition-specific effect already
seen for run-range band width
(`finding_blended_holdout_masks_ipl_accuracy.md`) -- plausibly because
IPL bowlers have denser, more role-consistent phase data (a death-overs
specialist stays a death-overs specialist across IPL seasons), while
T20I bowlers see more heterogeneous opposition/role variability across
national contexts, making a phase-specific recency split noisier there.

## Tried the standard fix: didn't cleanly work

`train_contract22_wicket_v12_bowler_phase_recency_ipl_only.py`: masked
`bowler_recency_phase_*` to 0 for T20I rows in both train and eval (same
convention as `bowler_known=False` masking elsewhere), the exact fix
this project already uses for this problem shape (IPL band width). If
the T20I regression were caused by noisy T20I-specific feature values
contaminating the fit, masking should isolate the IPL gain cleanly.

It didn't. T20I is *still* worse (0.6107 known / 0.6099 unknown vs
baseline 0.6113/0.6110) even with the feature forced to a constant 0 for
every T20I row -- meaning the regression isn't really about noisy T20I
values, it's some other generalization effect from the extra columns
changing the model's overall fit. The IPL numbers also reshuffled
unpredictably: known dropped from v11's +0.0025 to +0.0006, but unknown
jumped to +0.0055 (the single largest cut improvement seen all session)
-- a pattern with no obvious causal story, more consistent with real
noise on a small holdout (IPL is only 5,576 rows) than a stable effect.

## Recommendation

**Not promoted, either formulation.** Unlike the recency-form and
partnership-rate promotions (where every reported cut improved,
consistently, across reruns), this lever produces a genuine but unstable
effect that doesn't resolve cleanly with the tool this project normally
reaches for. Rather than force a promotion on a result this noisy,
leaving `contract22_wicket_v10_partnership_rate` as the live model is the
honest call. The infrastructure (`build_bowler_phase_recency_dataset`,
the third live snapshot) is committed and real -- if bowler-side recency
is revisited later, a larger IPL-specific holdout or a different
phase-specificity threshold (e.g. only apply it to bowlers with enough
phase-specific history) would be the next thing to try, not more
masking variants on this exact formulation.

Candidate artifacts: `models/candidates/contract22_wicket_v11_bowler_phase_recency/`,
`models/candidates/contract22_wicket_v12_bowler_phase_recency_ipl_only/`.
Neither promoted; `PredictionEngine` remains on `v10`.
