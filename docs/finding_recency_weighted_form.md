# Recency-weighted form: helps wicket prediction, doesn't help run-range — promoted the wicket side

Date: 2026-08-23. User-flagged direction: "we have enough data but we
aren't thinking enough about how to use it... weightage... economy
rates." Traced every prior-stat feature this codebase has ever used
(`striker_prior_runs_per_ball` in `ipl_phase_moe_dataset.py`,
`bowl_hist_avg_runs_conceded` in `train_contract22_rigorous.py`) and
found all of them are flat career-to-date cumulative averages --
`profile[field] += value`, applied once per match, no decay. A player's
rate from 40 matches ago has always counted exactly as much as last
week's. Recency-weighted form has never been tested in this codebase.

## What was built

`app/ml/recency_weighted_prior_dataset.py`: a match-indexed EWMA
(`state[field] = 0.95 * state[field] + this_match_event[field]`, ~13-14
match half-life, roughly one IPL season) added *alongside* every existing
flat feature, for both batter (runs/dots/boundaries/dismissals per ball)
and bowler (economy, wicket rate). Not a tuned/swept decay value --
deliberately one defensible default, per this project's own standing
lesson that hyperparameter sweeping on a single lever/architecture
combination is a dead end once tried a few times
(`candidate_ipl_wicket_v7_2_spell_features.md`).

Spot-checked against Virat Kohli's real IPL career before trusting it
(per this project's standing practice: verify a real, checkable name, not
just aggregate stats) -- his recency-weighted strike rate diverges from
his flat career average in directions consistent with known form swings
across seasons, not degenerate noise.

## Tested against both real live baselines

**Run-range** (`train_run_range_v10_recency_form.py`, vs. live `run_range_v7_competition_prior`): **rejected**.

| | Blended | IPL |
|---|---:|---:|
| Live `v7` | 28.75% | 25.00% |
| `v10` + recency | 28.55% | 24.18% |

Worse on both. Recency features ranked mid-to-low in importance (15th-36th of 69).

**Wicket** (`train_contract22_wicket_v9_recency_form.py`, vs. live `contract22_wicket_v2_batter_state`): **real, robust win, promoted.**

| | AUC known | AUC unknown | Brier known |
|---|---:|---:|---:|
| Live `v2`, blended | 0.6081 | 0.6072 | 0.2049 |
| `v9`, blended | 0.6106 | 0.6085 | 0.2047 |
| Live `v2`, IPL only | 0.6079 | 0.6043 | 0.1944 |
| `v9`, IPL only | 0.6097 | 0.6058 | 0.1940 |
| Live `v2`, T20I only | 0.6077 | 0.6072 | 0.2069 |
| `v9`, T20I only | 0.6100 | 0.6090 | 0.2067 |

Every one of the 6 reported AUC cuts improves; Brier improves everywhere
too. Recency features rank genuinely high in importance --
`striker_recency_balls` 4th of 49, `partner_recency_runs_per_ball` 7th,
`striker_recency_dismissal_rate` 8th. Not cherry-picked, not driven by
one competition.

## Why the split makes sense

A batter's *current* dismissal tendency is real, live signal for
wicket-in-over risk. An over's total *run count*, though, is dominated by
so much other match-state variance (required rate, phase, boundary
variance) that a modest refinement to the batter-quality estimate doesn't
move a discrete-band classification target. Consistent with the general
shape of this session's findings: not every real signal helps every
target equally.

## Promoted to production (contract22_wicket_v9_recency_form)

Per this project's own newly-corrected promotion policy
(`finding_wicket_alert_gate_is_vestigial.md` -- Brier/AUC improvement
over the real live baseline, not the retired alert gate):

- `build_recency_weighted_live_snapshots.py` (new): precomputes final EWMA
  state as of the most recent match into `data/live/recency_form_batter_stats.json`
  / `recency_form_bowler_stats.json` (gitignored, regenerate via this
  script -- same convention as every other `data/live/*.json` snapshot).
  Run once already; 7,061 batter profiles, 5,205 bowler profiles.
- `app/ml/wicket_contract22_features.py`: `WicketContract22FeatureComputer`
  now loads these snapshots and folds `striker_recency_*`/
  `partner_recency_runs_per_ball`/`bowler_recency_*` into `compute()`,
  same known/unknown-bowler degradation as existing `bowl_hist_*`
  features (unresolved player -> zeros, no crash).
- `app/ml/prediction_engine.py`: `WICKET_CONTRACT22_ARTIFACTS` repointed
  to `models/candidates/contract22_wicket_v9_recency_form`.
- Verified end-to-end, not just assumed: `PredictionEngine()` instantiates
  cleanly, `predict_wicket_probability` returns a sane probability
  (22.1% for a real Kohli-vs-Bumrah mid-innings scenario), and the
  recency lookups resolve to real non-zero values (Kohli's recency strike
  rate 1.56 runs/ball, Bumrah's recency economy 7.29) rather than
  silently falling back to defaults. 208 tests passing.

Run-range remains on `run_range_v7_competition_prior`, unchanged -- the
recency candidate for that side is not promoted (`production_changed:
false` in its report).

## Next: partnership scoring rate

Queued as the natural follow-up (matches the user's "partnership builds"
framing directly): the codebase already tracks
`partnership_legal_ball_age` (how long a pair has batted together) but
never the actual *rate* that pair is scoring at -- a more direct
"partnership quality" signal than age alone. Not yet built or tested.
