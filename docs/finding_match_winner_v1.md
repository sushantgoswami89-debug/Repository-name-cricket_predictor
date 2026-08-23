# A live win-probability model, added as a third prediction

Date: 2026-08-23. User proposal: "I think that model can be used for win
probability for each team rather than run & wicket prediction." Clarified
scope with the user first (add as a third prediction vs. replace vs.
validate-only) -- **confirmed: add alongside run-range and wicket-in-over,
not a replacement.**

## Why this, and why now

Several signals rejected earlier this session for over-level prediction
(toss, home-ground advantage, team-vs-team head-to-head, recency-weighted
venue par score) were rejected with the same explanation each time: they
plausibly affect the *whole match outcome*, they just don't concentrate
into a single over's runs or wicket risk once rich in-match state is
already known. A win-probability model is exactly the target where that
abandoned signal should matter -- this is the first real test of that
standing hypothesis from four separate finding docs.

## Label

`app/ml/match_winner_dataset.py`: does the team currently batting in this
innings go on to win the match? Known only at full-time, joined back onto
every over the same way `runs_in_over`/`wicket_in_over` already are --
not a new kind of leakage, the same relationship every existing target in
this codebase already has to its own label. 186 of 6,767 matches (2.7%,
verified directly) have no resolvable winner ("no result"/"tie") and are
excluded. Spot-checked against a real match (SRH beat Punjab Kings by 2
runs, batted first): every innings-1 row labeled 1, every innings-2 row
labeled 0, consistently -- not degenerate.

## Model

Binary LightGBM classifier, Platt-calibrated, same chronological split as
everything else this session (train<=2023 / calibration=2024 /
holdout>=2025). Reuses `contract22_wicket_v16_team_composition`'s entire
feature lineage (recency-weighted batter/bowler form, phase-specific
batter recency, partnership rate, team role composition, state/venue/
matchup features) plus the four contextual signals named above, added
here for the first time as a genuine test rather than assumed.

## Results (2025+ holdout, never seen in training)

| | AUC (known bowler) | AUC (unknown bowler) |
|---|---:|---:|
| Blended | 0.8587 | 0.8544 |
| IPL | 0.7889 | 0.7796 |
| T20I | 0.8703 | 0.8663 |
| 3-feature baseline (current rate, required rate, wickets, balls left) | 0.7486 | -- |

Beats the honesty-floor baseline by 11 AUC points -- the richer feature
set adds real value, not just noise from a target that's mostly decided
by the obvious match-state numbers already in the baseline.

**Phase progression -- the sanity check that matters most for a
win-probability model**: powerplay 0.803 -> middle 0.874 -> death 0.899.
Honestly uncertain early (T20 is genuinely volatile in the first six
overs) and sharpens as the match becomes more decided. This is the shape
a real win-probability model should have; a model that was already
confident in the powerplay would be a red flag, not a good sign.

**The previously-rejected contextual signals matter here, mostly as
predicted**:

| Feature | Rank of 68 | Was rejected for... |
|---|---:|---|
| `venue_recency_par_score` | 2nd | run-range AND wicket (both models, both cuts) |
| `h2h_batting_team_win_rate_shrunk` | 7th | run-range AND wicket (both models, every cut) |
| `batting_team_venue_context` (home/away) | 19th | wicket specifically |
| `h2h_matches_played` | 17th | (companion to the win-rate feature above) |
| `venue_recency_prior_innings` | 15th | (companion to the par-score feature above) |
| `toss_decision` / `batting_team_won_toss` | 47th / 36th | run-range AND wicket |

Three of four hypotheses confirmed strongly (venue recency and team H2H
now rank in the top 10 of 68 features, home/away meaningfully mid-tier).
Toss is the exception -- it ranks near the bottom here too, so its
earlier rejection wasn't a target-mismatch problem, toss genuinely
doesn't carry much signal at either level. IPL AUC is lower than T20I's
(0.789 vs 0.870 known) -- plausibly because IPL fixtures are more
competitive/closer on average, while T20I includes many mismatched
fixtures (major nations vs. associates) that are easier to call.

## Production wiring

- `app/ml/match_winner_features.py`: `MatchWinnerFeatureComputer`
  composes on top of `WicketContract22FeatureComputer` (delegates for
  the ~60 shared features) rather than duplicating that logic, and adds
  the four new contextual lookups.
- `build_match_winner_live_snapshots.py` (new): `data/live/
  team_h2h_pairs.json` and `data/live/venue_recency_par.json`. Team
  composition, recency form, phase recency, and partnership rate reuse
  snapshots already built earlier this session; toss is genuine live
  input, not history. Run once already: 113 team H2H pairings, 426 venue
  recency profiles.
- `app/ml/team_h2h_dataset.py` / `app/ml/venue_recency_dataset.py`: each
  gained a `compute_final_*_state` function (mirrors the pattern already
  used for player recency) so the live snapshot builder and the training
  dataset share the exact same accumulation logic.
- `runtime_match_winner.py`: `MatchWinnerRuntime`, mirrors
  `runtime_wicket_contract22.py`'s pattern exactly (integrity manifest,
  feature-contract validation, Platt calibration).
- `app/ml/prediction_engine.py`: new `MATCH_WINNER_ARTIFACTS` constant,
  `self._match_winner_runtime`, and a new prediction step in `predict()`.
  Resolves `bowling_team` (name) alongside the `batting_team` resolution
  that already existed for team composition, so both the wicket and
  win-probability calls share one answer.
- `app/ml/prediction_result.py`: new `win_probability: float = 0.5`
  field, validated in range, included in `to_dict()`.
- `app/live/pipeline.py`: **fixed another real drop-gap**, same pattern
  as `team1_players`/`team2_players` found and fixed earlier this
  session -- `ToiSnapshot.toss_won_by`/`toss_decision` have always been
  fetched but were never threaded into `MatchContext`. New
  `_toss_metadata()` helper populates `context.metadata["toss_decision"]`/
  `["batting_team_won_toss"]` at both construction sites, degrading to
  "unknown"/`False` when TOI's toss block isn't available yet (matches
  `ToiSnapshot`'s own best-effort convention). Telegram text now shows
  a `Win Probability: <team> NN.N%` line.

## Verification (not just trusted)

- 208 tests pass throughout.
- Direct sanity checks with hand-built scenarios: a chasing team needing
  10 off 30 balls with 8 wickets in hand scored **0.99**; a team needing
  60 off 12 balls with 2 wickets in hand scored **0.015**; a genuinely
  close early-innings scenario (on par, over 3, all wickets in hand)
  scored **0.576** -- appropriately close to even, not a false extreme.
- A real IPL match (`1426261.json`) replayed through the actual
  `VerifiedLivePredictionPipeline`: 0 errors across both innings, the
  win-probability call executes successfully on every over alongside
  run-range and wicket.
- Confirmed the Telegram text renders the new line correctly, including
  the graceful "Batting side" fallback when team names aren't set.

## What this doesn't cover yet

Run-range and wicket-in-over are unchanged -- this is purely additive.
No retroactive re-litigation of the rejected toss/home-away/H2H/venue-
recency findings for those two models; those rejections stand on their
own merits for their own targets. `match_winner_v1` is a first cut, not
separately hyperparameter-tuned against this specific target -- reused
the wicket line's LightGBM settings directly. A natural next step,
untested here: does the same phase-by-phase honesty extend to the very
final overs of a chase (over 19-20), where win probability should
approach 0 or 1 almost deterministically -- not checked in this pass.
