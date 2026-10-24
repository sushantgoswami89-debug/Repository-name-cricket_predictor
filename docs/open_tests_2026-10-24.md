# Open tests and what to add / delete (2026-10-24)

Rule from the user (2026-10-24): **no test is closed.** Every row below stays open; "result so far" is what the data says today, not a verdict.
All numbers: men's T20, trained on matches up to 2023, holdout 2025+ (60,988 overs, 1,704 matches), match-clustered bootstrap.

## Next-over forecast (runs, big over, quiet over, wicket)

| # | Test | Script | Result so far | Still open |
|---|---|---|---|---|
| 1 | Batter strike rate in THIS innings (shrunk to a typical rate) | experiment_batter_form.py | corr 0.418 -> 0.423, nothing | other shrink targets, per-phase, vs left/right bowler |
| 2 | Mean (L2) instead of median (L1) objective | experiment_batter_form.py, train_over_outlook.py | bias -0.6 -> -0.02 runs per over | live scoring of the mean forecast over several matches |
| 3 | Pitch / venue history (runs per over, first-innings average) | experiment_pitch_and_career_form.py | RMSE -0.02, big AUC +0.007, wicket AUC -0.005 | pitch from the live match itself (first-innings total as a pitch signal in innings 2), toss/dew, weather, venue type (day/night), more careful venue names |
| 4 | Career strike rates (overall and per phase) | experiment_pitch_and_career_form.py | RMSE -0.07, big AUC +0.021 (clear of zero) | bowler side (economy / wicket rate), batter vs bowler type, recency weighting, the live name mapping |
| 5 | Current strike rate as a hand-over (switch after N balls, blends K=10/20/40) | experiment_current_sr_handover.py | RMSE -0.005, AUC +0.001 | other cut-offs, per-phase hand-over |
| 6 | Current strike rate as a correction layer, weighted by balls | experiment_current_sr_layer.py | RMSE -0.003, AUC +0.0003 | layer on the wicket chance with wicket-specific features |
| 7 | Same layer, weighted by runs scored / excess runs | experiment_current_sr_runs_weight.py | same as #6 | per-ball weighting, boundaries-only weighting |
| 8 | Which inputs carry weight (drop one group at a time) | experiment_feature_ablation.py | see the ablation section | drop several redundant groups TOGETHER (phase, score, position, run rate overlap each other); rerun on the final feature set |
| 12 | Batter phase profile: SR in each phase + share of balls per phase (role), then current-phase SR, then in-innings SR, then the gap | experiment_batter_phase_profile.py | profile/role: RMSE -0.0007 (nothing); in-innings rate + gap: RMSE -0.005, wicket AUC +0.002 (borderline) | profile per bowler type, role from batting position instead of balls share, openers/finishers split |
| 13 | Required rate compared with current rate directly (gap, ratio, runs per wicket in hand, required vs last-2-overs rate) | experiment_required_rate_gap.py | RMSE -0.002, AUC +-0.0005 (nothing); the model already learns it from the two rates | win-probability model (not tested there), required rate against the batters' phase rates |

## Win probability

| # | Test | Result so far | Still open |
|---|---|---|---|
| 9 | Served GBM vs GBM+NN ensemble vs NN, live chases (20 matches) | Brier 0.083 / 0.086 / 0.107 (one match, 1442989, excluded: confident and wrong for 19 overs) | reliability by overs left and wickets down on thousands of historical chases; find out what happened in 1442989; GBM vs NN reaction speed in chases |
| 10 | Ours vs Cricbuzz's curve | NZ v IND: chase ours 0.264 vs 0.277, first innings ours 0.529 vs 0.371 (a 7% upset); 3 older matches 0.107 vs 0.085, mostly the women's match | more finished matches; a team-strength input for the in-play model (Cricbuzz appears to use one) |
| 11 | Women's matches | model is men's-only; Cricbuzz 5% where we said 30% | a women's model or a team-strength rating |

## Things to ADD (candidates, ordered)

1. Career strike rates in a shadow test (the only clear gain so far).
2. Bowler career economy / wicket rate (the mirror of #4; the live pipeline already knows the bowler).
3. Team-strength input for win probability (what Cricbuzz seems to do).
4. A live scoring record (we said / what happened) for every message, so these tests can be checked on real overs.

## Ablation: remove one group from the full set (holdout RMSE 4.110, big AUC 0.7352)

| Removed | RMSE change | Big AUC change | Verdict |
|---|---|---|---|
| career strike rates | +0.054 | -0.014 | the strongest group |
| chase pressure (is_chase, runs_required, required RR) | +0.034 | -0.010 | very strong, also the wicket chance (-0.009) |
| recent form (last 2 overs) | +0.012 | -0.002 | carries the current form |
| wickets (down, in hand) | +0.012 | -0.002 | keeps |
| current run rate | +0.002 | -0.000 | marginal, overlaps score + balls |
| score so far | +0.001 | -0.000 | no change on its own (overlaps run rate + balls) |
| position (over, balls bowled, balls remaining) | +0.001 | -0.000 | no change on its own |
| phase label | -0.000 | -0.000 | no change |
| pitch / venue history | +0.002 | +0.000 | no change on runs or big/quiet; the wicket AUC is 0.0025 BETTER without it (interval clear of zero), so it slightly hurts the wicket chance |

Caution: score, run rate, position and phase say overlapping things, so removing ONE of them costs nothing while removing all of them
would. They are delete candidates only as a set, to be tested together (open).

## Things to DELETE (candidates, none decided)

1. Phase label, score-so-far, position (kept only if the joint removal test shows no loss): fewer inputs to map from Cricbuzz live.
2. Pitch/venue history: no gain once career rates are in, extra venue-name mapping to maintain.
3. Old 'Most likely range' in the console text, after the mean forecast has been compared with it on live overs.
4. The wicket line on Telegram: AUC 0.60, the weakest forecast; decide after 3-5 matches of live scoring.
