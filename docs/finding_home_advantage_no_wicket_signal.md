# Home-ground advantage: real for match outcomes, no signal for over-level wicket risk

Date: 2026-08-23. User request: research what cricket experts/analysts
identify as real drivers of IPL wins, and test whether those convert into
usable data. Web research confirmed home-ground advantage is real and
substantial: ESPNcricinfo's "Numbers Game: Home advantage in the IPL" and
corroborating team-by-team figures put the overall home win rate at 55%
across IPL history, with a wide team-specific spread -- Rajasthan Royals
68% at home vs Lucknow Super Giants 40%.

## What was already there, unused

`app/ml/ipl_venues.py`'s `team_venue_context()` (home/away/unknown, by
franchise) has been computed since `run_range_enriched_v3_batting_style`
and is already a live feature in `run_range_v11_partnership_rate`'s
`VENUE_FEATURES`. But the wicket line's own `VENUE_FEATURES`
(`train_contract22_rigorous.py`) never included it -- the column already
existed in the exact dataframe (`build_ipl_venue_regime_dataset`'s
output) every wicket training script already merges, just never
selected.

## Tested: adding it to the wicket model

`train_contract22_wicket_v13_home_away.py`, on top of currently-live
`contract22_wicket_v10_partnership_rate`:

| | AUC known | AUC unknown |
|---|---:|---:|
| Live `v10`, blended | 0.6118 | 0.6102 |
| `v13`, blended | 0.6118 | 0.6094 |
| Live `v10`, IPL | 0.6104 | 0.6059 |
| `v13`, IPL | 0.6095 | 0.6036 |
| Live `v10`, T20I | 0.6113 | 0.6110 |
| `v13`, T20I | 0.6112 | 0.6100 |

Flat-to-worse on every cut; IPL-unknown is notably worse (-0.0023).
`batting_team_venue_context` ranked 36th of 53 features by importance --
low, not meaningfully used by the model.

## Why the real, well-documented effect doesn't show up here

Home advantage is a real, large effect on whole-*match* outcomes (crowd
support, travel/rest, pitch familiarity compounding over 20 overs and two
innings). But this model predicts wicket risk for one specific over, at a
point where the match state is already fully known -- current score,
wickets down, phase, the specific batter and bowler, recent-ball
momentum. Whatever advantage a home team carries has likely already
manifested in *how the match got to this state* (e.g. a stronger position
by over 10) rather than adding independent information about whether
*this specific over* produces a wicket, on top of everything the model
already sees. Same shape of explanation as toss's rejection
(`finding_toss_no_signal.md`): pre-match contextual factors add little
once rich in-match state is already in the feature set.

## Recommendation

**Not promoted.** `PredictionEngine` remains on
`contract22_wicket_v10_partnership_rate`. This doesn't call into question
`run_range_v11_partnership_rate`'s existing use of the same feature
(that inclusion predates this session and wasn't re-tested here) -- only
establishes that adding it *fresh* to the wicket line specifically
doesn't help. Candidate artifact:
`models/candidates/contract22_wicket_v13_home_away/validation_report.json`.
