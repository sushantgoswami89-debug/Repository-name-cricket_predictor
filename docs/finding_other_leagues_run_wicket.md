# Other T20 leagues for the run and wicket models: venue and win history no gain, extra training rows a tiny gain

Date: 2026-10-11. User: "for run and wicket you can use venue and winning also from other leagues".

## 1. Venue and winning history from other leagues (point-in-time, 2,264 T20I first innings since 2022)
| history used for the ground | corr(par, next total) | MAE | no prior at the ground | bat-first win log-loss (flat 0.6931) |
|---|---|---|---|---|
| T20I + IPL | 0.247 | 35.6 | 3% | 0.7023 |
| every men's T20 competition | 0.239 | 35.6 | 2% | 0.7015 |
| T20I only | 0.251 | 35.6 | 3% | 0.7023 |
The venue scoring level is weak for T20I totals whatever history feeds it, and a ground's bat-first win rate has no signal (worse than flat). Other leagues fill a few more grounds but add no accuracy. (Consistent with finding_team_phase_venue_no_signal and the pre-match leagues test.)

## 2. More training rows for the run-range / wicket GBM (Candidate v3 recipe; train <= 2023, calibrate 2024, holdout 2025+)
Added 142,326 overs from BBL, BPL, CPL, ILT20, LPL, MLC, MSL, T20 Blast, PSL, SA20, Syed Mushtaq Ali, Super Smash to 148,607 T20I+IPL training overs.
| holdout | variant | runs MAE | runs RMSE | wicket Brier | wicket AUC |
|---|---|---|---|---|---|
| T20I 2025+ (55,408 overs) | baseline | 3.1503 | 4.1176 | 0.20502 | 0.6010 |
| | + leagues, equal weight | 3.1609 | 4.1097 | 0.20481 | 0.6020 |
| | + leagues, weight 0.4 | 3.1536 | 4.1109 | 0.20480 | 0.6028 |
| T20I top-10 sides (7,248) | baseline | 3.5547 | 4.6077 | 0.20569 | 0.5957 |
| | + leagues, equal weight | 3.5319 | 4.5691 | 0.20520 | 0.5986 |
| | + leagues, weight 0.4 | 3.5397 | 4.5833 | 0.20530 | 0.5994 |
| IPL 2025+ (5,580) | baseline | 3.9727 | 5.1949 | 0.19555 | 0.6013 |
| | + leagues, equal weight | 3.9480 | 5.1535 | 0.19493 | 0.6038 |
| | + leagues, weight 0.4 | 3.9544 | 5.1689 | 0.19511 | 0.6037 |

Reading: a small, mostly consistent improvement (RMSE -0.8% and wicket Brier -0.0005 on top-10 sides; RMSE -0.8% and Brier -0.0006 on the IPL; all-T20I MAE slightly worse, +0.01). About 1% at best, with no significance test (per-row predictions were not saved) and on the simple state-only v3 GBM, not the deployed player-feature ensembles. NOT promoted. Next step if wanted: save predictions, match-level bootstrap, and repeat on the real run-range / wicket ensembles (needs their training pipelines).

Reproduce: `python backend/experiment_extra_leagues_run_wicket.py` (builds data/candidates/v3/extra_leagues_overs.csv if missing).
