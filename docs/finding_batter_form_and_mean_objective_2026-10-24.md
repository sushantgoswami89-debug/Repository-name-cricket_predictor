# Batter form in the innings, and mean versus median (2026-10-24)

Question (user, 2026-10-11 and 2026-10-24): two batters on strike rate ~200 and the forecast is 5-7; "we were flat in prediction".
Script: `backend/experiment_batter_form.py` (train <= 2023, holdout 2025+, same recipe as the live Candidate v3 GBM, n = 246,846 overs).
Added per over, before the over: striker's and partner's runs and balls in this innings, their shrunk strike rates (prior 1.25 runs/ball, 12 balls), pair rate and balls.

| 2025+ holdout | variant | MAE | RMSE | bias | pred sd | actual sd | corr |
|---|---|---|---|---|---|---|---|
| all | baseline L1 (live) | 3.226 | 4.228 | -0.589 | 1.90 | 4.61 | 0.418 |
| all | baseline L2 (mean) | 3.260 | 4.186 | -0.022 | 1.91 | 4.61 | 0.418 |
| all | + batter form L1 | 3.219 | 4.218 | -0.580 | 1.91 | 4.61 | 0.422 |
| all | + batter form L2 | 3.251 | 4.176 | -0.019 | 1.92 | 4.61 | 0.423 |
| T20I top-10 | baseline L1 (live) | 3.555 | 4.608 | -1.073 | 1.68 | 4.69 | 0.300 |
| T20I top-10 | baseline L2 (mean) | 3.530 | 4.510 | -0.510 | 1.70 | 4.69 | 0.300 |
| T20I top-10 | + batter form L2 | 3.512 | 4.491 | -0.460 | 1.71 | 4.69 | 0.308 |

Findings
1. Batter form in the innings adds almost nothing (correlation 0.418 -> 0.423, MAE 3.226 -> 3.219). Not worth shipping. The earlier tests of batter position / pressure / opponent also found no signal.
2. The live GBM is trained with the L1 (median) objective, so it forecasts the median over. That is the "low" part: bias -0.6 runs per over (-1.1 on top-10 sides). The mean objective (L2) removes most of it (bias -0.02 / -0.51) and has the better RMSE. The 3-run band hit rate is slightly lower for L2 (0.285 v 0.300) because a band around the median hits more often.
3. "Flat" is mostly the nature of the target: the predictions vary with sd 1.9 while the overs vary with sd 4.6, correlation 0.42 -> about 17% of the variance is explainable at best. The model cannot see a six or a wicket coming; forecasting more spread than it can justify would make it worse.
Recommendation: point forecast = the mean (L2); show the likelihood of a big or small over (for example P(10+ runs), P(wicket)) next to it instead of a narrow 3-run window that is right about 25-30% of the time; do not add batter form.
