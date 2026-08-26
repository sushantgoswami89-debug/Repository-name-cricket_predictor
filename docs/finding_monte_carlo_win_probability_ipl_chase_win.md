# WinViz-style Monte Carlo win-probability: real win for IPL chases, real loss for T20I

Date: 2026-08-24. Direct follow-up to the closed LSTM investigation
(`docs/finding_wicket_lstm_sequence_model_not_promoted.md`) and the
2026-08-23 strategic roadmap's #2 option: a genuinely different
mechanism from another trained classifier, motivated by CricViz's real
WinViz system (cricviz.com/winviz) and by the earlier chase-bug
investigation's "confirming, not leading" finding for `match_winner_v1`.

## Construction

Built `app/simulation/monte_carlo_win_probability.py`: given a real
mid-chase state (score, wickets in hand, balls bowled/remaining, target),
simulates the rest of the innings forward thousands of times using a
per-ball outcome distribution (wide/no-ball/wicket/0-6 runs) conditioned
on (phase, wickets_in_hand, pressure/dot-streak bucket), estimated from
real train<=2023 deliveries. Win probability = fraction of simulated
futures where the batting team reaches the target.

**Deliberate scope cut, disclosed up front**: real WinViz simulates at
the individual player level. This v1 is **team-agnostic** -- no player
identity, no batting order, just the same match-state granularity
`contract22_wicket_v16`'s `BASE_FEATURES` already uses. Tests the core
"simulate forward vs. classify directly" mechanism first.

**Real performance bug found and fixed before evaluating anything**: a
naive per-simulation Python loop measured ~0.7s/row (2000 sims) --
projected ~3.5 hours over the real ~18k-row chase holdout. Rewrote the
simulation to be fully vectorized across all `n_sims` trajectories with
numpy (dense array lookup + cumulative-sum categorical sampling),
achieving a 52x speedup (~0.013s/row) -- fixed the actual bottleneck
rather than silently subsampling, per standing project practice.

**Evaluation methodology**: reused `train_match_winner_v1.build_dataset()`
unchanged so the live GBM's predictions are on the exact same rows, not
re-derived. Scoped to chases only (`is_chase==1`) -- a one-stage
simulate-to-target problem, unlike innings-1 which would need a two-stage
simulation (rest of innings 1, then a hypothetical innings 2) -- out of
scope for v1, same "start with the cleanest testable case" practice used
for toss/LSTM in this project.

## A real self-correction during this investigation

First pass evaluated the raw Monte Carlo win-fraction directly, without
Platt-calibrating it -- an apples-to-oranges comparison, since the GBM's
reported Brier/accuracy are always calibration-fitted (every other script
in this project does this). Caught before drawing any conclusion: IPL
showed the MC's **AUC beating the GBM's** (0.916 vs 0.897) while its
Brier/accuracy looked much worse (0.234 vs 0.132 Brier) -- the classic
signature of a real ranking signal buried under bad calibration, not a
genuinely worse model. Added proper Platt calibration (fit on the 2024
calibration chase subset, same convention as every other model here).

A single global calibrator improved things but didn't close the IPL gap
(Brier 0.185, accuracy 72.0% -- still well below what the AUC advantage
implied was possible). Tested **per-competition Platt calibration**
(separate IPL-only and T20I-only calibrators) -- same "don't trust a
blended correction, split by population" practice this project has
applied repeatedly (IPL band-width fix, T20I associate/full-member
split). This closed the gap completely.

## Result (chase-only, real 2025+ holdout, identical rows both models)

| Metric | MC, per-competition calibrated | Live `match_winner_v1` GBM |
|---|---:|---:|
| IPL AUC | **0.9163** | 0.8972 |
| IPL Brier | **0.1188** | 0.1323 |
| IPL accuracy | **83.9%** | 80.6% |
| T20I AUC | 0.9208 | **0.9418** |
| T20I Brier | 0.1171 | **0.0980** |
| T20I accuracy | 83.2% | **85.8%** |
| Blended AUC | 0.9209 | **0.9357** |
| Blended Brier | 0.1174 | **0.1034** |
| Blended accuracy | 83.3% | **85.0%** |

**A team-agnostic Monte Carlo simulation with zero player identity
genuinely beats the full 60+ feature trained GBM classifier on IPL chases
specifically, on real holdout, on identical rows, on all three metrics.**
On T20I it clearly loses on all three. Blended, the GBM still wins
(T20I's 5.6x larger row count in this holdout dominates the average) --
but the IPL result is real, not noise: 2,634 holdout rows, consistent
across AUC (rank-invariant, unaffected by calibration), Brier, and
accuracy simultaneously.

**Plausible mechanism**: IPL's ~13 fixed franchise venues and stable
squad structures make aggregate phase/wickets-in-hand/pressure dynamics
more stable and less player-composition-dependent match to match than
T20I's much more heterogeneous population (83% associate-nation
fixtures, wildly varying team strength -- see
`finding_t20i_associate_vs_full_member_gap_all_models.md`). A
team-agnostic simulation is a better-fitting mechanism where team
identity matters less; T20I's real team-strength gaps are exactly what
the GBM's player/team features capture and the team-agnostic simulation
cannot.

## Initially promoted as primary, then overridden by the user -- see below for the final design

Recommendation confirmed by the user ("ok go ahead") and briefly wired
into `MatchWinnerEngine.predict()` as the PRIMARY served prediction for
IPL chases. **The user then explicitly overrode this**: "dont route now,
it should check when match is live and after 5 or 10 matches it should
show me summary and then remind me to pick one." The routing-to-primary
design below was reverted the same session -- see "Final design" further
down for what's actually live. Kept here for the historical record of
what was built and why it changed.

Originally: chases where `EngineRouter.resolve(context.format,
context.competition).family == EngineFamily.IPL` (the same IPL-detection
mechanism already trusted elsewhere in the live pipeline, e.g.
`_innings_limit`) routed to the Monte Carlo simulation as the served
result; every other case (T20I chases, all innings-1 predictions, any
resolution failure) kept using the GBM. The Monte Carlo call was wrapped
in its own try/except that fell back to the GBM path on any failure.

**Production artifacts** (`build_monte_carlo_win_probability_artifacts.py`,
reused the cached calibration-set probabilities rather than re-simulating):
`models/candidates/win_probability_monte_carlo_v1/outcome_table.npy`
(dense simulation lookup table) + `ipl_platt_calibrator.pkl` (fit on 1,325
real 2024 calibration IPL-chase rows), with an `ARTIFACT_MANIFEST.json`
sha256 integrity check, same convention as every other live runtime.
New `runtime_monte_carlo_win_probability.py` mirrors
`runtime_match_winner.py`'s loading/verification pattern.

**Verified end-to-end, not just unit-tested**: 5 new tests in
`tests/test_match_winner_engine.py` cover the full routing decision table
(IPL chase -> Monte Carlo; IPL non-chase -> GBM; T20I chase -> GBM;
Monte Carlo failure -> GBM fallback; missing artifacts at construction ->
GBM fallback) -- 216 tests pass total. Hand-verified 3 realistic IPL chase
scenarios directly against the real (non-mocked) engine: a comfortable
chase (need 40 off 60, 7 wickets) scored 88.8%, two genuinely-lost causes
(need 60 off 24 with 3 wickets; need 30 off the last over with 1 wicket)
both scored 0.6% -- sane, non-degenerate outputs. A real full IPL match
Cricsheet replay through `VerifiedLivePredictionPipeline` completed both
innings with 0 errors (this particular replay tool doesn't set
`context.competition`/`format="IPL"` on its synthetic snapshot, so it
didn't exercise the new branch specifically -- confirmed via direct
engine calls instead, above).

No changes to `match_winner_v1`'s own artifacts or the GBM path itself --
only routing changed (and was then reverted -- see below). Not yet
committed to git.

## Final design: GBM stays primary, Monte Carlo runs as a logged live shadow (user-requested, 2026-08-24)

Two user corrections landed in sequence the same session, both acted on
directly:

1. *"route in backend IPL also get prediction from GBM also in
   background so we can have comparison live"* -- first interpreted as
   "keep Monte Carlo primary, add GBM as a shadow." Built and verified
   (GBM shadow alongside a Monte-Carlo-primary result).
2. *"dont route now, it should check when match is live and after 5 or
   10 matches it should show me summary and then remind me to pick
   one"* -- corrected the interpretation. **The GBM is the only served
   prediction, completely unchanged from before this whole
   investigation.** Monte Carlo never serves a production result yet --
   it only runs as a logged shadow, IPL chases only, so a real live
   comparison can accumulate before any routing decision is made.

**What's actually live** (`app/ml/match_winner_engine.py`): every
prediction is served by the GBM, always. For IPL chases specifically,
Monte Carlo also runs and its result is attached as
`metadata["monte_carlo_shadow_win_probability"]` -- never served, purely
observational. Fault-isolated (a shadow failure only omits the shadow
value, never touches the served GBM result).

**Persistent live comparison log** (new module
`app/ml/win_probability_shadow_log.py`, `WinProbabilityShadowLog`):
`app/live/pipeline.py` appends one row per real IPL-chase over prediction
(`data/live/win_probability_shadow_log.jsonl`, gitignored like every
other `data/live/*` snapshot) with both models' numbers. When a chase
innings completes, the pipeline determines the real winner from the same
score-vs-target comparison `_is_complete` already uses (same convention
`app/ml/match_winner_dataset.py`'s historical label uses) and records it
to `data/live/win_probability_shadow_outcomes.jsonl`, keyed by
`match_id` and idempotent (a duplicate completion signal can't
double-count a match).

**Automatic threshold reminder**: once 5 real matches have a known
outcome (`MIN_MATCHES_FOR_SUMMARY = 5` -- the user said "5 or 10," picked
the lower bound so the first check-in arrives sooner), the pipeline sends
a one-time summary through the Telegram publisher (real AUC/Brier/accuracy
for both models over the real matches logged so far) prompting a decision
on which model to keep. Fires exactly once (a marker file prevents
re-firing on every subsequent match past the threshold).

**Verified end-to-end with real code paths, not mocks**: a real 3-match
IPL Cricsheet replay through the actual `VerifiedLivePredictionPipeline`
(with `competition="IPL"` set on every snapshot, matching what a real TOI
feed would report, and `min_matches_for_summary=3` for the test) produced
52 real per-over shadow-comparison rows, 3 correctly-recorded outcomes,
and exactly one summary message published with real computed metrics
(illustrative only given the tiny/non-representative 3-match smoke-test
sample -- MC 94.2%/GBM 82.7% accuracy on that specific replay, not a
result to draw any conclusion from). 216 tests pass, including 5 rewritten
`MatchWinnerEngine` tests confirming: GBM always serves; Monte Carlo shadow
present only for IPL chases; shadow failures never affect the served
result; missing Monte Carlo artifacts still serve GBM cleanly.

Also set up a weekly cloud check-in routine (Mondays 9am IST) as a
secondary, lower-frequency nudge independent of the pipeline's own
threshold-triggered reminder -- see project memory
`project_cricketbaba_status.md`'s "Standing plan" section for the full
decision criteria and why the live-data-driven reminder is the primary
mechanism, the cloud routine just a backup prompt.

## Monte Carlo anticipates dramatic reversals meaningfully earlier than the GBM (2026-08-25)

Extends the 2026-08-23 finding ("how early can match_winner_v1 call a
reversal -- it's a confirming indicator, not a leading one," median lead
over the scoreboard 0 overs) to compare both models directly, using the
identical real dramatic-recovery IPL match set (peak required run rate
>=11 after over 10, chasing team still won -- 19 matches, exact count
reproduced from the original investigation).

**Result** (`backend/evaluate_reversal_timing_gbm_vs_monte_carlo.py`):
GBM reproduces the original finding exactly (mean/median lead over the
scoreboard: 0.0 overs -- confirms fast, doesn't lead). **Monte Carlo leads
the scoreboard by a mean of 2.5 overs (median 1.0)** -- and in a direct
16-match head-to-head, Monte Carlo called the reversal earlier than the
GBM in 10 matches, the same over in 5, and the GBM was earlier in only 1.

**Why**: the GBM classifies the current snapshot against historical
patterns; Monte Carlo simulates thousands of actual remaining
trajectories from the current state, so when a team is behind but still
has enough overs/wickets in hand for a statistically real path back, the
simulation detects that residual chance directly -- before the scoreboard
itself confirms the turn. A genuinely different strength from the raw
AUC/Brier comparison, and arguably more valuable for a live-facing
feature (catching the "is this actually turning around" moment early is
what a nail-biting chase is watched for).
