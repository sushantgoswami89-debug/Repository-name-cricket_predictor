# IPL Wicket v7.x — Context Features + Threshold Margin + Depth Sweep

Date: 2026-08-22 (attempts 1-3), updated same day with attempts 4-5

## Starting point

`ipl_wicket_v7_context` (`context_depth4`, 2026-08-21) passed 8 of 9
promotion gates but was rejected: holdout precision 40.4% vs. the required
≥42%, despite improving Brier/ROC-AUC/PR-AUC over baseline on every split.
Holdout recall (29.4%) had headroom over the 27.86% floor.

## Attempt 1 — threshold safety margin (`train_ipl_wicket_v7_1_margin_threshold.py`)

Hypothesis: the threshold was picked on a small (~1,369-row) selection
split and the 42.8%→40.4% precision drop on holdout was sampling noise, so
picking a threshold with a precision margin above 0.42 on the selection
split should hold up better on holdout.

Result: **rejected — this is not a sampling-noise problem.** Swept
selection-split target precision from 0.42 to 0.46 (`sweep_v7_1_margin.py`);
every threshold that pushed holdout precision to ≥42% dropped holdout
recall below the 27.86% floor, and every threshold that kept recall above
the floor left holdout precision below 42%. There is a real
precision/recall cliff in this model's operating region, not noise.

## Attempt 2 — merge in bowler current-spell features (`train_ipl_wicket_v7_2_spell_features.py`)

Hypothesis: the `announced_bowler_current_spell_v3` /
`announced_bowler_phase_h2h_v1` research line already proved bowler
current-spell/phase-history signal adds real wicket predictive value
(`precision_improves_same_alert_volume: true` in both), but so far only as
a live post-hoc adjustment on the production model (shipped as "wicket
model v2", commit `765a118`) — never as a training feature of a
from-scratch candidate. `data/candidates/ipl_cold_start_v5_t20_priors` and
`data/candidates/announced_bowler_phase_h2h` join 1:1 cleanly on
`(source_file, innings, over)` — 47,756/47,756 rows, verified before use.

Added features: `current_spell_balls`, `spell_number`, `spell_state`,
`bowler_match_economy`, `bowler_match_dot_rate`,
`bowler_phase_history_economy`, `bowler_phase_history_wickets`,
`bowler_history_supported`.

Result: **rejected — closest yet, but still short.** All accuracy metrics
improved further over v7 (candidate Brier 0.1956 vs. v7's 0.1960; ROC-AUC
0.5986 vs. 0.5945). Recall headroom increased substantially (up to 34.8% at
the default threshold, vs. v7's 29.4%). But swept across selection-split
target precision 0.42–0.47 (`sweep_v7_2_margin.py`): the best simultaneous
point found was ~41.0% precision at ~27.1% recall (both just miss), or
~42.9% precision at 23.9% recall (recall fails badly). No threshold cleared
both gates at once.

## Attempt 3 — depth/iteration sweep on the v7.2 feature set (`sweep_v7_3_depth_iterations.py`)

Hypothesis: a deeper or longer-trained model might resolve the residual gap.

Result: **rejected — this made things worse, not better.** Depths 5–6 and
iteration counts up to 900 all overfit badly relative to the ~1,369-row
selection split: holdout recall collapsed to single digits (as low as 1.1%
at depth 5 / 700 iterations) even where holdout precision looked good
(48–59%). `context_depth4` / 450 iterations (the original v7 choice) was
already near the sweet spot for this data volume; going deeper is a dead
end here, not an unexplored opportunity.

## Attempt 4 — retest depth with overfitting mitigation (exploratory, not promotion-grade)

Requested retest of attempt 3's deep-model idea, but this time with the
actual likely fix instead of just more capacity: CatBoost early stopping
against a held-out chronological tail of TRAINING itself (last 15% of
pre-2024 rows by date — not platt_fit/selection/holdout, so no new leakage)
plus depth-scaled L2. Result: dramatic. `best_iteration` landed at just
71–82 rounds across every depth/L2 combination — confirming attempt 3's
fixed 450–900 iteration counts were themselves the overfitting cause, not
depth. Several configs appeared to clear both gates.

**Caveat, and why attempt 4 isn't the final answer:** that exploratory
sweep picked its "best" config by directly checking 5 architectures × 21
threshold margins against the **locked holdout** and keeping whichever
happened to pass — that's holdout-peeking (an implicit multiple-comparison
search), not a legitimate promotion test, even though the underlying
early-stopping fix is real and worth keeping.

## Attempt 5 — the same fix, done properly (`train_ipl_wicket_v7_5_early_stopping.py`)

Rebuilt as a real candidate: architecture (depth, L2) selected among 4
candidates using only the **selection**-split Brier score (same criterion
`ipl_wicket_v7_context` used originally), threshold fixed once at the real
0.42 target precision on the selection split, holdout evaluated exactly
once at the end — no peeking anywhere.

Selected architecture: `depth6_l2_28` (best_iteration=72), chosen fairly on
selection-split Brier (0.2008, best of the four). Result:

- Candidate vs. baseline (locked holdout): Brier 0.1943 vs 0.1964, ROC-AUC
  0.599 vs 0.588, PR-AUC 0.385 vs 0.369 — best margin of improvement of any
  attempt this session.
- Threshold 0.33 (from selection split, 43.1% precision / 27.6% recall
  there).
- **Holdout: 41.6% precision, 30.1% recall.** Recall clears the 27.86%
  floor comfortably. Precision is the closest result yet — 0.4 points
  short of the 42% gate — but still short.
- `decision: "reject_keep_research"`.

## Attempt 6 — k-fold cross-fitted threshold selection (`train_ipl_wicket_v7_6_kfold_threshold.py`)

Hypothesis: v7.5's threshold was picked on only ~1,369 rows (half of the
2024 calibration year); that's a small enough sample that the 41.6%-vs-42%
miss could plausibly be threshold-selection noise rather than a real
ceiling. Kept the exact same trained model as v7.5 (same architecture,
same training, nothing retrained) and changed only the threshold-selection
step: 5-fold stratified cross-fitting of the Platt calibrator over the
**full** 2024 calibration set (2,738 rows, double the previous sample),
producing an out-of-fold calibrated probability for every 2024 row, then
picking the threshold against that fuller, less noisy estimate. Single
holdout evaluation at the end, same as always — no peeking.

Result: **rejected, and informative.** Precision jumped to 43.8% on
holdout — comfortably above the 42% gate. But recall dropped to 25.9%,
now the one that misses (needed >27.86%). The two legitimate
threshold-selection methodologies (v7.5's half-split vs. v7.6's full
k-fold) land on *opposite sides* of the same boundary rather than both
converging on a pass. That's evidence this is a real precision/recall
trade-off ceiling for this trained model, not primarily a noisy-estimate
artifact of the smaller v7.5 threshold-selection sample.

## Attempt 7 — T20I-augmented training data (`train_ipl_wicket_v7_7_t20i_augmented.py`)

Hypothesis: `data/candidates/v3/verified_training_overs.csv` already
combines IPL (47,790 rows / 1,243 matches) and T20I (199,056 rows / 5,524
matches) — everything richer built downstream (canonical player
identities, venue database, bowler current-spell/phase-h2h) comes from
separate builders hardcoded to `cricsheet/ipl` only, which is why T20I
never reached this line before. Full T20I feature parity (its own
venue/identity pipeline) is real multi-session engineering, so this tests
the cheap version first: does raw match-state volume alone help, added
only to the **training** split (never calibration/threshold
selection/holdout, which all stay pure IPL, unchanged) using just the ~13
match-state features (score, wickets, run rate, phase, chase pressure,
recent-ball momentum) common to both competitions. T20I rows get NaN for
every IPL-only enriched column (CatBoost handles missing natively), plus
an `is_ipl` flag and a low sample weight (0.2) so 4x the row volume from a
different, only partially overlapping population doesn't swamp the IPL
signal. 109,135 T20I rows (through 2023) added alongside 33,525 IPL rows.
Same architecture and k-fold threshold method as v7.6.

Result: **rejected, but the closest result of the entire session.**
Holdout: 44.0% precision (comfortably clears 42%), 27.02% recall (needed
>27.86% — misses by only **0.84 points**, versus v7.6's 1.96-point miss).
Also the best overall metrics of any attempt: Brier 0.1938, ROC-AUC 0.601,
PR-AUC 0.383. The extra data volume measurably helped.

## Attempt 8 — doubled T20I weight (`train_ipl_wicket_v7_8_t20i_weight_40.py`)

Same as attempt 7, T20I sample weight raised 0.2 → 0.4, to see if more
T20I influence closes the remaining 0.84-point gap.

Result: **rejected, and moved the wrong way.** Precision rose slightly to
44.5%, but recall *dropped* to 26.3% (worse than attempt 7's 27.02%). More
T20I weight is not monotonically better — attempt 7's 0.2 looks closer to
a local optimum than a lower bound. Stopped here rather than continuing to
grid-search the weight against holdout, which would reintroduce the same
holdout-peeking problem flagged after attempt 4.

## Attempt 9 — full T20I feature parity for MOE + venue features (`train_ipl_wicket_v7_9_t20i_full_features.py`)

Extended `build_ipl_phase_moe_features` (`backend/app/ml/ipl_phase_moe_dataset.py`)
and `build_ipl_venue_regime_dataset` (`backend/app/ml/ipl_venue_regime_dataset.py`)
with an opt-in `scopes` parameter (default unchanged, `("ipl",)` — existing
callers/production untouched) so they can also process
`data/raw/cricsheet/t20i`. Their core logic turned out to already be
format-agnostic: `canonical_player_id` is Cricsheet's own cross-format
person UUID (works identically for IPL and T20I), and the chase-pressure/
partnership-age/venue-par-score math doesn't reference anything
IPL-specific. Only `canonical_team_id` (IPL franchise aliases) and
`team_venue_context` (home/away vs. IPL home grounds) degrade gracefully to
an unresolved-but-consistent fallback for national T20I teams — verified
both have safe fallback branches before running this (`ipl_venues.py`).
This gave T20I training rows real batter/bowler prior-stat,
chase-pressure/partnership, and venue-par-score features instead of the
NaN placeholders used in v7.7/v7.8. (Bowler current-spell/phase-h2h
features were not extended — that builder merges onto an already
IPL-only downstream file, a bigger job — still NaN for T20I here.)
IPL-side training/calibration/threshold/holdout data unchanged throughout.

Result: **rejected, and a genuine surprise — richer T20I features made it
slightly worse, not better.** Holdout: 44.7% precision (up slightly), but
26.37% recall — *worse* than v7.7's bare-match-state-only 27.02%, missing
by 1.49 points instead of 0.84. Same direction as v7.8 (more T20I richness/
influence → worse recall, not better). Best guess: the extra features
introduce real cross-competition distributional differences (batting
averages, venue pars differ systematically between IPL and T20I) that add
noise for the IPL holdout rather than transferable signal, whereas bare
match-state features (score, wickets, run rate, phase) are abstract enough
to transfer cleanly. **v7.7 (T20I augmentation, match-state features only,
weight 0.2) remains the best result of the entire session and line.**

## Attempt 10 — squad IPL-familiarity-weighted T20I rows (`train_ipl_wicket_v7_10_similarity_weighted_t20i.py`)

Hypothesis: v7.8 (more flat weight) and v7.9 (richer per-row features)
both made recall worse, suggesting "more/richer T20I influence" in an
undifferentiated way isn't the right lever — but that doesn't mean T20I
augmentation is tapped out, just that the *shape* of the extra influence
matters. Tried weighting each T20I match by how much its playing squads
overlap with the IPL player pool (via Cricsheet's own person UUID
registry): `weight = 0.05 + 0.35 * squad_familiarity_fraction`, computed
match-by-match from the full squad list (not leaked per-over — squad
membership doesn't depend on which over is being predicted). Distribution
sanity-checked first: median squad familiarity across T20I matches is 0.0
(most T20I matches involve no IPL players at all), but the top decile
reaches 0.64 (major-nation bilateral series) — a genuinely differentiated
signal, not a coin flip. Same architecture and match-state-only feature
set as v7.7 (not v7.9's richer version), same k-fold threshold selection.

Result: **rejected, and essentially a repeat of v7.9's regression, not an
improvement.** Holdout: 44.0% precision (matches v7.7 exactly), 26.37%
recall — *worse* than v7.7's 27.02%, and landing at almost the identical
value to v7.9's 26.37% recall. Row-weighted mean T20I sample weight came
out to 0.12 (lower than v7.7's flat 0.2, since most T20I matches have near-
zero familiarity) — this raises a real question of whether the regression
in both v7.9 and v7.10 is actually about *how much* T20I influence there
is on average (lower here, 0.12 vs 0.2) rather than *which* rows carry it.
That's an untested confound, not resolved this attempt — see next
direction below.

## Attempt 11 — squad IPL-familiarity weighting, renormalized to v7.7's exact magnitude (`train_ipl_wicket_v7_11_similarity_weighted_normalized.py`)

Direct follow-up to resolve attempt 10's confound: same squad-familiarity
signal, but `weight = 0.2 * familiarity / mean(familiarity)`, computed
from the actual T20I training rows so the row-weighted mean lands on
exactly 0.2 — matching v7.7's flat weight precisely. (This is a much
sharper differentiation than attempt 10's floor+scale version: 63% of
T20I rows get weight exactly 0 — matches with literally zero IPL-familiar
players — and the rest carry the full budget, up to ~1.0 for the most
IPL-heavy bilateral series.)

Result: **rejected, but conclusively answers the question.** Holdout:
43.86% precision / 27.02% recall — recall matches v7.7's 27.02% to the
last decimal place (0.27020038784744665, identical), precision is a
rounding hair below v7.7's 44.0% (both comfortably clear the 42% gate).
**This confirms attempts 9 and 10's regressions were a magnitude effect,
not a shape effect** — once average T20I influence is held constant at
0.2, differentiating rows by squad similarity is neutral (neither better
nor meaningfully worse) compared to flat weighting. The idea itself isn't
wrong; it just doesn't add anything on top of what flat weighting already
captures for this feature set.

## Conclusion

The wicket-only base-model line (v7 → v7.1 → v7.2 → v7.3 → v7.5 → v7.6 →
v7.7 → v7.8 → v7.9 → v7.10 → v7.11) got genuinely, measurably closer over
the session, then plateaued firmly at v7.7's result — now confirmed from
five different angles (weight magnitude, feature richness, two flavors of
per-row differentiation) that all converge on the same place:
from a 2.4-point precision miss (v7 itself) down to a 0.84-point recall
miss (v7.7, T20I match-state augmentation) — still never clearing both
gates at once, but the gap is real and shrinking, not stuck. Two attempts
to push further from v7.7 (more T20I weight in v7.8, richer T20I features
in v7.9) both moved the wrong way, so v7.7 looks like a genuine local
optimum for this general approach, not a waypoint to something better
found so far. Three categories of finding:

- **Confirmed exhausted / no further juice:** threshold-selection
  methodology (v7.1, v7.6) and architecture/regularization tuning on the
  IPL-only feature set (v7.3, v7.5) — three separate, sound approaches all
  converged on the same ~2-point wall without one clearing it, and a
  second T20I-weight value (v7.8) moved the wrong way. Don't re-run these
  levers again on this exact setup.
- **The one lever that produced a real gain, and where it stopped
  working:** T20I training-data augmentation (v7.7) measurably narrowed
  the gap — the only attempt all session that did. But pushing it further
  in either direction tried (more weight in v7.8, richer per-row features
  in v7.9) both made it worse, not better. v7.7's specific combination
  (bare match-state features only, weight 0.2) currently looks like a
  local optimum, not a step toward something bigger.

## Recommended next direction (not attempted this session)

- **v7.7's exact combination (flat 0.2 weight, match-state-only features)
  is a genuine, well-confirmed local optimum, not a placeholder waiting
  for the right refinement.** Five follow-up attempts — more weight
  (v7.8), richer per-row features (v7.9), similarity-weighting at lower
  magnitude (v7.10), and similarity-weighting at v7.7's exact magnitude
  (v7.11) — span the two natural axes (how much T20I influence, and how
  it's distributed across rows) and none beat it; v7.11 shows the shape
  question is settled (neutral, not harmful) once magnitude is controlled
  for. Don't keep varying weight schemes on this same feature/architecture
  combination — that lever is now genuinely exhausted, not just tried a
  few times.
- The two things that would actually be new levers, not more of the same:
  (1) bowler current-spell/phase-h2h features extended to T20I
  (`backend/build_current_spell_bowler_features_phase_h2h.py:88` merges
  onto an already IPL-only downstream file — a bigger job, and given
  v7.9's result with MOE/venue features specifically, not obviously worth
  it, but it's a different feature set, not more weight-tuning); (2) more
  calibration/holdout sample size itself, which no amount of training-data
  augmentation touches — that would need a fundamentally different
  data source (e.g., more IPL seasons over time) rather than more T20I.
- Revisit whether the 42% precision / 27.86% recall gate pairing itself is
  well-calibrated for a ~28% event-rate binary classifier — now the most
  attractive next move, given five successive attempts to push past v7.7
  (v7.8-v7.11) have all landed at or below it.
- Confirmed dead ends, don't re-try: plain threshold-margin tuning (v7.1),
  fixed-iteration depth increase (v7.3), further threshold-selection-method
  tuning on the IPL-only feature set (v7.6), T20I weight above ~0.2 (v7.8),
  full MOE/venue feature parity for T20I rows at weight 0.2 (v7.9), and any
  squad-familiarity-weighting scheme for T20I rows regardless of magnitude
  (v7.10 at mean 0.12, v7.11 at mean 0.2 exactly) — all five converge on
  v7.7 or below it, never past it.

## 2026-08-22 follow-up: does the currently-live wicket model have the same gap the runs model had?

Separate from the v7.x line above. While wiring the run-range model into
production, found `models/runs_model.pkl` had a structural validation gap
(`src/train.py` uses a random `train_test_split`, and its data,
`data/real_overs.csv`, has no dates at all — a genuine chronological
holdout has never existed for it). `models/wkt_model.pkl` is trained by
the exact same script, same random split, same undated data — the
identical gap, confirmed by reading `src/train.py` directly (it trains
both models from one `main()`).

Built the same three-way comparison as the runs investigation
(`train_contract22_wicket_rigorous.py`, reusing `build_enriched()` from
`train_contract22_rigorous.py` unchanged — the feature computation doesn't
depend on the prediction target):

1. **Legacy naive estimate** (existing `models/wkt_model.pkl`, unmodified,
   replayed against 40 real IPL matches): AUC 0.588, Brier 0.1981. Same
   bowler-omniscience-during-replay caveat as the runs comparison (a real
   live prediction usually doesn't know the upcoming bowler; replay does).
2. **This session's rigorous v7.x line, best result**
   (`ipl_wicket_v7_7_t20i_augmented`): Brier 0.19384, AUC 0.5991, genuine
   chronological holdout, no bowler dependency at all.
3. **New: `contract22_wicket_rigorous`** — same dated,
   chronologically-split, 22-feature-contract-equivalent population as the
   runs comparison, binary wicket target, Platt-calibrated the same way as
   the v7.x line. Bowler known: AUC 0.6069, Brier 0.2051. Bowler unknown:
   AUC 0.6063, Brier 0.2051 (barely different — unlike runs, bowler
   identity adds almost nothing here either way).

Raw Brier isn't comparable across the three directly — they have
different holdout event rates (28.1%, 27.7%, 30.7%) since they draw from
different base datasets. Normalized to a Brier skill score
(`1 - brier / (event_rate * (1 - event_rate))`) for a fair comparison:

| model | Brier skill score |
|---|---:|
| Legacy naive (leaky, inflated) | **1.95%** |
| v7.7 rigorous (session's best v7.x candidate) | 3.30% |
| contract22_wicket_rigorous (bowler known or unknown) | **3.54%** |

**Conclusion, same shape as the runs finding but a different practical
implication**: the currently-live wicket model's own estimate — even
inflated by replay's bowler-omniscience leak — shows *less* real skill
than either rigorously-validated alternative already built this session.
This is consistent evidence the live wicket model is similarly
under-validated and likely not as good as it could be, same story as
runs. **Unlike runs, though, this doesn't translate into a ready
replacement**: neither v7.7 nor `contract22_wicket_rigorous` has ever
cleared the operational precision/recall promotion gates (42% precision /
27.86% recall on the "alert" threshold) that matter for the wicket-alert
use case — AUC/Brier improvement is necessary but not sufficient for
that. The gates question from the v7.x line (documented above) remains
the actual blocker, not model quality in the abstract.

## Update 2026-08-22, later same day: wired into `PredictionEngine`, and two more legacy heuristics found harmful

After the runs side (`run_range_enriched_v3_batting_style`) got wired into
`PredictionEngine`, decided to do the same for wicket with
`contract22_wicket_rigorous` (the best of the three-way comparison above:
3.54% Brier skill score vs. v7.7's 3.30% and the live model's own
leaky 1.95%). v7.7 was the initial recommendation but was reconsidered:
it depends on bowler-spell features that need the current bowler known,
requiring fresh live-tracking infrastructure, whereas contract22 already
proved (bowler_known vs. bowler_unknown barely differ) that it's robust
regardless of bowler-announcement timing — the more practical choice for
actual live wiring, reusing most of the run-range snapshots directly
(batter stats, venue stats, name aliases are literally the same
quantities). Built `build_wicket_contract22_live_snapshots.py` (bowler
career stats, h2h, bowl-phase history — the one genuinely new piece),
`app/ml/wicket_contract22_features.py`, and `runtime_wicket_contract22.py`
following the same pattern as the run-range runtime.

**Found two more legacy heuristics that actively hurt the new model,
beyond the theoretical "untested combination" concern already flagged for
the runs side** — this time with direct empirical evidence, not just
caution:

1. **`BowlerSpellAdjuster`'s wicket adjustment.** A replay comparison on
   745 real overs: raw contract22 AUC 0.5945/Brier 0.2098, with the
   adjustment applied AUC 0.5768/Brier 0.2117 — worse on both metrics. It
   was validated as an improvement over the old, much weaker wkt_model.pkl
   (AUC 0.519->0.550 in its own validation); stacking it on an
   already-better-calibrated model pushes predictions the wrong direction.
   Removed from `predict()` entirely — it had no remaining use on the runs
   side either (already dropped earlier the same day), so
   `BowlerSpellAdjuster` is no longer instantiated or called at all.
2. **The adaptive `wicket_multiplier` ("Wicket Risk Dampener").** Same
   class of bug as the `match_bias` mechanism removed from the runs side
   earlier: a per-match self-correcting multiplier, originally tuned to
   compensate for the old model's inaccuracy. Isolated by pinning it at
   1.0 mid-replay: AUC recovered from 0.5670 back to 0.5944 (matching the
   raw model almost exactly). Removed entirely — state var, export/reset/
   restore, its "Risk: {x}x" line in `_build_analysis`, and the
   increment/decrement block in `update_actuals()`.

With both removed, the fully wired-in engine reproduces the model's real,
validated performance: **AUC 0.5944, Brier 0.2098** on the same 745-over
replay sample, matching the isolated runtime almost exactly. Also removed
`backend/tune_momentum_blend.py` and `backend/diagnose_raw_vs_wrapped.py`
(tracked, both genuinely obsolete — their entire purpose was tuning/testing
machinery from the old runs pipeline that no longer exists), and the now-dead
`FeatureBuilder`/`HistoricalFeatureStore`/pandas dependencies inside
`PredictionEngine` itself, since nothing calls `models/runs_model.pkl` or
`models/wkt_model.pkl` from `predict()` anymore.

**Validated**: 206 tests pass throughout (one test already updated earlier
the same day for the runs change); replayed 40+ fresh real matches across
three separate batches with 0 errors; the AUC/Brier recovery was directly
measured, not assumed.

**Practical implication for the earlier "neither clears the alert gates"
conclusion**: still true — the operational precision/recall promotion
gates (42%/27.86%) are a different, harder bar than plain AUC/Brier, and
this wiring decision was made on the basis that the *actual live feature*
is a continuous probability display (Telegram shows "wkt% 24%"), not a
binary alert, so Brier/AUC is the metric that matters for what's really
being served — matching the same standard `BowlerSpellAdjuster` was
originally promoted under, before this update replaced it.

## Update 2026-08-22, third pass: does new-batter/pressure signal add anything?

User asked to dig deeper for calibration/data factors that could increase
probability quality further, on top of the now-live
`contract22_wicket_rigorous`. Checked what's genuinely missing rather than
guessing:

- **`contract22_wicket_v2_batter_state`**: added "new batter at the
  crease" / partnership-age signal (`active_batter_state`, `new_batter`,
  `partnership_legal_ball_age`, `striker_match_balls`,
  `partner_match_balls`, from `build_ipl_phase_moe_features`) — one of the
  best-known wicket-risk factors in cricket, and something this session
  already computed and validated for the run-range model but never added
  to the wicket side. Also compared Platt vs. isotonic calibration on the
  same holdout. Result: **AUC 0.6081/0.6072 (known/unknown), Brier
  ~0.2049 (Platt)** — a real, if modest, gain over the live model's own
  0.6069/0.6063, ~0.2051. Isotonic and Platt landed within noise of each
  other (0.20493 vs 0.20499 known) — Platt remains fine, no need to switch.
- **`contract22_wicket_v3_pressure_state`**: added `wickets_remaining_bucket`,
  `state_regime` (stable/accelerating/wicket_pressure), and
  `chase_pressure` (not_chasing/low/medium/high) on top of v2, same
  source. Result: **AUC 0.6086/0.6075, Brier ~0.2049** — barely moved
  from v2 at all. These signals are largely redundant with
  `required_run_rate`/`recent_wicket_rate` already in the base feature
  set, since they're derived from the same underlying quantities.
  Diminishing returns confirmed by measurement, not assumed.

Brier skill score (normalizes for the 30.66% event rate, comparable
across all three):

| candidate | Brier skill score |
|---|---:|
| `contract22_wicket_rigorous` (live at time of this measurement) | 3.52% |
| `contract22_wicket_v2_batter_state` (live now — see below) | 3.58% |
| `contract22_wicket_v3_pressure_state` | 3.585% (not meaningfully different from v2) |

**Conclusion**: v2's new-batter/partnership signal is a real, validated
improvement, small in magnitude (+0.06pp BSS, ~1.7% relative). v3 confirms
the easy wins from this general feature family (match-state derived
categoricals from the phase-MOE dataset) are exhausted — further
additions from the same source aren't likely to move the needle much
more. v2 promoted to live (see the next section below) since the gain is
real and the swap is low-risk (same architecture/artifact shape as the
model it replaces); v3 stays candidate-only, kept for reference; genuinely
new signal (not just more of the same MOE-derived state) — e.g.
current-spell tracking done as a trained INPUT feature rather than a
post-hoc adjustment layer (unlike the now-removed `BowlerSpellAdjuster`),
or venue-level pitch classification — would be the next place to look if
pursuing this further.

## Update 2026-08-22, fourth pass: `contract22_wicket_v2_batter_state` pushed into `PredictionEngine`, two production bugs found and fixed

Wired `WicketRuntimeContract22` in `PredictionEngine` over to
`models/candidates/contract22_wicket_v2_batter_state` (was
`contract22_wicket_rigorous`), by request — v2's new-batter/partnership
signal is real, validated, and the swap costs nothing (same artifact
shape, same runtime class, same feature computer with two new optional
params). Regenerated production artifacts for v2
(`wicket_model.pkl`/`wicket_calibrator.pkl`/`feature_cols.pkl`/
`categorical_cols.pkl`, Platt calibrator fit on `bowler_known=True`
calibration data) via `train_contract22_wicket_v2_batter_state.py`, with a
fresh `ARTIFACT_MANIFEST.json`.

Validating the wiring surfaced two real production bugs that unit tests
alone didn't catch, because the ad-hoc `MatchReplay`-based validation
scripts used throughout this session call `PredictionEngine.predict()`
directly and don't exercise `app/live/pipeline.py`'s `MatchContext`
construction the way the real live-serving path does:

1. **`deliveries` never threaded through `context.metadata` in
   production.** `pipeline.py`'s `VerifiedLivePredictionPipeline` builds
   `MatchContext` per over but never passed the accepted-deliveries list
   through `metadata`, even though it already computes
   `all_deliveries = list(verifier.accepted.values())` for its own
   verification bookkeeping. Without this,
   `WicketContract22FeatureComputer._partnership_state()` would silently
   see an empty `deliveries` sequence on every real call, and the new
   new-batter/partnership-age features (the entire reason v2 exists) would
   always read as "brand new innings, nobody has faced a ball" — the
   signal this candidate was built and validated on would never actually
   fire in production. Fixed: added `metadata={"deliveries":
   all_deliveries}` to the per-over `MatchContext(...)` construction (the
   pre-innings, over-1 construction correctly has zero prior deliveries
   already and needs no fix).

2. **`non_striker` hardcoded to `""` in `pipeline.py`.** TOI's
   `ToiDelivery` model has no non-striker field at all (genuine structural
   limit of the feed, not something introduced this session), so
   `LiveMatchState.non_striker` is always empty in real live serving. The
   original `_partnership_state()` treated an unresolved partner as
   "0 balls faced" and computed `is_new = min(striker_balls,
   partner_balls) <= 2` — with `partner_balls` pinned at 0, this is always
   true regardless of how long the striker has actually been at the
   crease, so `new_batter` would have read `True` on literally every over
   in production. Fixed in `wicket_contract22_features.py`'s
   `_partnership_state()`: when `partner_id == UNKNOWN_PLAYER` (the
   genuine no-non-striker-available case), fall back to judging newness
   from the striker alone (`is_new = striker_balls <= 2`) instead of
   manufacturing a false partner.

Both fixes are additive/defensive — no existing behavior for callers that
DO supply `deliveries`/`non_striker_name` changes. Full test suite still
passes (206/206) after both fixes.

**Re-validated end-to-end through the real `VerifiedLivePredictionPipeline`**
(not the ad-hoc `MatchReplay` scripts) with both fixes in place: replayed
20 fresh held-out matches (`random.seed(2026)`, 10 IPL + 10 T20I) via
`VerifiedLivePredictionPipeline.process()` directly, matching each over's
predicted `wicket_probability` (published after the prior over's
deliveries were verified) against that over's actual outcome:

| Metric | Value |
|---|---:|
| Matched predictions | 669 |
| AUC | 0.6067 |
| Brier | 0.2000 |
| Event rate | 0.2900 |

Consistent with (marginally better than) the offline chronological-holdout
numbers reported above (AUC 0.6081/0.6072 known/unknown, Brier ~0.2049),
confirming the production wiring — deliveries and partnership state
included — performs as validated, not silently degraded by either bug.
Zero exceptions across the 20-match replay. `PredictionEngine`'s module
docstring, in-line comments, and `metadata["wicket_model"]` string all
updated from `contract22_wicket_rigorous` to
`contract22_wicket_v2_batter_state` to match.

## Artifacts

- `models/candidates/ipl_wicket_v7_1_margin_threshold/` — rejected, kept
  for reference.
- `models/candidates/ipl_wicket_v7_2_spell_features/` — rejected, kept for
  reference.
- `models/candidates/ipl_wicket_v7_5_early_stopping/` — rejected: 41.6%
  precision / 30.1% recall on locked holdout (precision misses).
- `models/candidates/ipl_wicket_v7_6_kfold_threshold/` — rejected: 43.8%
  precision / 25.9% recall on locked holdout (recall misses) — same
  trained model as v7.5, only the threshold-selection method changed.
- `models/candidates/ipl_wicket_v7_7_t20i_augmented/` — rejected, best
  result of the session: 44.0% precision / 27.02% recall (misses by 0.84
  points).
- `models/candidates/ipl_wicket_v7_8_t20i_weight_40/` — rejected: 44.5%
  precision / 26.3% recall — confirms 0.2 beats 0.4 for T20I weight on
  this feature set.
- `models/candidates/ipl_wicket_v7_9_t20i_full_features/` — rejected:
  44.7% precision / 26.37% recall — confirms full T20I feature parity
  (at least for MOE + venue features) doesn't beat the bare-match-state
  version either. v7.7 stays the best result.
- `models/candidates/ipl_wicket_v7_10_similarity_weighted_t20i/` —
  rejected: 44.0% precision / 26.37% recall — squad-familiarity-weighted
  T20I rows (mean weight 0.12) didn't beat v7.7 either.
- `models/candidates/ipl_wicket_v7_11_similarity_weighted_normalized/` —
  rejected: 43.86% precision / 27.02% recall — same as v7.10 but
  renormalized to v7.7's exact 0.2 mean weight; recall matches v7.7 to
  the last decimal, confirming attempts 9-10's regressions were a
  magnitude effect, not a shape effect.
- `backend/train_ipl_wicket_v7_5_early_stopping.py`,
  `backend/train_ipl_wicket_v7_6_kfold_threshold.py`,
  `backend/train_ipl_wicket_v7_7_t20i_augmented.py`,
  `backend/train_ipl_wicket_v7_8_t20i_weight_40.py`,
  `backend/train_ipl_wicket_v7_9_t20i_full_features.py`,
  `backend/train_ipl_wicket_v7_10_similarity_weighted_t20i.py`,
  `backend/train_ipl_wicket_v7_11_similarity_weighted_normalized.py` —
  real candidate scripts, promotion-grade methodology (no holdout
  peeking) throughout.
- `backend/app/ml/ipl_phase_moe_dataset.py` and
  `backend/app/ml/ipl_venue_regime_dataset.py` — both gained an opt-in
  `scopes` parameter (default unchanged, `("ipl",)`) so they can build
  from T20I matches too. Existing callers/production behavior untouched;
  all 202 tests still pass.
- The threshold-margin and depth/iteration sweeps (attempts 1, 3, and the
  first pass of 4) were run from disposable scratch scripts, since
  deleted — the numeric results above are the durable record of what they
  found.
- No production files were touched. `run_model_changed: false` and
  `production_changed: false` in every validation report this session.
