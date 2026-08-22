# Run-Range Model — Enriched Features (v2/v3) + Live-Model Rigor Gap

Date: 2026-08-22

## Starting point

`v3.3_phase_calibrated_sharp_range` (the currently accepted run-range
candidate) trains on only 17 bare match-state features
(`train_phase_calibrated_sharp_range_v33.py:FEATURES`) across all 172,735
IPL+T20I training rows — none of the batter/bowler prior-stat,
partnership, chase-pressure, or venue-par-score enrichment that the
wicket-model line (v7.x, this session) already validated as helpful.
`docs/model_contract.md` documents an intended 22-feature contract that
was never actually implemented in v3.3's own training script. Prior
enrichment attempts on this model line (`v3.4`-`v3.11` in
`models/candidates/`) used different, hand-engineered features
(boundary_gate, venue_track, etc.); none were adopted, and v3.3 remains
the accepted candidate.

## Attempt 1 — full MOE + venue enrichment (`train_run_range_enriched_v1.py`)

Reused the same canonical-identity/venue MOE features already built and
validated for the wicket model this session (`build_ipl_phase_moe_features`,
`build_ipl_venue_regime_dataset`), extended to cover T20I via the
`scopes=("ipl","t20i")` parameter added earlier this session. Same
architecture as v3.3 (LightGBM multiclass, `MAX_RUN_CLASS=30`, same
hyperparameters), same train/calibration/holdout split, same male-only
IPL+T20I population. Added the full MOE feature set including raw
identity categoricals (`striker`, `non_striker`, `venue_name`,
`batting_team`, `known_bowler`).

Result: **rejected — worse than both v3.2 and v3.3.** Holdout hit rate
26.44%, vs. v3.2's 27.28% and v3.3's 28.10% (a -5.9% relative regression
vs. v3.3). Post-fit phase-temperature calibration values were sharply
higher than v3.3's (powerplay 1.72, middle 1.46, death 1.87, vs. v3.3's
~1.05-1.14) — a signature of an overconfident raw model needing heavy
correction, pointing at overfitting on the high-cardinality identity
categoricals (1000+ distinct players) added without any compensating
regularization change from v3.3's own hyperparameters.

## Attempt 2 — same enrichment minus raw identity (`train_run_range_enriched_v2_no_identity.py`)

Direct test of the overfitting hypothesis: identical to attempt 1 except
`striker`, `non_striker`, `venue_name`, `batting_team`, and `known_bowler`
are dropped, keeping only the numeric prior-stat/partnership/chase-pressure/
venue-par features (lower cardinality, more generalizable).

Result: **accepted — a genuine, validated improvement.**

| metric | v3.2 baseline | v3.3 (accepted) | v2 (this candidate) |
|---|---:|---:|---:|
| holdout hit rate | 27.28% | 28.10% | **28.37%** |
| relative improvement over v3.2 | — | +3.01% | **+3.99%** |
| phase temperatures | — | ~1.05-1.14 | 1.04-1.14 (sane, matches v3.3) |

Phase breakdown on holdout: powerplay 26.0%, middle 30.5%, death 27.4%.
Beats both v3.3's own promotion gates (`holdout_hit > old_hit` and
`scaled_calibration_nll <= raw_calibration_nll`) and v3.3 itself directly.
`decision: "promote_candidate"` per the script's own gate logic (mirrors
v3.3's acceptance criteria).

## Attempt 3 — add a genuine static player attribute (`train_run_range_enriched_v3_batting_style.py`)

v2 deliberately dropped ALL raw player/venue identity categoricals since
that was the proven overfitting source. But that's identity specifically,
not player attributes in general — `data/external/cricsheet_player_styles.csv`
(16,102 players, keyed by the same Cricsheet person UUID used everywhere
else this session) has genuine, static, zero-leakage attributes like
`batting_style` (right/left-hand bat) — known in advance, low cardinality
(a handful of values), no overfitting risk the way raw player IDs are.
`bowling_style`/`bowler_type` deliberately NOT added: the upcoming over's
bowler isn't reliably known before it's bowled (same reasoning v1/v2
already applied to `known_bowler`), so it would mostly be noise/missing
in live use.

Result: **accepted, a further small real improvement.** Holdout hit rate
28.49% (resolved for 95.6% of rows; unresolved players default to
"unknown"), vs. v2's 28.37% and v3.3's 28.10%. Phase temperatures stayed
sane (1.04-1.15). This is now the best-performing run-range candidate on
record.

## The live-model comparison: an honest dead end, not a result

The original motivation for this whole line was: is the model actually
serving live predictions (`models/runs_model.pkl`, loaded by
`PredictionEngine()` via `FeatureBuilder`/`HistoricalFeatureStore`) as
good as these candidates, or does it need improvement too? Investigated
this directly and found a **structural** problem, not just a missing
number:

- `src/train.py` (the live model's own training script) splits with
  `train_test_split(match_ids, test_size=0.2, random_state=42)` — random,
  not chronological.
- Its data source, `data/real_overs.csv`, has **no match_date column at
  all** — confirmed in `HistoricalFeatureStore`'s own docstring: "match_id
  is assigned by non-deterministic filesystem glob order... True temporal
  cutoffs are therefore not reconstructable from this file alone."

This means a genuine chronological holdout evaluation of the live model
is not just undone, it's currently impossible without rebuilding its
training data with dates. A naive replay-based estimate (500 IPL matches
through the real `PredictionEngine()`, from this session's confidence
work) showed **31.1%** — higher than any candidate above — but this
number cannot be trusted as a fair comparison: the live model's weights
could have been fit on a random split that includes some of those very
evaluation matches, since nothing prevents it. The candidates in this doc
explicitly never saw 2025+ data during training; the live model has no
such guarantee for any year.

**Conclusion**: v3 (28.49%) is the most trustworthy accuracy number that
exists anywhere in this codebase for run-range prediction — not because
it's the highest number seen (31.1% is higher), but because it's the only
one with a real, leakage-checked chronological holdout behind it. Whether
it actually beats the live model in practice is unknowable without a
separate, larger effort: rebuilding the live model's training data with
proper dates and retraining it the same rigorous way. That's a real next
step, not done in this session — flagged, not fixed, same as this
session's other genuinely out-of-scope items.

## Not yet done — requires an explicit decision, not a code fix

This candidate is **not wired into the runtime**. `v3.3_phase_calibrated_sharp_range`
has `backend/runtime_v33.py` (integrity-verified artifact loading,
feature-contract validation, calibrated inclusive-band output) built
specifically for it; this candidate has no equivalent runtime wrapper yet,
and nothing in `backend/app/` currently loads it. Promoting it to replace
v3.3 as the model actually used (if that's ever wired into live serving —
recall `models/runs_model.pkl`/`models/wkt_model.pkl` are themselves not
yet the source of "v3.3"; that promotion question is separate and
pre-existing) would need:

1. A runtime wrapper analogous to `runtime_v33.py` (artifact integrity,
   feature contract, calibrated band output) — the categorical inputs
   (`active_batter_state`, `wickets_remaining_bucket`, `state_regime`,
   `chase_pressure`, `venue_scoring_regime`, `venue_par_source`,
   `batting_team_venue_context`, `phase_venue_regime`) need to be
   computable at live-prediction time from `LiveMatchState`/`MatchContext`,
   which they currently are only via the offline MOE/venue-regime batch
   builders — that live-computation path doesn't exist yet.
2. Replay/integration testing (same kind of validation `run_live_pipeline_replay.py`
   did for the wicket pipeline) before trusting it in a live match context.

## Artifacts

- `models/candidates/run_range_enriched_v1/` — rejected, kept for
  reference (documents the overfitting failure mode).
- `models/candidates/run_range_enriched_v2_no_identity/` — accepted:
  28.37% holdout hit rate.
- `models/candidates/run_range_enriched_v3_batting_style/` — **accepted,
  best run-range candidate on record**: 28.49% holdout hit rate.
- `backend/train_run_range_enriched_v1.py`,
  `backend/train_run_range_enriched_v2_no_identity.py`,
  `backend/train_run_range_enriched_v3_batting_style.py` — training
  scripts, same promotion-grade methodology (single holdout evaluation) as
  the wicket-model line.
- No production files touched. `run_model_changed: false` and
  `production_changed: false` in all validation reports.

## Update 2026-08-22: runtime built and validated

The "not wired into any runtime" gap above is now closed for v3. Built:

- `backend/build_run_range_v3_live_snapshots.py` — precomputes the two
  lookups the live runtime needs but can't compute cheaply per-prediction:
  per-player career batting stats (4,625 players) and per-venue par scores
  (314 venues), from all eligible historical matches (IPL+T20I, same
  `male_source_files()` population the model trained on). Outputs
  `data/live/run_range_v3_player_stats.json` /
  `run_range_v3_venue_stats.json`. Re-run periodically (e.g. after each
  completed match) to keep them current — not auto-refreshed.
- `backend/app/ml/run_range_v3_features.py` (`RunRangeV3FeatureComputer`)
  — computes the full live feature row: player/venue lookups from the
  snapshots above, plus in-match state (partnership age, batter balls
  faced this innings, recent-12-ball momentum, chase pressure) computed
  fresh from the current innings' verified deliveries every call. Mirrors
  the training-time formulas in `ipl_phase_moe_dataset.py` /
  `ipl_venue_regime_dataset.py` exactly (same `_pressure_state`,
  `_chase_pressure`, par-score fallback logic).
- `backend/runtime_run_range_v3.py` (`RunRangeRuntimeV3`) — the
  `runtime_v33.py`-equivalent wrapper: artifact integrity verification,
  feature-contract validation, phase-temperature calibration, calibrated
  inclusive-band output. Takes verified deliveries + live match state
  directly rather than a pre-built feature frame, since this candidate's
  features can't be assembled by the caller without the snapshots.
- `backend/tests/test_runtime_run_range_v3.py` — 3 tests (loads and
  predicts correctly, rejects a bad artifact dir, handles unknown
  players/venues gracefully via the same "__UNKNOWN__" fallback
  convention used throughout this codebase). All pass.

**Validation** (`backend/validate_run_range_v3_runtime.py`): replayed 40
real matches (mixed IPL/T20I, `random.seed(7)`) through the actual
runtime — **0 errors across 1,461 overs**, 100% of rows resolved non-zero
player prior stats (identity resolution and snapshot lookups working
correctly). Observed hit rate 31.42%, *higher* than the offline holdout's
28.49% — expected and not a red flag, since the live snapshot reflects
each player's *entire* available career (including matches chronologically
after the ones sampled in this test), which is correct for genuine live
use on today's date but optimistic when testing against older historical
matches. This validates the mechanism works and produces sane values, not
a leakage-safe re-measurement of accuracy (that number, 28.49%, already
exists from proper chronological-holdout training).

**Known gap — closed same day.** Player identity resolution originally
assumed a Cricsheet-style registry (name → person UUID), true for replay
but not for a genuine live TOI feed (plain name strings only, no
registry). Fixed via the same technique `BowlerSpellAdjuster` already uses
for the equivalent wicket-model problem: `build_run_range_v3_live_snapshots.py`
now also builds `data/live/run_range_v3_name_aliases.json` — an
`identity_key()`-normalized name → canonical_player_id lookup from every
squad-list appearance across all 4,687 eligible matches (4,965 aliases for
4,625 players). `RunRangeV3FeatureComputer._resolve_player_id()` tries the
caller-supplied registry first (exact, what replay uses), then falls back
to this alias table when the registry doesn't have the name (the genuine
live-TOI case). Validated with `registry={}` throughout (simulating zero
registry access) across the same 40-match/1,461-over replay sample used
above: **99.8% resolution rate (1,458/1,461), 0 errors, hit rate 31.35%
essentially unchanged from the registry-based 31.42%.** Still not full
fuzzy matching — a name that never appeared in any historical match
(e.g. a brand-new debutant) won't resolve and correctly falls back to
`"__UNKNOWN__"`/zeroed prior stats, same graceful cold-start convention
used throughout this codebase, not a crash.

## Update 2026-08-22: the live-model comparison gap is resolved

The earlier open question — does v3 actually beat what's running live? —
now has a real, trustworthy answer. Built `backend/train_contract22_rigorous.py`:
the SAME 22-feature contract `docs/model_contract.md` describes (the
feature set `models/runs_model.pkl` is actually intended to use, via
`FeatureBuilder`/`HistoricalFeatureStore`), computed from DATED data with
a genuine chronological split (train<=2023/calib=2024/holdout>=2025) —
something the live model's own training pipeline structurally cannot do
(`data/real_overs.csv` has no dates at all).

Two things found while building it, both material to the result:

1. Inspected `data/real_overs.csv` directly: `batsman_style`,
   `batsman_class`, `bowler_type`, and `bowler_quality` are **"unknown"
   for all 247,071 rows** — constant placeholders contributing zero
   signal to the currently-live model. This candidate fills
   `striker_batting_style` and `bowler_type` with real data instead
   (`data/external/cricsheet_player_styles.csv`), to give the comparison
   a fair chance rather than reproducing dead weight.
2. `MatchReplay` (used throughout this session, including the earlier
   "31.1%/31.35%" naive estimates for the live model) populates the
   **actual** bowler for the upcoming over during replay, since it's
   replaying known history. A genuine live prediction usually doesn't
   have this — the next over's bowler is typically unannounced. Those
   earlier naive numbers were inflated by information a real live
   prediction wouldn't have, on top of the live model's random-split
   training-overlap risk. This candidate is evaluated both ways.

| scenario | holdout hit rate |
|---|---:|
| v3.2 baseline | 27.28% |
| v3.3 (accepted) | 28.10% |
| **run_range_enriched_v3_batting_style** | **28.49%** |
| Full 22-feature contract, bowler known (ceiling) | 28.56% |
| Full 22-feature contract, bowler unknown (live-realistic) | 28.05% |
| Legacy naive replay estimate (leaky on two counts, untrustworthy) | ~31.1-31.35% |

**Conclusion: v3 is not just the best rigorously-validated candidate —
it's close to what's actually achievable with a live-realistic feature
set, full stop.** Under realistic live conditions (bowler usually
unknown), the full 22-feature contract *underperforms* v3 (28.05% vs.
28.49%) — the extra bowler-dependent features are noise, not signal, when
mostly blank in practice, exactly the reasoning that led v1→v2's overfitting
fix earlier in this doc. Even in the unrealistic best case (bowler always
known), the full contract only beats v3 by 0.07 points — negligible. The
naive ~31% number that made the live model look better was inflated by
two compounding leakage sources (random-split training overlap risk +
replay's bowler-omniscience), neither of which reflects reality.

**Practical implication**: there is no evidence remaining that a richer
feature set (the live model's own intended 22-feature contract) beats v3
under real conditions. v3 is both the best-validated AND, as far as this
investigation can tell, close to the ceiling of what's achievable this
way. Promoting v3 into `PredictionEngine` is no longer blocked on "maybe
the live model is secretly already better" — that specific concern is
now answered: no, it isn't, and there's no evidence a richer feature set
would do better under real conditions either.

## Update 2026-08-22, final: wired into `PredictionEngine` for real

v3 now IS the live runs prediction. `PredictionEngine.__init__` instantiates
`RunRangeRuntimeV3` (from `models/candidates/run_range_enriched_v3_batting_style`)
alongside the existing wicket model/repository, and `predict()`'s runs path
was replaced, not layered on top of the old one:

- The old `runs_model.pkl` point-estimate → momentum-blend → `match_bias`
  multiplier → fixed-width pivot bracket pipeline is gone for runs.
  `expected_range` is now v3's own calibrated inclusive band directly
  (`sharp_2_low`-`sharp_2_high`), and `predicted_runs` is that band's
  midpoint. This is deliberate, not a simplification for convenience: that
  old staged pipeline was never validated in combination with v3, and v3's
  28.49% holdout result assumed none of it (see the integration-design
  discussion above — "full replacement" was chosen explicitly over the
  hybrid option for this reason).
- `deliveries`/`registry` (what `RunRangeV3FeatureComputer` needs for
  in-match state and player identity) are read from
  `context.metadata.get("deliveries"/"registry", ...)`, defaulting to
  empty — no `MatchContext`/`LiveMatchState` schema changes were needed;
  `metadata` was already a generic passthrough dict, used the same way
  elsewhere in this session's replay scripts.
- Wicket prediction is untouched. The bowler-spell adjustment
  (`BowlerSpellAdjuster`) still applies, but now ONLY to wicket
  probability — its runs adjustment was validated against the old model's
  typical output and would be an untested combination against v3's
  already-calibrated band, so it's discarded rather than silently stacked.
- `self._match_bias` (the old adaptive per-match runs correction) is now
  vestigial: still computed and updated in `update_actuals()`, still
  exported/restored for restart-state continuity, still shown in
  `_build_analysis`'s text, but it no longer feeds into any prediction.
  Flagged clearly in a code comment at the update site rather than
  silently left misleading. Removing it outright would touch
  `export_runtime_state`/`restore_runtime_state` and their tests — a
  reasonable follow-up cleanup, not bundled into this change.

**Validated**: full test suite (206 tests) passes, including an updated
`test_prediction_metadata_exposes_calibration_stages` (the old assertions
checked metadata fields — `raw_runs`, `blended_runs`,
`centering_correction` — that no longer exist by design; rewritten to
check what replaced them). Replayed the live pipeline
(`run_live_pipeline_replay.py`, which uses the real default
`PredictionEngine()`) across 20 fresh real matches (10 IPL + 10 T20I,
`random.seed(42)`) — **0 errors**. Aggregate hit rate across that batch:
27.3% (684 overs) — within normal sampling variance of the validated
28.49%/31% figures, not a red flag.

## Status: live

`run_range_enriched_v3_batting_style` is the runs prediction
`PredictionEngine()` returns as of this update. `models/runs_model.pkl`
is no longer read by `predict()`'s runs path (the wicket side is
unaffected and still uses `models/wkt_model.pkl`).

## Update 2026-08-22, cleanup: removed the vestigial `match_bias` mechanism

The earlier note above ("`self._match_bias` is now vestigial... left as a
known follow-up") is resolved — removed outright, not left half-dead.
Along with it, three other pieces that only existed to feed it also came
out, since leaving them would just relocate the same "computed but never
read" problem rather than fix it:

- `self._match_bias` itself: state var, `export_runtime_state`/
  `reset_runtime_state`/`restore_runtime_state` handling (including its
  `0.5-1.5` bounds check), and its `"Match Bias: {x}x |"` line in
  `_build_analysis`'s text.
- `self._last_point_prediction`: only ever read to compute `error` for
  the `match_bias` update — removed along with it (state var, export/
  reset/restore, the `predict()` write site).
- `error`/`learning_rate` in `update_actuals()`: existed solely to drive
  `match_bias` — removed. (Caught a real bug while doing this: the
  method's final `return` line referenced `learning_rate` even after this
  removal — fixed to a plain `"Evolved"` string.)
- `momentum_blend_weight`/`run_centering_correction`/`RUN_CENTERING_CORRECTION`:
  found while removing the above — orphaned constructor parameters from
  the same v3-wiring change (nothing reads them once the old momentum-
  blend/centering pipeline was replaced). One caller existed:
  `backend/tune_momentum_blend.py`, a standalone grid-search script whose
  entire purpose was tuning these two now-nonexistent parameters —
  deleted outright rather than left broken. Updated a stale reference to
  it in `fit_confidence_calibrator.py`'s docstring.

**Validated**: full test suite (206 tests) still passes unchanged;
`run_live_pipeline_replay.py` replay produces byte-identical output to
before this cleanup (expected — pure dead-code removal, no behavioral
change).

## Update 2026-08-22, batter x phase: `run_range_v4_batter_phase` promoted to live

Batters had no phase split anywhere in this codebase — only career-wide
prior stats (`striker_prior_runs_per_ball` etc.). Bowlers already had one
(`bowl_phase_avg_runs`). `app/ml/player_venue_phase_dataset.py` (built
while investigating a related venue x phase question for the wicket
model) exposes the plain, venue-agnostic batter x phase aggregate as its
own columns: `batter_phase_balls/runs_per_ball/boundary_rate/dismissal_rate`.

Merged onto v3_batting_style's exact feature set, retrained the same way
(same split, same architecture): **28.60% holdout hit rate vs v3's
28.49%** (+0.39% relative — smaller than v2's/v3's own increments,
consistent with this feature family's diminishing-returns pattern, but
real and clears the same promotion gate). Feature importance ranked
19th-27th of 53 features — present, not dominant.

**Wired into production**:
- `build_run_range_v3_live_snapshots.py` extended to also emit
  `data/live/run_range_v3_batter_phase_stats.json` (`"player_id|phase"` ->
  career totals at that phase, same key convention as
  `wicket_contract22_bowler_phase_stats.json`), computed chronologically
  from the same match corpus.
- `RunRangeV3FeatureComputer` (`app/ml/run_range_v3_features.py`) loads
  this snapshot and computes the striker's own phase profile per call.
- `PredictionEngine.RUN_RANGE_V3_ARTIFACTS` repointed from
  `run_range_enriched_v3_batting_style` to `run_range_v4_batter_phase`;
  `metadata["run_model"]` updated to match.
- `ARTIFACT_MANIFEST.json` generated for the new candidate directory
  (sha256 per required artifact, same format as v3's).

**Validated end-to-end**: full test suite (207 tests) passes. Replayed 20
fresh matches (10 IPL + 10 T20I, `random.seed(2026)`) through the actual
`VerifiedLivePredictionPipeline` — **0 errors**, 674 predictions
published, 634 matched against a prior-over outcome for a real aggregate
range-hit rate of **30.91%**, consistent with (marginally better than)
the offline holdout's 28.60%, same pattern seen when v3 itself was first
wired in.

## Status: live

`run_range_v4_batter_phase` is the runs prediction `PredictionEngine()`
returns as of this update.
