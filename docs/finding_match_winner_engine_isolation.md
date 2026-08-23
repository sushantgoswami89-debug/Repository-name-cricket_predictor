# Decoupling match_winner_v1 into its own engine, with real fault isolation

Date: 2026-08-23. User request, immediately after `match_winner_v1` was
first wired in: "create this as a different engine with all these as
base files so that it dont interfere with our previous prediction model,
also we will have its output separately so that we dont get
interference." Two concerns, both addressed: architectural coupling
(a fault in the new model shouldn't be able to break run-range/wicket)
and output shape (its result shouldn't be merged into the existing
prediction dict).

## What changed

The first cut of `match_winner_v1` (see `finding_match_winner_v1.md`)
called `MatchWinnerRuntime` directly inside `PredictionEngine.predict()`,
alongside the wicket and run-range calls, and merged `win_probability`
into the same `PredictionResult` object those two already share. That
meant a bug or missing artifact in the win-probability path -- a
corrupted snapshot, a stale model file, an unhandled exception in its
feature computer -- would propagate up through the *same* `predict()`
call and take down run-range/wicket predictions too, even though those
are independently validated, already-live, and have nothing to do with
match outcome prediction.

**Reverted** `app/ml/prediction_engine.py` and `app/ml/prediction_result.py`
to their exact pre-match_winner_v1 state (no `MatchWinnerRuntime` import,
no `win_probability` field, no win-probability call in `predict()`) --
verified this by checking `PredictionEngine()` no longer has a
`_match_winner_runtime` attribute at all.

**Added** `app/ml/match_winner_engine.py`: a standalone `MatchWinnerEngine`
class with its own `MatchWinnerResult` dataclass (not `PredictionResult`),
its own team-identity/toss resolution (a small, deliberate duplication of
~15 lines from the old inline version -- cheap, and avoids any shared
state with `PredictionEngine`), wrapping the exact same
`MatchWinnerRuntime`/`MatchWinnerFeatureComputer`/model artifacts as
before. The underlying model, features, and validated numbers are
completely unchanged -- only the call boundary moved.

**Wired into `app/live/pipeline.py`** as a genuinely separate call,
mirroring the fault-isolation pattern the pipeline already uses for
`BowlerShadowPredictor` (try/except around the call, a graceful
`"status": "not_applied"` fallback on any exception, never let it
propagate). New `_predict_win_probability()` helper, called once per
prediction, output stored under its own top-level `output["win_probability"]`
key -- never nested inside `output["prediction"]`. Telegram text reads
from that separate key and simply omits the win-probability line if the
call failed, rather than showing a stale or fabricated number.

## Verified, not just asserted

- **Fault isolation, proven directly**: constructed the pipeline with a
  deliberately broken `MatchWinnerEngine` (raises on every call) and
  confirmed (a) the win-probability call degrades gracefully to
  `{"status": "not_applied", "reason": "win_probability_inference_failed", ...}`
  and (b) `PredictionEngine.predict()` -- called completely independently
  right after -- still returns a normal, correct run-range/wicket
  prediction. A broken win-probability engine cannot crash the rest of
  the pipeline.
- **Output separation, proven directly**: inspected a real pipeline
  output dict and confirmed `win_probability` exists only as a top-level
  key (`{"win_probability": 0.295, "batting_team": ..., ...}`) and
  `output["prediction"]` contains exactly the same 9 keys it always
  did (`expected_runs`, `expected_range`, `wicket_probability`,
  `confidence`, `confidence_percent`, `confidence_level`,
  `confidence_meter`, `analysis`, `metadata`) -- no win-probability leak
  into the shared prediction dict.
- **No regression**: 208 tests pass. A real IPL match
  (`1426261.json`) replayed through the actual production pipeline
  produces *identical* run/wicket predictions to before this refactor
  (same predicted runs, same wicket%, over by over) -- confirming the
  decoupling changed nothing about run-range/wicket behavior, only where
  win-probability's call boundary sits.

## What this means going forward

`MatchWinnerEngine` can be improved, retrained, or even temporarily
disabled without any risk to run-range/wicket. Its artifacts, snapshots,
and runtime are entirely its own; `PredictionEngine` has no knowledge
that it exists. If a future session wants to add a fourth prediction
type, this is the pattern to follow: separate engine, separate result
type, separate output key, wrapped in the same try/except isolation
`app/live/pipeline.py` already uses for `BowlerShadowPredictor`.
