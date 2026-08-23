# The win-probability model had the batting/bowling teams swapped during every chase

Date: 2026-08-23. Follow-up to a direct user question: given a real
scenario like India vs Pakistan where the team behind claws back via a
partnership, how genuinely does `match_winner_v1` track that swing?

## What looked like the finding, at first

Replayed the real 2022 T20 World Cup MCG chase (Pakistan 159, India won
by 4 wickets via the famous Kohli/Ashwin stand) through the actual
production pipeline, over by over. The win-probability trajectory was
genuinely erratic: a 20-run over crashed India's win probability from
89% to 62%; the very next over -- 3 runs and a wicket, objectively bad --
sent it back up to 95%.

To find out whether this was a one-match fluke, mined the historical
corpus for real "dramatic-recovery chases" (peak required run rate >=11
after over 10, chasing team still won) restricted to the 2025+/2026
holdout population (never used in training, per this project's standing
practice): **19 IPL + 37 T20I matches**. Replayed a 29-match sample (all
19 IPL, 10 T20I involving at least one full-member nation) through the
real pipeline and measured, for every over transition, whether the
win-probability move agreed in sign with the over's actual quality (runs
scored vs. required rate, wickets lost).

**Result: 18.2% agreement across 457 real over-transitions** -- worse
than a coin flip. Not noise; anti-correlated more often than not.

## Chasing the wrong fix first

Checked the trained model's feature importances directly
(`match_winner_model.pkl`): `required_run_rate` (#1), `runs_required`
(#3), and `current_run_rate` (#5) dominate by a wide margin over
per-player identity features. So the working hypothesis became "an
unconstrained gradient-boosted tree, scored independently at every over,
has no constraint forcing smooth output as one real match evolves" --
plausible, and built two EWMA-style live-smoothing layers to test it.
Plain EWMA reduced the wrong-direction rate but also diluted genuine
signal (a real death-overs required-rate spike that should have crashed
win probability to 54% only reached a lagging 72%). A smarter
"directional gate" (only damp a step when it disagrees with what
required_run_rate/wickets say happened) was built next -- and made
agreement rate **worse** as the gate tightened, the opposite of the
intended effect. That result was the real signal: a fix that makes a
diagnostic-consistent gate perform worse than doing nothing means the
diagnostic itself is measuring something other than what was assumed.

## The actual bug

Instrumented `MatchWinnerEngine._smooth` to log its internal
`required_run_rate`/`raw_delta` decisions and compared them directly,
transition by transition, against the audit's own independently-computed
numbers for the same over. They were near-perfect algebraic mirror
images -- `external_delta ≈ -(internal_delta)` -- which is exactly what
you get when one side is silently reading `100 - p` instead of `p`.

Root cause, in `app/ml/match_winner_engine.py`'s `predict()`:

```python
batting_team = (
    (context.bowling_first or context.team2)
    if is_chase
    else (context.batting_first or context.team1)
)
```

`context.batting_first`/`context.bowling_first` are declared on
`MatchContext` but **never set anywhere in the codebase** -- confirmed by
grep, zero matches outside their declaration. So the fallback always
fires: `context.team2` whenever `is_chase` is true, i.e. every second
innings. But `app/live/pipeline.py` sets `context.team1` /`context.team2`
to whichever team is *currently* batting/bowling
(`snapshot.batting_team`/`snapshot.bowling_team`) -- true for **both**
innings, not a fixed "team that batted first" role. `is_chase` is true
only in innings 2, where `context.team1` is *already* the chasing team.
The `is_chase` branch was written assuming a different, static-role
convention that this codebase's live pipeline has never actually used.

Net effect: **for the entire second innings of every live match this
session, `MatchWinnerEngine` fed the model the bowling team's name and
roster, labeled as the batting team.** Confirmed the training convention
matches `context.team1` exactly (`app/ml/match_winner_dataset.py`:
`batting_team = innings.get("team")`, per-innings, same idea) -- so the
fix is simply to stop branching and always trust `context.team1`/
`context.team2` as-is:

```python
batting_team = context.team1
bowling_team = context.team2
```

No test ever caught this: grepped `tests/*.py` for `MatchWinnerEngine`/
`match_winner` -- zero references. The "hand-built scenario" sanity
checks in `docs/finding_match_winner_v1.md` (0.99/0.015/0.576) were
constructed directly against the runtime, bypassing this exact
team-resolution code path.

## What this did and didn't affect

- **The trained model itself was never wrong.** `evaluate_match_winner_v1_accuracy.py`
  (the source of the previously-reported 76.3% accuracy figure) rebuilds
  features directly from the training-side dataset builder and never
  imports `MatchWinnerEngine` -- confirmed by inspection. That number
  stands unchanged.
- **Only live serving during a chase was affected.** Innings-1 (setting)
  predictions were never touched, since `is_chase` is false there and the
  buggy branch never fired.
- Downstream, this means the real Telegram output and dashboard would
  have shown a probability computed against the wrong team's roster
  (feeding wrong team-composition/venue-context/H2H features into the
  model) and, worse, potentially labeled with the wrong team name for the
  whole second innings of every live match to date.

## Verification

- Same 29-match, 2025+/2026 holdout audit, fix applied, **zero other
  changes**: agreement rate **18.2% -> 81.7%** (376/460 real
  over-transitions), with no smoothing of any kind.
- Remaining ~18% disagreement is now small-magnitude (a few percentage
  points on plausible overs), not the 20-30 point pathological reversals
  from before -- looked directly at the worst remaining cases and they
  read as ordinary model uncertainty, not a bug.
- Re-replayed the original 2022 IND-vs-PAK demo match end to end: the
  trajectory is now a coherent story -- 66.9% early, crashing to a low
  of 6.2% at the actual historical low point of the chase (45/4, over
  10), climbing steadily through the real Kohli/Ashwin partnership to
  28%, dipping again as the required rate spiked to 16, recovering to
  45.9% by the second-to-last over. No wild single-over reversals.
- 208 tests pass. A full real-match replay (`run_live_pipeline_replay.py`)
  completes with 0 errors on both innings; run-range and wicket numbers
  are byte-identical to before this change, since this fix only touches
  `MatchWinnerEngine`.

## What was tried and deliberately not kept

Two live-output smoothing layers (plain EWMA, then a "directional gate"
EWMA) were built and validated against the same 29-match audit while
chasing what turned out to be the wrong root cause. Once the real bug
was fixed, the smoothing gate's own marginal benefit shrank to +4 points
(81.7% -> 85.6%) against added statefulness and real risk of dulling
genuine signal (demonstrated directly: plain EWMA diluted a real 54%
death-overs crash into a lagging 72%). Not worth the complexity now that
the dominant distortion is gone -- reverted `MatchWinnerEngine` to a
plain, stateless `predict()`. If output smoothness becomes a real
complaint once this is observed on genuine live matches, revisit with
this audit harness already in hand (`swing_pattern_audit.py`-style
approach, not saved to the repo -- rebuild from this doc's method if
needed).

## Follow-up: LightGBM monotone_constraints tested, also not kept

Researching real-world cricket win-probability systems (WASP uses
dynamic-programming backward induction, which is monotonic by
construction) suggested the more "correct" fix flagged above --
`monotone_constraints` -- was worth actually testing rather than
speculating about. Constrained the 5 features with an unambiguous,
universally-true relationship to the outcome (`required_run_rate`,
`runs_required`, `wkts_down_before_over`: negative; `current_run_rate`,
`wickets_in_hand`: positive), left everything else (recency/context/
team features, where the "obvious" direction isn't always true) free.

Tested both `monotone_constraints_method` variants against the real
29-match audit and the full holdout:

| | AUC (known) | AUC (IPL known) | Brier (known) | Trajectory agreement |
|---|---:|---:|---:|---:|
| Live (unconstrained) | 0.8587 | 0.7889 | 0.15378 | 81.7% |
| `method="basic"` | 0.8579 | 0.7845 | 0.15421 | 82.6% |
| `method="advanced"` | 0.8566 | 0.7840 | 0.15484 | 83.0% |

**Not promoted, either variant.** Both buy a small trajectory-smoothness
gain (+0.9 to +1.3 points over the already-fixed 81.7%) at a real,
consistent cost to AUC and Brier on every cut -- IPL (already the
harder population) hit hardest. `"advanced"` trades even more accuracy
for barely more smoothness than `"basic"`, the opposite of what its own
documentation would suggest. The team-swap bug fix already captured the
overwhelming majority of the real trajectory problem (63.5 of the
~63.5-point total gain); what's left in the remaining ~18% looks more
like genuine model uncertainty in ambiguous match states than a
structural monotonicity defect a blunt 5-feature constraint can cleanly
fix without cost.

## Standing gap this surfaces

No test in this repo exercised `MatchWinnerEngine`'s team-identity
resolution at all -- how this went undetected for the whole session.
Added `tests/test_match_winner_engine.py`: asserts `batting_team`/
`bowling_team` resolve to `context.team1`/`context.team2` for both
`is_chase=True` and `is_chase=False`, so this can't silently regress.
