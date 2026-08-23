# Full repo staleness audit — nothing broken in the live serving path, made the check permanent

Date: 2026-08-23. User asked for a full audit of every file, folder,
link, and path in the codebase for stale/dead/deleted references and
eligible deletions -- then pushed back on a first-pass "looks fine"
answer with a fair concern: how confident is that, and what stops a
future refactor from silently breaking something?

## What was checked

- **Every import in every tracked `.py` file** (292 files): static AST
  analysis (no code execution -- an earlier attempt to `import` every
  script directly to check for errors accidentally *executed* several
  scripts' full pipelines as a side effect, regenerating a 37MB data
  file; killed that approach immediately and switched to pure static
  analysis). Zero broken imports found (one false positive: a
  re-exported name resolves fine at runtime, confirmed directly).
- **Every `data/live/*.json` snapshot** the live feature computers
  depend on (`WicketContract22FeatureComputer`, `MatchWinnerFeatureComputer`):
  all present.
- **The two "current truth" module docstrings** that actually assert
  live status (`app/ml/prediction_engine.py`, `app/ml/match_winner_engine.py`):
  both accurate, correctly name the live model versions.
- **Every `docs/*.md` finding doc** mentioning a superseded model
  version: all are deliberately dated historical records ("vs. live X"
  describing what was true *at that point*), by design -- not staleness.
  The one doc using a different, non-dated convention
  (`candidate_run_range_enriched_v2.md`) was already fixed earlier this
  session.
- **A stray runtime PID file** (`data/live/auto-watcher.pid`, gitignored,
  pointed to a confirmed-dead process): removed. Confirmed the code
  already self-heals this case (`telegram_control.py`'s `_running_pid()`
  catches `ProcessLookupError` and deletes the stale file) -- not a
  functional bug, just tidiness.
- **Every hardcoded file-path string literal in every tracked `.py`
  file** (`check_stale_paths.py`, see below) -- the exhaustive pass done
  after the pushback.

## The exhaustive path-literal scan

Regex-matched every string constant that looks like a relative file
path (`data/...`, `models/...`, ending in a known data/model
extension) across all 292 tracked files, checked each against both the
repo root and `backend/` (scripts are inconsistent about which they
assume as cwd). 59 distinct literals found; **16 don't resolve to a
real file**, 25 total references.

**Triaged all 16, not just eyeballed them:**
- 1 is a deliberate negative test case (`../custom.pkl` in
  `tests/test_model_repository.py` -- testing that a path-traversal
  attempt is correctly rejected; the file is *supposed* to not exist).
- The other 15 are all `models/candidates/**/*.pkl|.cbm|.csv` or
  `data/candidates|reports|replays/**` paths from clearly-superseded
  research lineages (old `announced_bowler_current_spell_v1`/`v2`
  iterations before the live `v3` won; several early CatBoost-based IPL
  engine experiments predating this session's LightGBM-based
  `contract22`/`run_range` architecture entirely). `models/candidates/**/*.pkl`
  etc. are gitignored by explicit project convention -- only the
  lightweight `validation_report.json` evidence trail is committed
  (confirmed via an earlier real commit, `5d3e63f "Commit prior
  candidate-model development history as durable record"` -- this
  project deliberately preserves old research scripts, it doesn't
  delete them). It's expected and normal for their specific
  intermediate artifacts to not be currently materialized locally.

**The distinction that actually matters, verified not assumed**: every
real bug found this session (venue never wired into `MatchContext`,
batting/bowling teams swapped during every chase, hardcoded prediction
metadata) was a case where *wrong data silently reached a live
prediction* -- nothing crashed, it just quietly got worse. Checked
directly whether any of these 16 unresolved literals could do that:

1. **None of the 16 appear anywhere in the live serving path.** Built an
   actual import-graph BFS (not a guess) from every referencing file
   back to `app/ml/prediction_engine.py`, `app/ml/match_winner_engine.py`,
   and `app/live/pipeline.py` -- zero are reachable. All 16 are only
   referenced by standalone `train_*.py`/`diagnose_*.py`/`evaluate_*.py`/
   `audit_*.py`/`compare_*.py`/`backtest_*.py` research scripts, never
   imported by anything a live prediction touches.
2. **They fail loudly, not silently.** Every reference is a
   `pd.read_csv(...)`-style read -- a missing file raises
   `FileNotFoundError` immediately if anyone actually runs one of these
   scripts without first regenerating its upstream dependency. Not a
   silent-corruption risk, an obvious-crash risk.

## Made the check permanent instead of a one-time snapshot

The real gap in the first pass wasn't the finding -- it was that "I
checked today" doesn't answer "what protects this after tomorrow's
refactor." Built `check_stale_paths.py`: reusable, rerunnable, same
exhaustive literal scan plus the import-graph risk classification
(HIGH RISK = reachable from live serving, LOW RISK = standalone
script). Wired it into the permanent test suite
(`tests/test_stale_paths.py`) -- asserts zero HIGH RISK hits. Runs in
under a second, so it costs nothing to keep it in the standing 211-test
suite; it will fail loudly the next time a refactor breaks a live-path
reference, rather than requiring someone to remember to audit again.

## Also fixed along the way

- Deleted the stale, gitignored `auto-watcher.pid` (dead process,
  self-healing code confirmed, purely tidiness).
- Caught and cleaned up my own mistake: an earlier `import`-based
  broken-import check accidentally executed several scripts' full data
  pipelines as a side effect, regenerating a 37MB
  `data/real_overs_styled.csv`. Removed it immediately once noticed;
  confirmed via `git status` that nothing else in the working tree was
  dirtied.

## What this doesn't cover

Regex-based literal matching, no f-string interpolation resolution --
a path built dynamically (`f"{root}/{name}.json"`) wouldn't be caught by
this scan. The `_load(...)`-style snapshot dependencies (already
verified to exist) and the four live model-artifact pointers are
exactly this dynamic-path shape and were checked separately, by hand,
earlier in this same audit -- `check_stale_paths.py` covers the
complementary, more mechanical category (literal strings), not a
replacement for that manual check.

211 tests pass.
