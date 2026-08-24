# LSTM over raw ball-by-ball sequences for wicket prediction: real architecture test, negative, not promoted

Date: 2026-08-24. Direct follow-up to the 2026-08-23 session handoff: a
long streak of rejected single-feature GBM additions (toss, home
advantage, team H2H, venue recency, monotone constraints, the wicket side
of the matchup-score investigation -- six-plus real, honest rejections)
plus a diffuse feature-importance check (32-33 of 61-63 features needed
for 80% of importance on the live model) supported a real conclusion:
further single-feature additions to `contract22_wicket_v16_team_composition`
were hitting a genuine information ceiling for the current flat/single-row
GBM architecture, not "haven't found the right feature yet." The proposed
next lever was the untouched one -- model *structure* itself, specifically
a sequential model (LSTM) over ball-by-ball sequences, motivated by real
production research (Gurpinar-Morgan et al., "You Cannot Do That Ben
Stokes," arxiv.org/pdf/2102.01952). This tests that lever directly.

Installed PyTorch for the first time in this project (`.venv`, previously
LightGBM/sklearn-only).

## Construction

Built `app/ml/wicket_sequence_dataset.py`: parses raw Cricsheet JSON
directly (ipl + t20i), producing one sequence per
`(source_file, innings, over)` key -- every legal/illegal delivery bowled
strictly BEFORE that over starts, capped at the most recent 120 balls.
Deliberately minimal per-ball features (9 total: batter runs, extras,
total runs, is-wicket, is-legal, is-boundary, is-dot, normalized over
number, normalized ball-in-over) -- no player identity, no engineered
rolling windows. **Exact leakage-safe parity check**: 246,846 sequences
built independently from raw JSON matched
`data/candidates/v3/verified_training_overs.csv`'s row count exactly
(246,846), confirming the same target population before trusting anything
built on it.

Same `wicket_in_over` label, same eligible-file filter
(`male_source_files`), same train(<=2023-12-31)/calibration(2024)/
holdout(2025+) split, same IPL/T20I split convention as every other
wicket candidate in this project.

## Three real tests, same split/eval harness

**v1 -- pure sequence, no context** (`train_wicket_lstm_v1.py`): single-layer
LSTM (hidden=64) over the raw 9-feature ball sequence only, no match-state
context at all, no bowler identity. Result: **AUC 0.588 blended holdout**
(0.585 IPL / 0.588 T20I) -- meaningfully below the live model's 0.613
known / 0.611 unknown blended. Expected in isolation: the live GBM's
dominant features are match-state (wickets in hand, required run rate,
phase) this LSTM never saw and would have to re-derive purely from
counting deliveries -- a much harder implicit task than being handed the
aggregates directly.

**v2a -- context-only MLP, no sequence** (`train_wicket_lstm_v2_context.py`,
control arm): small MLP over just the 16 `BASE_FEATURES` match-state
columns (score, wickets in hand, run rate, required rate, phase one-hot,
etc.) already in `verified_training_overs.csv`, zero sequence input.
Result: AUC 0.606 blended (0.594 IPL / 0.608 T20I) -- as expected, below
the full 60-feature live GBM (which also has player identity, recency,
partnership, venue, h2h), but a genuine improvement over v1's
sequence-only result.

**v2b -- hybrid (LSTM sequence + same context)**: v1's LSTM branch
concatenated with the same 19-dim context vector (16 numeric + phase
one-hot) before the classifier head, isolating whether raw-sequence order
adds anything ON TOP of context, not just whether a bigger model wins.
Result: AUC 0.604 blended (0.595 IPL / 0.605 T20I) -- **slightly worse than
context-alone (v2a)**, not better. The sequence branch contributed nothing
net-positive; if anything it added noise.

## Verdict

**Not promoted, closed.** Two independent formulations (sequence alone,
sequence+context together) both failed to show any positive contribution
from raw ball-sequence order -- the hybrid losing to its own context-only
control is the clean tell, not just "the LSTM underperforms the GBM"
(expected, given the huge feature-richness gap) but specifically "adding
sequence information never helped, in either formulation tested."

**Likely root cause, consistent with the earlier matchup-score
investigation's conclusion**: this project's available ball-by-ball
history is coarse outcome data (runs scored, wicket or not, boundary or
not) -- not the granular line/length/shot-type/ball-tracking detail the
Gurpinar-Morgan production system was built on. The mechanism the paper
validates (personalized sequential modeling beats flat averages) is real,
but likely needs richer per-ball input than "did 4 runs happen on ball N"
to have something worth learning temporal structure over. Same shape as
the four-attempt matchup-score conclusion: the binding constraint here
looks like **data granularity, not modeling approach**.

**Not fully closed on the general "does architecture matter" question**:
this test used a small, feature-light neural net (16-19 inputs) against a
60-feature GBM -- not a controlled architecture-only comparison (same
full feature set, GBM vs. neural net). If that specific question becomes
a priority later, it would need player-identity embeddings and the rest
of the live feature set ported into a neural architecture, a materially
larger build than this test. Not attempted here; flagged for a future
session if worth prioritizing.

No code/model changes to the live pipeline -- `contract22_wicket_v16_team_composition`
remains live, unchanged. New infrastructure kept (not deleted, per project
convention): `app/ml/wicket_sequence_dataset.py`,
`backend/train_wicket_lstm_v1.py`, `backend/train_wicket_lstm_v2_context.py`,
candidate artifacts under `models/candidates/wicket_lstm_v1/` and
`models/candidates/wicket_lstm_v2_context/` (both gitignored `.pt` weights
+ tracked `validation_report.json`). PyTorch now a real project dependency
(`.venv/lib/python3.12/site-packages/torch`).
