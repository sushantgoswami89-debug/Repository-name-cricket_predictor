# Cricket Predictor CTO Handover

Date: 2026-07-24  
Gate evaluated: 15:30 IST  
Decision: **IPL promotion deliverable missed—CTO role revoked**

## Executive status

The agreed 14:00 IST promotion gate passed before the scheduled automation
woke. No callable, calibrated, replay-tested IPL wicket predictor was delivered
by the gate. Existing IPL work is experimental and must not be promoted.

The strongest observed untouched-2026 wicket diagnostic is from IPL v10:

- Brier score: `0.1948232943`
- Prior Brier score: `0.2009095634`
- AUC: `0.5999477166`
- Integration recommended: `false`

This is a modest improvement over the prior, not sufficient production
evidence. Calibration by probability band and innings phase, a stable callable
runtime package, state-sensitivity examples, leakage audit, and replay/live
integration proof remain incomplete.

## IPL candidate evidence

| Candidate | Wicket result | Promotion result |
|---|---:|---|
| v5 phase intelligence | Brier `0.1951947457` | Rejected |
| v6 phase transition | Brier `0.1955988783` | Rejected |
| v7 home/phase transition | Brier `0.1954614834` | Rejected |
| v8 ball simulator | Brier `0.2001902792` | Rejected; latency gate also failed |
| v9 joint event | Brier `0.2420528497` | Rejected; accuracy and latency failed |
| v10 shock gate, 2026 wicket diagnostic | Brier `0.1948232943`, AUC `0.5999477166` | Rejected; overall gates failed |

Primary evidence:

- `models/candidates/ipl_engine_v5_phase_intelligence/validation_report.json`
- `models/candidates/ipl_engine_v6_phase_transition_scoring/validation_report.json`
- `models/candidates/ipl_engine_v7_home_phase_transition/validation_report.json`
- `models/candidates/ipl_engine_v8_ball_simulator/validation_report.json`
- `models/candidates/ipl_engine_v9_joint_event/validation_report.json`
- `models/candidates/ipl_v10_shock_gate/validation_report.json`

Training and evaluation entry points:

- `backend/train_ipl_phase_intelligence_v5.py`
- `backend/train_ipl_ball_simulator_v8.py`
- `backend/train_ipl_joint_event_v9.py`
- `backend/train_ipl_shock_gate_v10.py`
- `backend/backtest_ipl_v5_selective.py`

These files are currently untracked and therefore are not durable repository
history.

## Delivered and committed work

Branch: `codex/live-pipeline-hardening`

- `3046205 Harden verified live prediction pipeline`
  - TOI ingestion and delivery verification
  - Candidate v3 17-feature live contract
  - chase target and recent legal-ball window
  - innings-transition handling
  - Telegram publishing path and timing/retry work
- `496e656 Add fail-closed CricketData summary adapter`
  - secure local credential loading
  - match discovery and aggregate-score parsing
  - explicit refusal to feed Candidate v3 without deliveries

At the last completed checkpoint, the backend suite passed 106 tests. Model
loading emitted a version warning because artifacts were trained with
scikit-learn 1.8 and the runtime used 1.9.

## Known defects and limitations

1. No IPL wicket candidate meets the full promotion contract.
2. The repeated `50%` live wicket display has not been isolated conclusively
   between model output, fallback/default behavior, and presentation rounding.
3. CricketData's inexpensive endpoints do not supply verified ball-by-ball
   deliveries. Its scorecard request used roughly 10 of the 100 daily hits and
   still did not satisfy Candidate v3.
4. TOI commentary can lag the scorecard by approximately one delivery.
5. No complete real-match proof exists for:
   verified over -> prediction -> exactly one Telegram message.
6. The worktree contains many modified and untracked user files. Do not bulk
   stage, reset, clean, or overwrite them.
7. API and Telegram credentials live in ignored backend configuration files.
   Never print, commit, or copy their values into a handover.

## Required next actions

### 1. Re-establish a wicket-model promotion contract

Before training, freeze these pass/fail checks:

- untouched-season Brier improvement over a documented prior;
- AUC and reliability/calibration table by probability band;
- calibration and discrimination by powerplay, middle, and death overs;
- temporal and match-level leakage audit;
- materially different probabilities across representative match states;
- inference latency below 50 ms;
- stable callable predictor with artifact schema/version metadata;
- replay and live-pipeline integration tests.

Reject the candidate if any mandatory gate fails. Do not couple wicket
promotion to the run-range model's accuracy gate unless product requirements
explicitly require joint promotion.

### 2. Diagnose the `50%` output before retraining

Trace one prediction end to end:

1. raw estimator probability;
2. calibration transform;
3. fallback/default selection;
4. serialization;
5. Telegram formatting and rounding.

Add tests with at least three distinct match states and assert that raw and
displayed probabilities are not all identical unless the estimator genuinely
returns identical values.

### 3. Package only after the model passes

Create a production-facing IPL wicket predictor that:

- loads a versioned artifact and exact feature manifest;
- refuses missing or stale features;
- returns probability plus model/version metadata;
- exposes calibration provenance;
- has unit, artifact-load, replay, and integration tests.

### 4. Complete live-source selection

Test free providers with minimal calls for genuine delivery objects, feed
freshness, legal-ball/extras semantics, daily allowance, and full-match cost.
Do not confuse HTTP latency with score freshness. Retain TOI as a fallback only
with delivery verification. Do not spend more CricketData hits without a
specific evidence need.

### 5. Complete match-ready safety

- 100-hit quota guard, reserve, cache, and duplicate-poll protection;
- verified over completion with 3-minute normal check and 3.5-minute retry;
- exactly-once Telegram idempotency;
- full replay/shadow test with measured end-to-end delay;
- match-day operating and failure-recovery procedure.

## Safe continuation commands

Run from the repository root with the existing virtual environment:

```sh
.venv/bin/python -m pytest backend/tests -q
git status --short
git log --oneline -10
```

Do not use destructive Git cleanup commands. Stage only explicitly reviewed
files.
