# CricketBaba Match Intelligence Principles

## Phase-Relative Event Value

> An event has a fixed scoreboard value, but its match impact depends on when
> and under what pressure it happens.

CricketBaba must not treat the same delivery outcome as equally informative in
every phase.

### Powerplay

- An early wicket has a large future cost because it exposes the middle order
  and reduces scoring potential across many remaining overs.
- A boundary exploits the fielding restrictions and contributes to Powerplay
  momentum.
- A single has lower opportunity value when boundary-scoring conditions are at
  their most favourable, although strike rotation can still matter by matchup.

### Middle overs

- Singles and twos have greater strategic value because rotation prevents dot-
  ball pressure and disrupts bowler control.
- A boundary can break a containment pattern and should be evaluated relative
  to the lower phase scoring expectation.
- A wicket can interrupt rebuilding or expose a weaker batter, with its cost
  depending on wickets in hand and lineup depth.

### Death overs

- Boundaries have high immediate value because few deliveries remain and the
  opportunity cannot be recovered later.
- Singles often carry a high opportunity cost unless they preserve strike for
  the stronger batter.
- A wicket has less time to damage the full-innings trajectory than an early
  wicket, but losing a set finisher can still sharply reduce the remaining
  scoring expectation.

## Implementation Rule

CricketBaba should estimate event impact as the change in expected future match
state, not through fixed hand-written multipliers:

```text
event_impact =
    expected_future_value_before_event
    - expected_future_value_after_event
```

The learned impact should account for:

- format and phase;
- innings progress and balls remaining;
- score, wickets in hand, and chase pressure;
- batter quality, bowler quality, and matchup;
- partnership state and lineup depth;
- field restrictions and recent momentum.

This principle will guide candidate v3 features such as phase-adjusted boundary
impact, dot-ball pressure, strike-rotation value, and wicket cost.
