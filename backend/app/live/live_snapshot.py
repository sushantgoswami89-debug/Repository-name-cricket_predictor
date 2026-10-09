"""Source-agnostic live-match snapshot contract.

Moved out of app/live/toi_reader.py on 2026-09-18, together with renaming
ToiSnapshot/ToiDelivery/ToiPlayer to LiveSnapshot/LiveDelivery/LivePlayer.
Both app/live/toi_reader.py (Times of India) and
app/live/cricbuzz_reader.py (Cricbuzz, a fallback source added
2026-09-15 after TOI failed to publish a real live match at all -- see
docs/live_field_test_2026-09-15.md) build this exact shape, so
VerifiedLivePredictionPipeline never needs to know which feed a snapshot
actually came from.

Why the rename: the old TOI-branded names were a real trap living inside
a file named toi_reader.py. It is easy to read `ToiSnapshot` inside
cricbuzz_reader.py, assume the data or its guarantees are TOI's, and make
a wrong edit on that assumption -- e.g. assuming a field TOI always
populates (a full from-ball-1 history, say) is equally reliable when the
snapshot actually came from Cricbuzz's rolling window. There is no
TOI-specific meaning left in these types themselves; TOI- and
Cricbuzz-specific behavior lives entirely in each reader's own module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class LiveDelivery:
    match_id: str
    innings: int
    over: int
    ball: int
    bowler: str
    striker: str
    total_runs: int
    batter_runs: int
    extras: dict[str, int]
    wicket_kind: str | None
    feed_total: int
    feed_wickets: int
    timestamp_ms: int
    commentary: str

    @property
    def key(self) -> tuple[int, int, int]:
        return self.innings, self.over, self.ball

    @property
    def is_legal(self) -> bool:
        return "wides" not in self.extras and "noballs" not in self.extras


@dataclass(frozen=True, slots=True)
class LivePlayer:
    """One player's entry from a team's confirmed/probable lineup.

    ``role`` is the source feed's own classification (e.g.
    ``Batter``/``All-Rounder``/``Wicket Keeper``/``Bowler``), and
    ``position`` is the batting-order slot (1-11), best-effort per
    source -- both already computed upstream, not derived here.
    ``confirm_xi`` is False for a probable/predicted lineup shown before
    the official XI is confirmed (typically around toss time), or for a
    nominated bench/impact substitute once play has started.
    """

    position: int
    name: str
    role: str
    confirm_xi: bool


@dataclass(frozen=True, slots=True)
class LiveSnapshot:
    match_id: str
    match_format: str
    batting_team: str
    bowling_team: str
    innings: int
    score: int
    wickets: int
    overs: str
    is_live: bool
    deliveries: tuple[LiveDelivery, ...]
    target: int = 0
    scheduled_overs: int = 0
    innings_complete: bool = False
    match_complete: bool = False
    is_super_over: bool = False
    competition: str = ""
    announced_bowler: str = ""
    announced_bowler_over: int = 0
    # Best-effort enrichment from the same raw payload already fetched for
    # match state -- never required for correctness, so a missing or
    # malformed block degrades to the empty default rather than raising.
    team_players: dict[str, tuple[LivePlayer, ...]] = field(default_factory=dict)
    venue_name: str = ""
    toss_won_by: str = ""
    toss_decision: str = ""
    pitch_type: str = ""
    pitch_surface: str = ""
    weather_condition: str = ""
    weather_humidity_pct: float | None = None
    weather_temperature_c: float | None = None
    weather_wind_speed_ms: float | None = None
    # The source's own live scoreboard ("miniscore"): authoritative score,
    # overs, striker/non-striker, current bowler, partnership. Used to
    # cross-check the state we rebuild from ball-by-ball commentary.
    board: dict | None = None
    raw_html: str = field(default="", repr=False, compare=False)
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
