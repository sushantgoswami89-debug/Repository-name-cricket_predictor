"""Live-time feature computation for match_winner_v1, the win-probability
model added 2026-08-23 as a third prediction alongside run-range and
wicket-in-over (not a replacement).

Composes on top of `WicketContract22FeatureComputer` rather than
duplicating its ~60 features from scratch: `match_winner_v1`'s feature
set is exactly that computer's output plus four match-level additions
that were tested and rejected for over-level prediction but validated as
genuinely important for match-level win probability (see
train_match_winner_v1.py's validation report -- venue_recency_par_score
ranked 2nd of 68 features, team H2H 7th):

- `batting_team_venue_context` (home/away/unknown): pure function, no
  snapshot needed (`app.ml.ipl_venues.team_venue_context`).
- `toss_decision`/`batting_team_won_toss`: genuine LIVE input (who won
  the toss and what they decided), not derived from history -- passed
  in by the caller, same as striker/bowler names.
- `h2h_matches_played`/`h2h_batting_team_win_rate_shrunk`: from
  data/live/team_h2h_pairs.json (build_match_winner_live_snapshots.py),
  same shrinkage formula as the training-side
  app/ml/team_h2h_dataset.py (H2H_SHRINKAGE_MATCHES=6.0, neutral 0.5
  prior for an unseen pairing).
- `venue_recency_par_score`/`venue_recency_prior_innings`: from
  data/live/venue_recency_par.json, same shrinkage formula as
  app/ml/venue_recency_dataset.py (VENUE_SHRINKAGE_INNINGS=8.0).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.ml.ipl_venues import normalize_ipl_venue, team_venue_context
from app.ml.team_h2h_dataset import H2H_SHRINKAGE_MATCHES, _pair_key
from app.ml.venue_recency_dataset import DEFAULT_PAR_SCORE, VENUE_SHRINKAGE_INNINGS
from app.ml.wicket_contract22_features import WicketContract22FeatureComputer

UNKNOWN_CATEGORY = "__UNKNOWN__"
H2H_NEUTRAL_PRIOR = 0.5


class MatchWinnerFeatureComputer:
    """Loads the offline snapshots once, computes live feature rows cheaply."""

    def __init__(
        self,
        project_root: Path | None = None,
        *,
        wicket_computer: WicketContract22FeatureComputer | None = None,
    ) -> None:
        root = project_root or Path(__file__).resolve().parents[3]
        self._wicket_computer = wicket_computer or WicketContract22FeatureComputer(root)

        live_dir = root / "data/live"

        def _load(name: str) -> dict:
            path = live_dir / name
            if not path.is_file():
                raise FileNotFoundError(
                    f"Live snapshot missing -- run build_match_winner_live_snapshots.py "
                    f"first: {path}"
                )
            return json.loads(path.read_text(encoding="utf-8"))

        h2h = _load("team_h2h_pairs.json")
        self._pair_matches: dict[str, int] = h2h["pair_matches"]
        self._pair_wins: dict[str, dict[str, int]] = h2h["pair_wins"]

        venue_recency = _load("venue_recency_par.json")
        self._venue_ewma: dict[str, dict[str, float]] = venue_recency["venues"]
        self._global_ewma: dict[str, float] = venue_recency["global"]

    def _team_h2h(self, batting_team: str, bowling_team: str) -> dict[str, float]:
        if not batting_team or not bowling_team:
            return {"matches_played": 0, "win_rate_shrunk": H2H_NEUTRAL_PRIOR}
        key = _pair_key(batting_team, bowling_team)
        prior_matches = self._pair_matches.get(key, 0)
        raw_win_rate = (
            self._pair_wins.get(key, {}).get(batting_team, 0) / prior_matches
            if prior_matches
            else H2H_NEUTRAL_PRIOR
        )
        shrunk = (
            prior_matches * raw_win_rate + H2H_SHRINKAGE_MATCHES * H2H_NEUTRAL_PRIOR
        ) / (prior_matches + H2H_SHRINKAGE_MATCHES)
        return {"matches_played": prior_matches, "win_rate_shrunk": shrunk}

    def _venue_recency(self, venue_name: str) -> dict[str, float]:
        venue = normalize_ipl_venue(venue_name)
        state = self._venue_ewma.get(venue)
        global_weight = self._global_ewma.get("weight", 0.0)
        global_avg = (
            self._global_ewma["value"] / global_weight if global_weight > 0 else DEFAULT_PAR_SCORE
        )
        if state is None or state.get("weight", 0.0) <= 0:
            return {"par_score": global_avg, "prior_innings": 0.0}
        prior_innings = state["weight"]
        venue_avg = state["value"] / state["weight"]
        par_score = (
            prior_innings * venue_avg + VENUE_SHRINKAGE_INNINGS * global_avg
        ) / (prior_innings + VENUE_SHRINKAGE_INNINGS)
        return {"par_score": par_score, "prior_innings": prior_innings}

    def compute(
        self,
        *,
        registry: Mapping[str, str],
        striker_name: str,
        non_striker_name: str = "",
        bowler_name: str,
        venue_name: str,
        phase: str,
        deliveries: Sequence[Any] = (),
        batting_team_players: Sequence[str] = (),
        bowling_team_players: Sequence[str] = (),
        batting_team_name: str = "",
        bowling_team_name: str = "",
        toss_decision: str = "",
        batting_team_won_toss: bool = False,
    ) -> dict[str, Any]:
        result: dict[str, Any] = dict(
            self._wicket_computer.compute(
                registry=registry, striker_name=striker_name, non_striker_name=non_striker_name,
                bowler_name=bowler_name, venue_name=venue_name, phase=phase, deliveries=deliveries,
                batting_team_players=batting_team_players, bowling_team_players=bowling_team_players,
            )
        )
        result["batting_team_venue_context"] = (
            team_venue_context(batting_team_name, venue_name) if batting_team_name else UNKNOWN_CATEGORY
        )
        result["toss_decision"] = toss_decision if toss_decision in ("bat", "field") else "unknown"
        result["batting_team_won_toss"] = int(bool(batting_team_won_toss))

        h2h = self._team_h2h(batting_team_name, bowling_team_name)
        result["h2h_matches_played"] = h2h["matches_played"]
        result["h2h_batting_team_win_rate_shrunk"] = h2h["win_rate_shrunk"]

        venue_recency = self._venue_recency(venue_name)
        result["venue_recency_par_score"] = venue_recency["par_score"]
        result["venue_recency_prior_innings"] = venue_recency["prior_innings"]
        return result
