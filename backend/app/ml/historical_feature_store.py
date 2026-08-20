"""Serving-time computation of the historical/matchup features the
production models were trained on.

``src/train.py`` trained ``models/runs_model.pkl`` and ``models/wkt_model.pkl``
using ``src/features.py``'s leakage-safe rolling stats computed from
``data/real_overs.csv``. This store reproduces that computation at
prediction time so the live/replay serving path stops feeding the model
constant zeros for 17 of its 22 features.

Leakage handling: ``data/real_overs.csv`` carries no match date, and its
``match_id`` is assigned by non-deterministic filesystem glob order in
``src/parse_cricsheet.py`` -- it is not chronological. True temporal
cutoffs are therefore not reconstructable from this file alone. Instead,
when replaying a historical match this store excludes exactly that match's
own rows ("leave-this-match-out") from every aggregate, computed in O(1)
per lookup via precomputed per-entity/per-match partial sums. This
prevents a match from being predicted using its own outcome data, but a
player's historical average may still include matches that occurred after
the one being replayed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.ipl_identities import identity_key


def match_exclude_key(match: Any) -> tuple[str, str, str] | None:
    """Derive the (venue, first-over batsman, first-over bowler) key used to
    exclude a replayed match's own rows from historical lookups.

    Returns ``None`` if the match has no innings/overs/deliveries to key on
    (nothing to exclude, e.g. an abandoned match with no play).
    """

    for innings in match.innings:
        for over in innings.overs:
            if over.deliveries:
                first = over.deliveries[0]
                return (match.info.venue or "", first.batter, first.bowler)
    return None

DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "real_overs.csv"
)

NUMERIC_FIELDS = (
    "venue_avg_score",
    "bat_career_overs_faced",
    "bat_hist_avg_runs_per_over",
    "bat_hist_wicket_rate",
    "bowl_career_overs_bowled",
    "bowl_hist_avg_runs_conceded",
    "bowl_hist_wicket_rate",
    "h2h_overs",
    "h2h_avg_runs",
    "bat_vs_bowltype_avg_runs",
    "bat_vs_bowltype_wicket_rate",
    "bowl_phase_avg_runs",
)
CATEGORICAL_FIELDS = (
    "batsman_style",
    "batsman_class",
    "bowler_type",
    "bowler_quality",
    "pitch_type",
)


class _RunningStat:
    """Sum/count/wicket-sum for one group, with O(1) leave-one-match-out."""

    __slots__ = ("sum_runs", "count", "sum_wkt", "by_match")

    def __init__(self) -> None:
        self.sum_runs = 0.0
        self.count = 0
        self.sum_wkt = 0.0
        self.by_match: dict[int, tuple[float, int, float]] = {}

    def add(self, match_id: int, runs: float, wicket: float) -> None:
        self.sum_runs += runs
        self.count += 1
        self.sum_wkt += wicket
        m_runs, m_count, m_wkt = self.by_match.get(match_id, (0.0, 0, 0.0))
        self.by_match[match_id] = (m_runs + runs, m_count + 1, m_wkt + wicket)

    def excluding(self, match_id: int | None) -> tuple[float, int, float]:
        if match_id is None or match_id not in self.by_match:
            return self.sum_runs, self.count, self.sum_wkt
        m_runs, m_count, m_wkt = self.by_match[match_id]
        return self.sum_runs - m_runs, self.count - m_count, self.sum_wkt - m_wkt


_PROCESS_CACHE: dict[str, dict[str, Any]] = {}

_CACHED_ATTRS = (
    "_global_run_mean",
    "_global_wkt_mean",
    "_bat_stats",
    "_bowl_stats",
    "_h2h_stats",
    "_bat_vs_type_stats",
    "_bowl_phase_stats",
    "_bat_categorical",
    "_bowl_categorical",
    "_venue_avg_score",
    "_venue_pitch_type",
    "_match_key_to_id",
)


class HistoricalFeatureStore:
    """Answers 'what did we know about this batter/bowler/venue historically'.

    Construction is cheap and re-entrant: the expensive per-row aggregation
    is cached process-wide keyed by resolved CSV path, so creating a fresh
    ``HistoricalFeatureStore()`` per engine/innings (as the replay scripts
    do) does not re-scan the dataset each time.
    """

    def __init__(self, csv_path: Path | str | None = None) -> None:
        self._csv_path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        cache_key = str(self._csv_path.resolve())
        cached = _PROCESS_CACHE.get(cache_key)
        if cached is not None:
            for attr in _CACHED_ATTRS:
                setattr(self, attr, cached[attr])
            self._loaded = True
            return

        df = pd.read_csv(self._csv_path)
        df["bat_key_col"] = df["batsman"].map(identity_key)
        df["bowl_key_col"] = df["bowler"].map(identity_key)
        df["venue_key_col"] = df["venue"].map(identity_key)

        self._global_run_mean = float(df["runs_in_over"].mean())
        self._global_wkt_mean = float(df["wicket_in_over"].mean())

        self._bat_stats: dict[str, _RunningStat] = {}
        self._bowl_stats: dict[str, _RunningStat] = {}
        self._h2h_stats: dict[tuple[str, str], _RunningStat] = {}
        self._bat_vs_type_stats: dict[tuple[str, str], _RunningStat] = {}
        self._bowl_phase_stats: dict[tuple[str, str], _RunningStat] = {}

        self._bat_categorical: dict[str, dict[str, str]] = {}
        self._bowl_categorical: dict[str, dict[str, str]] = {}
        self._venue_avg_score: dict[str, float] = {}
        self._venue_pitch_type: dict[str, str] = {}

        # Match key (venue, first-over batsman, first-over bowler) -> match_id,
        # used to identify "this exact historical match" for leave-one-out.
        self._match_key_to_id: dict[tuple[str, str, str], int] = {}

        for match_id, group in df.groupby("match_id"):
            first = group.loc[group["over"].idxmin()]
            key = (
                identity_key(first["venue"]),
                identity_key(first["batsman"]),
                identity_key(first["bowler"]),
            )
            self._match_key_to_id.setdefault(key, int(match_id))

        for row in df.itertuples(index=False):
            match_id = int(row.match_id)
            runs = float(row.runs_in_over)
            wkt = float(row.wicket_in_over)
            bat_key = row.bat_key_col
            bowl_key = row.bowl_key_col
            venue_key = row.venue_key_col
            bowler_type = row.bowler_type

            self._bat_stats.setdefault(bat_key, _RunningStat()).add(
                match_id, runs, wkt
            )
            self._bowl_stats.setdefault(bowl_key, _RunningStat()).add(
                match_id, runs, wkt
            )
            self._h2h_stats.setdefault((bat_key, bowl_key), _RunningStat()).add(
                match_id, runs, wkt
            )
            self._bat_vs_type_stats.setdefault(
                (bat_key, bowler_type), _RunningStat()
            ).add(match_id, runs, wkt)
            self._bowl_phase_stats.setdefault(
                (bowl_key, row.phase), _RunningStat()
            ).add(match_id, runs, wkt)

            self._bat_categorical.setdefault(
                bat_key,
                {
                    "batsman_style": row.batsman_style,
                    "batsman_class": row.batsman_class,
                },
            )
            self._bowl_categorical.setdefault(
                bowl_key,
                {
                    "bowler_type": row.bowler_type,
                    "bowler_quality": row.bowler_quality,
                },
            )

        venue_groups = df.groupby("venue_key_col")
        for venue_key, group in venue_groups:
            self._venue_avg_score[venue_key] = float(
                group["venue_avg_score"].mean()
            )
            self._venue_pitch_type[venue_key] = str(
                group["pitch_type"].mode().iat[0]
            )

        self._loaded = True
        _PROCESS_CACHE[cache_key] = {attr: getattr(self, attr) for attr in _CACHED_ATTRS}

    def lookup(
        self,
        batsman: str,
        bowler: str,
        venue: str,
        phase: str,
        exclude_match_key: tuple[str, str, str] | None = None,
    ) -> dict[str, Any]:
        """Return the 17 historical/matchup features for this over.

        ``exclude_match_key`` is ``(venue, first_over_batsman, first_over_bowler)``
        for the match currently being replayed, so that match's own rows are
        excluded from every aggregate (see module docstring on leakage scope).
        """

        self._ensure_loaded()

        bat_key = identity_key(batsman)
        bowl_key = identity_key(bowler)
        venue_key = identity_key(venue)

        exclude_match_id: int | None = None
        if exclude_match_key is not None:
            lookup_key = tuple(identity_key(part) for part in exclude_match_key)
            exclude_match_id = self._match_key_to_id.get(lookup_key)

        result: dict[str, Any] = {}

        bat_stat = self._bat_stats.get(bat_key)
        bat_runs, bat_count, bat_wkt = (
            bat_stat.excluding(exclude_match_id) if bat_stat else (0.0, 0, 0.0)
        )
        result["bat_career_overs_faced"] = bat_count
        result["bat_hist_avg_runs_per_over"] = (
            bat_runs / bat_count if bat_count else self._global_run_mean
        )
        result["bat_hist_wicket_rate"] = (
            bat_wkt / bat_count if bat_count else self._global_wkt_mean
        )

        bowl_stat = self._bowl_stats.get(bowl_key)
        bowl_runs, bowl_count, bowl_wkt = (
            bowl_stat.excluding(exclude_match_id) if bowl_stat else (0.0, 0, 0.0)
        )
        result["bowl_career_overs_bowled"] = bowl_count
        result["bowl_hist_avg_runs_conceded"] = (
            bowl_runs / bowl_count if bowl_count else self._global_run_mean
        )
        result["bowl_hist_wicket_rate"] = (
            bowl_wkt / bowl_count if bowl_count else self._global_wkt_mean
        )

        h2h_stat = self._h2h_stats.get((bat_key, bowl_key))
        h2h_runs, h2h_count, _h2h_wkt = (
            h2h_stat.excluding(exclude_match_id) if h2h_stat else (0.0, 0, 0.0)
        )
        result["h2h_overs"] = h2h_count
        result["h2h_avg_runs"] = (
            h2h_runs / h2h_count if h2h_count else self._global_run_mean
        )

        bowler_categorical = self._bowl_categorical.get(bowl_key, {})
        bowler_type = bowler_categorical.get("bowler_type", "unknown")
        bat_vs_type_stat = self._bat_vs_type_stats.get((bat_key, bowler_type))
        bvt_runs, bvt_count, bvt_wkt = (
            bat_vs_type_stat.excluding(exclude_match_id)
            if bat_vs_type_stat
            else (0.0, 0, 0.0)
        )
        result["bat_vs_bowltype_avg_runs"] = (
            bvt_runs / bvt_count if bvt_count else self._global_run_mean
        )
        result["bat_vs_bowltype_wicket_rate"] = (
            bvt_wkt / bvt_count if bvt_count else self._global_wkt_mean
        )

        bowl_phase_stat = self._bowl_phase_stats.get((bowl_key, phase))
        bp_runs, bp_count, _bp_wkt = (
            bowl_phase_stat.excluding(exclude_match_id)
            if bowl_phase_stat
            else (0.0, 0, 0.0)
        )
        result["bowl_phase_avg_runs"] = (
            bp_runs / bp_count if bp_count else self._global_run_mean
        )

        batsman_categorical = self._bat_categorical.get(bat_key, {})
        result["batsman_style"] = batsman_categorical.get("batsman_style", "unknown")
        result["batsman_class"] = batsman_categorical.get("batsman_class", "unknown")
        result["bowler_type"] = bowler_categorical.get("bowler_type", "unknown")
        result["bowler_quality"] = bowler_categorical.get("bowler_quality", "unknown")

        result["venue_avg_score"] = self._venue_avg_score.get(
            venue_key, self._global_run_mean * 20
        )
        result["pitch_type"] = self._venue_pitch_type.get(venue_key, "unknown")

        return result
