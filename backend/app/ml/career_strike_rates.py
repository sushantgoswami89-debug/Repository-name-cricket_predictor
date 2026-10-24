"""Career strike rates of T20 batters, overall and per phase, for the over forecast (backend/experiment_pitch_and_career_form.py: the only input that
clearly improved the next-over forecast). One class serves both training and live so the two cannot drift apart:

  * `CareerTable` accumulates runs and balls per Cricsheet player id (overall and powerplay / middle / death), `features()` turns two batters and the
    coming over into the seven model inputs (shrunk toward the typical T20 rate so a debutant is not a 0 or a 300);
  * `build_training_features(root)` walks the men's T20 matches in date order and gives each over the features from matches BEFORE that match;
  * `CareerStrikeRates` is the saved table (models/career_sr_v1) plus the name matching that turns a Cricbuzz name into a Cricsheet player id.
    A name that cannot be matched falls back to the typical rate, so the worst case is the forecast without this input.

Name matching tiers, in order: Cricsheet's own name; full name from data/external/cricsheet_player_styles.csv by first + last name; the live first and last
name both appearing in a player's full name (handles 'Wanindu Hasaranga' = P W H de Silva); surname and first initial. If several players fit, the one with most balls faced is taken
and the match is reported as 'ambiguous'.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = ROOT / "models" / "career_sr_v1"
PRIOR_RPB, PRIOR_BALLS = 1.25, 60.0     # typical T20 runs per ball; career figures shrink toward it until ~60 balls are faced
PHASES = ("powerplay", "middle", "death")
FEATURES = ["striker_career_sr", "partner_career_sr", "striker_phase_sr", "partner_phase_sr", "pair_phase_sr", "striker_career_balls", "partner_career_balls"]


def phase_of(over: int) -> str:
    return "powerplay" if over <= 6 else "middle" if over <= 15 else "death"


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def _first_last(full_name: str) -> str:
    tokens = str(full_name).split()
    return _key(f"{tokens[0]} {tokens[-1]}") if len(tokens) >= 2 else _key(full_name)


def _surname_initial(name: str) -> tuple[str, str] | None:
    tokens = str(name).replace(".", " ").split()
    if len(tokens) < 2:
        return None
    return _key(tokens[-1]), _key(tokens[0])[:1]


class CareerTable:
    """Runs and balls faced per player id: overall and by phase."""

    def __init__(self) -> None:
        self.overall: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.by_phase: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])

    def add(self, player: str, phase: str, runs: int, balls: int) -> None:
        self.overall[player][0] += runs
        self.overall[player][1] += balls
        self.by_phase[(player, phase)][0] += runs
        self.by_phase[(player, phase)][1] += balls

    def batter(self, player: str | None, phase: str) -> tuple[float, float, int]:
        """(career rate, rate in `phase`, career balls); the typical rate for a player never seen."""
        runs, balls = self.overall[player] if player in self.overall else (0, 0)
        career = (runs + PRIOR_RPB * PRIOR_BALLS) / (balls + PRIOR_BALLS)
        p_runs, p_balls = self.by_phase[(player, phase)] if (player, phase) in self.by_phase else (0, 0)
        return career, (p_runs + career * PRIOR_BALLS) / (p_balls + PRIOR_BALLS), balls

    def features(self, striker: str | None, partner: str | None, over: int) -> dict[str, float]:
        phase = phase_of(over)
        s_career, s_phase, s_balls = self.batter(striker, phase)
        p_career, p_phase, p_balls = self.batter(partner, phase)
        return {"striker_career_sr": s_career, "partner_career_sr": p_career, "striker_phase_sr": s_phase, "partner_phase_sr": p_phase,
                "pair_phase_sr": (s_phase + p_phase) / 2, "striker_career_balls": s_balls, "partner_career_balls": p_balls}

    def to_json(self) -> dict:
        return {"overall": {k: v for k, v in self.overall.items()},
                "by_phase": {f"{k[0]}|{k[1]}": v for k, v in self.by_phase.items()}}

    @classmethod
    def from_json(cls, data: dict) -> "CareerTable":
        table = cls()
        for k, v in data["overall"].items():
            table.overall[k] = list(v)
        for k, v in data["by_phase"].items():
            player, phase = k.rsplit("|", 1)
            table.by_phase[(player, phase)] = list(v)
        return table


def _men_t20_matches(root: Path) -> list[tuple[str, str, dict]]:
    matches = []
    for scope in ("ipl", "t20i"):
        for path in sorted((root / "data/raw/cricsheet" / scope).glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            info = raw["info"]
            if str(info.get("gender", "")).lower() != "male" or str(info.get("match_type", "")).upper() not in {"T20", "IT20"}:
                continue
            matches.append((str(info["dates"][0]), path.name, raw))
    matches.sort(key=lambda m: (m[0], m[1]))
    return matches


def _balls_of(raw: dict) -> Iterable[tuple[int, int, str, str, str, int, int]]:
    """(innings, over number, batter id, non-striker id, phase, batter runs, ball faced 0/1) for every delivery."""
    people = raw["info"].get("registry", {}).get("people", {})
    for index, innings in enumerate(raw.get("innings", []), start=1):
        for over in innings.get("overs", []):
            over_no = int(over["over"]) + 1
            for d in over.get("deliveries", []):
                extras = d.get("extras", {})
                yield (index, over_no, people.get(d.get("batter"), d.get("batter")), people.get(d.get("non_striker"), d.get("non_striker")),
                       phase_of(over_no), int(d["runs"]["batter"]), 0 if "wides" in extras else 1)


def build_final_table(root: Path = ROOT) -> tuple[CareerTable, str, int]:
    table, last, count = CareerTable(), "", 0
    for date, _, raw in _men_t20_matches(root):
        for _, _, batter, _, phase, runs, faced in _balls_of(raw):
            table.add(batter, phase, runs, faced)
        last, count = date, count + 1
    return table, last, count


def build_training_features(root: Path = ROOT) -> pd.DataFrame:
    """One row per (source_file, innings, over) with the seven inputs computed from matches played BEFORE that match."""
    table, rows = CareerTable(), []
    for date, name, raw in _men_t20_matches(root):
        people = raw["info"].get("registry", {}).get("people", {})
        for index, innings in enumerate(raw.get("innings", []), start=1):
            for over in innings.get("overs", []):
                deliveries = over.get("deliveries", [])
                if not deliveries:
                    continue
                first = deliveries[0]
                features = table.features(people.get(first.get("batter"), first.get("batter")), people.get(first.get("non_striker"), first.get("non_striker")),
                                          int(over["over"]) + 1)
                rows.append({"source_file": name, "innings": index, "over": int(over["over"]) + 1, **features})
        for _, _, batter, _, phase, runs, faced in _balls_of(raw):    # the match joins the history only after all its rows are made
            table.add(batter, phase, runs, faced)
    return pd.DataFrame(rows)


def save(directory: Path = DEFAULT_DIR, root: Path = ROOT) -> dict:
    table, as_of, matches = build_final_table(root)
    native: dict[str, str] = {}
    for _, _, raw in _men_t20_matches(root):
        for person, pid in raw["info"].get("registry", {}).get("people", {}).items():
            native[person] = pid
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"as_of": as_of, "matches": matches, "players": len(table.overall), **table.to_json(), "native": native}
    path = directory / "career_sr.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = {"version": "career_sr_v1", "as_of": as_of, "matches": matches, "players": len(table.overall),
                "prior_runs_per_ball": PRIOR_RPB, "prior_balls": PRIOR_BALLS, "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}
    (directory / "CAREER_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class CareerStrikeRates:
    def __init__(self, directory: Path = DEFAULT_DIR, styles_csv: Path | None = None) -> None:
        manifest = json.loads((directory / "CAREER_MANIFEST.json").read_text(encoding="utf-8"))
        for name, digest in manifest["sha256"].items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"career strike rate artifact {name} does not match its manifest")
        data = json.loads((directory / "career_sr.json").read_text(encoding="utf-8"))
        self.as_of: str = data["as_of"]
        self.table = CareerTable.from_json(data)
        self._native = {_key(name): pid for name, pid in data["native"].items()}
        self._by_surname: dict[tuple[str, str], list[str]] = defaultdict(list)
        for name, pid in data["native"].items():
            parts = _surname_initial(name)
            if parts:
                self._by_surname[parts].append(pid)
        self._full: dict[str, str] = {}
        self._tokens: dict[str, set[str]] = defaultdict(set)
        csv_path = styles_csv or ROOT / "data" / "external" / "cricsheet_player_styles.csv"
        known = set(self.table.overall)
        try:
            with csv_path.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    pid = row.get("cricsheet_id") or ""
                    if pid in known and row.get("full_name"):
                        self._full.setdefault(_first_last(row["full_name"]), pid)
                        for token in _key(row["full_name"]).split():
                            self._tokens[token].add(pid)
        except OSError:
            pass

    @classmethod
    def try_load(cls, directory: Path = DEFAULT_DIR) -> "CareerStrikeRates | None":
        try:
            return cls(directory)
        except Exception:
            return None

    def resolve(self, name: str | None) -> tuple[str | None, str]:
        """(player id or None, tier used): 'native', 'full name', 'surname+initial', 'ambiguous' (most balls taken) or 'unmatched'."""
        if not name:
            return None, "unmatched"
        pid = self._native.get(_key(name))
        if pid:
            return pid, "native"
        pid = self._full.get(_first_last(name))
        if pid:
            return pid, "full name"
        tokens = _key(name).split()
        if len(tokens) >= 2:
            fits = sorted(self._tokens.get(tokens[0], set()) & self._tokens.get(tokens[-1], set()))
            if len(fits) == 1:
                return fits[0], "given name"
            if len(fits) > 1:
                return max(fits, key=lambda c: self.table.overall[c][1] if c in self.table.overall else 0), "ambiguous"
        parts = _surname_initial(name)
        candidates = list(dict.fromkeys(self._by_surname.get(parts, []))) if parts else []
        if len(candidates) == 1:
            return candidates[0], "surname+initial"
        if len(candidates) > 1:
            return max(candidates, key=lambda c: self.table.overall[c][1] if c in self.table.overall else 0), "ambiguous"
        return None, "unmatched"

    def features(self, striker: str | None, partner: str | None, over: int) -> tuple[dict[str, float], dict[str, str]]:
        s_id, s_tier = self.resolve(striker)
        p_id, p_tier = self.resolve(partner)
        return self.table.features(s_id, p_id, over), {"striker": s_tier, "partner": p_tier}
