"""Cricbuzz's dedicated scorecard page (/live-cricket-scorecard/<id>/x): full batting and bowling
cards, extras, fall of wickets and the match header, as an independent check on the state we rebuild
from ball-by-ball commentary (the main match page carries only a link to it).

Fields confirmed on a finished match, 2026-10-10 (India v West Indies, 151576): per batter runs/balls/
fours/sixes/outDesc/wicketCode; per bowler overs/maidens/runs/wickets/no_balls/wides; extras by kind;
wicketsData (dismissed batter, over, team score at the fall); matchHeader state/status/complete/result.
"""

from __future__ import annotations

import json
from typing import Any

import requests

from app.live.cricbuzz_reader import _extract_flight_chunks

_DECODER = json.JSONDecoder()


def _decode_after(chunks: list[str], key: str) -> Any:
    marker = f'"{key}":'
    for chunk in chunks:
        at = chunk.find(marker)
        if at == -1:
            continue
        try:
            value, _ = _DECODER.raw_decode(chunk[at + len(marker):].lstrip())
            return value
        except json.JSONDecodeError:
            continue
    return None


def parse_scorecard(html: str) -> dict | None:
    """{"header": {...}, "innings": [{"innings", "team", "runs", "wickets", "overs", "batting", "bowling",
    "extras", "fall"}]} or None when the page has no scorecard data."""
    chunks = _extract_flight_chunks(html)
    raw = _decode_after(chunks, "scoreCard")
    if not isinstance(raw, list) or not raw:
        return None
    innings = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        team = item.get("batTeamDetails") or {}
        bowl = item.get("bowlTeamDetails") or {}
        score = item.get("scoreDetails") or {}
        innings.append(
            {
                "innings": item.get("inningsId"),
                "team": team.get("batTeamName"),
                "runs": score.get("runs"),
                "wickets": score.get("wickets"),
                "overs": score.get("overs"),
                "batting": [
                    {
                        "name": b.get("batName"), "runs": b.get("runs"), "balls": b.get("balls"),
                        "fours": b.get("fours"), "sixes": b.get("sixes"),
                        "out": b.get("outDesc") or "", "wicket_code": b.get("wicketCode") or "",
                    }
                    for b in (team.get("batsmenData") or {}).values()
                ],
                "bowling": [
                    {
                        "name": b.get("bowlName"), "overs": b.get("overs"), "maidens": b.get("maidens"),
                        "runs": b.get("runs"), "wickets": b.get("wickets"),
                        "no_balls": b.get("no_balls"), "wides": b.get("wides"),
                    }
                    for b in (bowl.get("bowlersData") or {}).values()
                ],
                "extras": item.get("extrasData") or {},
                "fall": [
                    {"batter": w.get("batName"), "over": w.get("wktOver"), "runs": w.get("wktRuns"), "number": w.get("wktNbr")}
                    for w in (item.get("wicketsData") or {}).values()
                ],
            }
        )
    header = _decode_after(chunks, "matchHeader") or {}
    result = header.get("result") if isinstance(header.get("result"), dict) else {}
    return {
        "header": {
            "state": header.get("state"), "status": header.get("status"), "complete": header.get("complete"),
            "result_type": result.get("resultType"), "winner": result.get("winningTeam"),
        },
        "innings": innings,
    }


def fetch_scorecard(match_id: str, session: requests.Session | None = None, timeout: float = 10.0) -> dict | None:
    """Fetch and parse the scorecard page; any failure returns None (it is a cross-check, never required)."""
    try:
        http = session or requests
        response = http.get(
            f"https://www.cricbuzz.com/live-cricket-scorecard/{match_id}/x",
            headers={"User-Agent": "Mozilla/5.0 (compatible; CricketBaba/3.0)", "Cache-Control": "no-cache"},
            timeout=timeout,
        )
        response.raise_for_status()
        return parse_scorecard(response.text)
    except (requests.RequestException, ValueError):
        return None


def compare_with_deliveries(card: dict, deliveries, innings: int) -> list[str]:
    """Differences between the scorecard and the deliveries we rebuilt for one innings (empty = agree).
    Totals are always checked; per-batter/bowler figures and the fall of wickets only when we saw the
    innings from its first ball, because a mid-innings attach cannot reproduce them."""
    ours = sorted((d for d in deliveries if d.innings == innings), key=lambda d: d.key)
    theirs = next((i for i in card.get("innings", []) if i.get("innings") == innings), None)
    if not ours or theirs is None:
        return []
    problems = []
    last = ours[-1]
    if theirs.get("runs") is not None and (last.feed_total, last.feed_wickets) != (theirs["runs"], theirs["wickets"]):
        problems.append(f"total ours {last.feed_total}/{last.feed_wickets} vs card {theirs['runs']}/{theirs['wickets']}")
    first = ours[0]
    from_start = first.over == 1 and first.ball == 1 and first.feed_total - first.total_runs == 0
    if not from_start:
        return problems
    runs_by_batter: dict[str, int] = {}
    for d in ours:
        runs_by_batter[d.striker] = runs_by_batter.get(d.striker, 0) + d.batter_runs
    for b in theirs["batting"]:
        if b["name"] in runs_by_batter and b["runs"] is not None and runs_by_batter[b["name"]] != b["runs"]:
            problems.append(f"{b['name']} ours {runs_by_batter[b['name']]} vs card {b['runs']}")
    out_ours = [d.dismissed or d.striker for d in ours if d.wicket_kind]
    out_card = [w["batter"] for w in sorted(theirs["fall"], key=lambda w: w.get("number") or 0)]
    if out_ours != out_card[: len(out_ours)] and len(out_ours) == len(out_card):
        problems.append(f"fall of wickets ours {out_ours} vs card {out_card}")
    return problems
