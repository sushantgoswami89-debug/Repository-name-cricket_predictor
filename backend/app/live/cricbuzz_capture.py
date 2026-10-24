"""Save everything Cricbuzz publishes about a match, and extract its own win-probability curve.

A Cricbuzz match has eleven pages (inspect_cricbuzz_pages.py lists what each carries). The live driver reads
the main page every few seconds; this module captures the rest once, at the end of the match (or on demand for
any finished match, which still works days later):

  scorecard  full batting/bowling cards, extras, fall of wickets, partnerships, powerplay, result header
  graphs     Cricbuzz's own win probability after every over of both innings (+ runs, run rate, wickets)
  over-by-over, full-commentary, highlights, squads (playing XI, bench, roles), facts, forecast, news, videos

Raw pages go to data/raw/cricbuzz-pages/<id>/final/<page>.html.gz; the win curve to data/live/cb-win-curve-<id>.json.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import requests

from app.live.cricbuzz_reader import _extract_flight_chunks
from app.live.cricbuzz_scorecard import parse_scorecard
from app.live.prematch_model import ROOT

PAGES = (
    "live-cricket-scorecard", "live-cricket-graphs", "live-cricket-over-by-over", "live-cricket-full-commentary",
    "cricket-match-highlights", "cricket-match-squads", "cricket-match-facts", "cricket-match-forecast",
)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CricketBaba/3.0)", "Cache-Control": "no-cache"}
_DECODER = json.JSONDecoder()


def fetch_page(kind: str, match_id: str, timeout: float = 20.0) -> str | None:
    try:
        response = requests.get(f"https://www.cricbuzz.com/{kind}/{match_id}/x", headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        return None



def per_over_from_graphs(html: str) -> dict[tuple[int, int], dict] | None:
    """{(innings, over): {"runs": n, "wicket": bool}} for EVERY over of the match from the graphs page's runsPerOverChartData
    (team1 batted first = innings 1, team2 second). The main page keeps only the last 2-3 overs; this page keeps all of them."""
    for chunk in _extract_flight_chunks(html):
        at = chunk.find('"graphsData":')
        if at == -1:
            continue
        try:
            graphs, _ = _DECODER.raw_decode(chunk[at + len('"graphsData":'):].lstrip())
        except json.JSONDecodeError:
            continue
        rows = (graphs.get("data") or {}).get("runsPerOverChartData")
        if not isinstance(rows, list):
            continue
        table: dict[tuple[int, int], dict] = {}
        for row in rows:
            over = row.get("over")
            if not isinstance(over, int):
                continue
            for innings, team_key, wicket_key in ((1, "team1", "isTeam1Wicket"), (2, "team2", "isTeam2Wicket")):
                if row.get(team_key) is not None:
                    table[(innings, over)] = {"runs": row[team_key], "wicket": bool(row.get(wicket_key))}
        return table
    return None


def parse_win_curve(html: str) -> dict | None:
    """{"teams": {1: name, 2: name}, "innings": {1: team, 2: team}, "points": {innings: [{over, <team>: pct}]}}
    from the graphs page. `over` n = the state after n overs of that innings; over 0 of innings 1 is 50/50."""
    for chunk in _extract_flight_chunks(html):
        at = chunk.find('"graphsData":')
        if at == -1:
            continue
        try:
            graphs, _ = _DECODER.raw_decode(chunk[at + len('"graphsData":'):].lstrip())
        except json.JSONDecodeError:
            continue
        data = graphs.get("data") or {}
        innings = {str(i["inningsId"]): i["teamName"] for i in data.get("innings", [])}
        points: dict[str, list[dict]] = {}
        for innings_id, series in (data.get("winProbabilityHorizontalChartData") or {}).items():
            points[innings_id] = [
                {"over": p["over"], p["team1"]["name"]: p["team1"]["percent"], p["team2"]["name"]: p["team2"]["percent"]}
                for p in series
            ]
        return {"innings": innings, "points": points, "team1": graphs.get("team1Name"), "team2": graphs.get("team2Name")}
    return None


def capture_match(match_id: str, root: Path | None = None) -> dict:
    """Fetch and store every page; returns {"saved": [...], "winner": name|None, "curve_points": n}."""
    root = root or ROOT
    folder = root / "data" / "raw" / "cricbuzz-pages" / match_id / "final"
    folder.mkdir(parents=True, exist_ok=True)
    saved, winner, result_text, curve = [], None, None, None
    for kind in PAGES:
        html = fetch_page(kind, match_id)
        if html is None:
            continue
        with gzip.open(folder / f"{kind}.html.gz", "wt", encoding="utf-8") as handle:
            handle.write(html)
        saved.append(kind)
        if kind == "live-cricket-scorecard":
            card = parse_scorecard(html)
            winner = (card or {}).get("header", {}).get("winner")
            result_text = (card or {}).get("header", {}).get("status")
        elif kind == "live-cricket-graphs":
            curve = parse_win_curve(html)
    if curve:
        curve.update(match_id=match_id, winner=winner, result=result_text)
        out = root / "data" / "live" / f"cb-win-curve-{match_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(curve), encoding="utf-8")
    return {"saved": saved, "winner": winner, "curve_points": sum(len(v) for v in (curve or {}).get("points", {}).values())}
