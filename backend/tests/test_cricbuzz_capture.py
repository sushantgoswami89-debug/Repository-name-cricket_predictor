"""Cricbuzz's own win-probability curve and the capture of a match's pages."""

import gzip
import json

from app.live import cricbuzz_capture as capture
from app.live.cricbuzz_capture import capture_match as REAL_CAPTURE  # before conftest patches it
from app.live.cricbuzz_capture import parse_win_curve


def _page(payload: dict) -> str:
    text = ",".join(f'"{k}":{json.dumps(v)}' for k, v in payload.items())
    return '<script>self.__next_f.push([1,' + json.dumps(text) + '])</script>'


def _point(over, t1, t2):
    return {"matchId": 1, "over": over,
            "team1": {"id": 2, "percent": t1, "name": "India", "shortName": "IND"},
            "team2": {"id": 10, "percent": t2, "name": "West Indies", "shortName": "WI"}}


GRAPHS = {"graphsData": {
    "team1Name": "IND", "team2Name": "WI",
    "data": {
        "innings": [{"inningsId": 1, "teamName": "India"}, {"inningsId": 2, "teamName": "West Indies"}],
        "winProbabilityHorizontalChartData": {"1": [_point(1, 60, 40), _point(2, 62, 38)], "2": [_point(1, 98, 2)]},
    },
}}


def test_win_curve_is_read_per_innings_and_over():
    curve = parse_win_curve(_page(GRAPHS))
    assert curve["innings"] == {"1": "India", "2": "West Indies"}
    assert curve["points"]["1"][0] == {"over": 1, "India": 60, "West Indies": 40}
    assert curve["points"]["2"] == [{"over": 1, "India": 98, "West Indies": 2}]
    assert parse_win_curve("<html>nothing</html>") is None


def test_capture_saves_every_page_and_the_curve(tmp_path, monkeypatch):
    pages = {kind: "<html>" + kind + "</html>" for kind in capture.PAGES}
    pages["live-cricket-graphs"] = _page(GRAPHS)
    monkeypatch.setattr(capture, "fetch_page", lambda kind, match_id, timeout=20.0: pages.get(kind))
    info = REAL_CAPTURE("9", root=tmp_path)
    assert len(info["saved"]) == len(capture.PAGES) and info["curve_points"] == 3
    saved = tmp_path / "data" / "raw" / "cricbuzz-pages" / "9" / "final" / "live-cricket-graphs.html.gz"
    assert gzip.open(saved, "rt").read() == pages["live-cricket-graphs"]
    curve = json.loads((tmp_path / "data" / "live" / "cb-win-curve-9.json").read_text())
    assert curve["match_id"] == "9" and curve["points"]["1"][1]["India"] == 62
