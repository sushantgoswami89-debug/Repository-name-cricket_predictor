"""Replay the Cricbuzz pages the live driver saved (data/raw/cricbuzz-pages/<match id>/) through a fresh match session.

One page = one poll, in the order they were saved, so the console you get is what a driver started at the first saved page would
have printed (no restarts). Used to find out WHEN each over-end forecast comes out and why a forecast is late or held, without
waiting for a live match. Nothing is written to the real logs: every log goes to a temp folder, scorecard fetches are off.

    cd backend && ../.venv/bin/python replay_saved_pages.py 170697 [--from 0] [--to 99]
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import tempfile
from pathlib import Path

from app.live import cricbuzz_session, prematch_log
from app.live.cricbuzz_reader import CricbuzzLiveReader
from app.live.cricbuzz_session import CricbuzzMatchSession
from app.live.pipeline import VerifiedLivePredictionPipeline
from app.ml.match_winner_ensemble_shadow_log import MatchWinnerEnsembleShadowLog
from app.ml.run_range_ensemble_shadow_log import RunRangeEnsembleShadowLog
from app.ml.wicket_ensemble_shadow_log import WicketEnsembleShadowLog
from app.ml.win_probability_shadow_log import WinProbabilityShadowLog

ROOT = Path(__file__).resolve().parents[1]


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _PagePlayer:
    """Stands in for requests.Session: each get() returns the next saved page."""

    def __init__(self, pages: list[tuple[int, str]]) -> None:
        self.pages = pages
        self.index = -1
        self.headers: dict = {}

    def advance(self) -> int | None:
        self.index += 1
        return self.pages[self.index][0] if self.index < len(self.pages) else None

    def get(self, url, timeout=None):  # noqa: ARG002
        return _Response(self.pages[min(self.index, len(self.pages) - 1)][1])


def load_pages(match_id: str) -> list[tuple[int, str]]:
    folder = ROOT / "data" / "raw" / "cricbuzz-pages" / match_id
    pages = []
    for path in sorted(folder.glob("*.html.gz")):
        pages.append((int(path.name.split("-")[0]), gzip.decompress(path.read_bytes()).decode("utf-8", "ignore")))
    return pages


def build_session(match_id: str, tmp: Path, player: _PagePlayer) -> CricbuzzMatchSession:
    prematch_log.LOG_PATH = tmp / "prematch-predictions.jsonl"
    prematch_log.RESULTS_PATH = tmp / "prematch-results.jsonl"
    cricbuzz_session.fetch_scorecard = lambda *_a, **_k: None  # the scorecard page is a live request
    pipeline = VerifiedLivePredictionPipeline(
        publisher=None,
        output_file=tmp / f"cricbuzz-{match_id}.json",
        win_probability_shadow_log=WinProbabilityShadowLog(tmp / "wp.jsonl", tmp / "wp-out.jsonl", tmp / "wp.marker"),
        wicket_ensemble_shadow_log=WicketEnsembleShadowLog(tmp / "wk.jsonl", tmp / "wk-out.jsonl", tmp / "wk.marker"),
        run_range_ensemble_shadow_log=RunRangeEnsembleShadowLog(tmp / "rr.jsonl", tmp / "rr-out.jsonl", tmp / "rr.marker"),
        match_winner_ensemble_shadow_log=MatchWinnerEnsembleShadowLog(tmp / "mw.jsonl", tmp / "mw-out.jsonl", tmp / "mw.marker"),
    )
    def reader_on_saved_pages() -> CricbuzzLiveReader:
        reader = CricbuzzLiveReader()
        reader._session = player
        return reader

    # a re-sync builds a fresh reader inside the session: it must read the saved pages too, never the live site
    cricbuzz_session.CricbuzzLiveReader = reader_on_saved_pages
    reader = reader_on_saved_pages()
    session = CricbuzzMatchSession(
        match_id=match_id,
        match_url=f"https://www.cricbuzz.com/live-cricket-scores/{match_id}/x",
        pipeline=pipeline,
        reader=reader,
        commentary_log_path=tmp / f"commentary-{match_id}.jsonl",
        tag_lines=False,
    )
    session._capture_all_pages = lambda: None
    return session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("match_id")
    parser.add_argument("--from", dest="first", type=int, default=0)
    parser.add_argument("--to", dest="last", type=int, default=10**6)
    args = parser.parse_args()
    pages = load_pages(args.match_id)[args.first:args.last + 1]
    if not pages:
        raise SystemExit(f"no saved pages for {args.match_id}")
    player = _PagePlayer(pages)
    with tempfile.TemporaryDirectory() as folder:
        session = build_session(args.match_id, Path(folder), player)
        import time as _time

        for _ in pages:
            stamp = player.advance()
            print(f"--- page {player.index} saved {_time.strftime('%H:%M:%S', _time.localtime(stamp / 1000))}")
            sys.stdout.flush()
            try:
                session.poll_once()
            except Exception as error:  # a replay must show the failure, not stop at it
                print(f"!! poll raised {type(error).__name__}: {error}")
            if session.finished:
                break


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
