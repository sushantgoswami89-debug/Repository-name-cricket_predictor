"""Run the verified TOI → Candidate v3 → Telegram live pipeline."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from app.live.pipeline import VerifiedLivePredictionPipeline
from app.live.telegram import TelegramPublisher
from app.live.toi_reader import ToiFeedError, ToiLiveReader
from app.live.verification import VerificationError
from app.ml.model_repository import ModelRepository
from app.ml.prediction_engine import PredictionEngine


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("match_url", help="Full TOI match-center URL")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--incomplete-retry-seconds",
        type=float,
        help=(
            "When no over-boundary prediction is due, retry after this shorter "
            "interval instead of the normal poll interval."
        ),
    )
    parser.add_argument("--once", action="store_true", help="Fetch and verify once")
    parser.add_argument(
        "--output", type=Path, default=project_root / "live_prediction.json"
    )
    parser.add_argument("--telegram", action="store_true")
    parser.add_argument(
        "--stuck-alert-seconds",
        type=float,
        default=300.0,
        help=(
            "If the feed stays unreconciled (ToiFeedError) continuously for "
            "this long, emit a clear 'feed_stuck' alert -- and one Telegram "
            "message, if --telegram is set -- instead of silently retrying "
            "forever. A stuck feed and a working-but-quiet one look "
            "identical from outside without this."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help=(
            "Model directory to load explicitly "
            "(use the locked candidate for shadow)."
        ),
    )
    args = parser.parse_args()

    publisher = None
    if args.telegram:
        saved_config = project_root / "backend" / ".telegram.json"
        config = {}
        if saved_config.exists():
            config = json.loads(saved_config.read_text(encoding="utf-8"))
        publisher = TelegramPublisher(
            os.environ.get("TELEGRAM_BOT_TOKEN", config.get("bot_token", "")),
            os.environ.get("TELEGRAM_CHAT_ID", config.get("chat_id", "")),
        )
    reader = ToiLiveReader()
    def make_engine():
        if args.model_dir:
            return PredictionEngine(repository=ModelRepository(args.model_dir))
        return None

    engine = make_engine()
    pipeline = VerifiedLivePredictionPipeline(
        engine=engine, publisher=publisher, output_file=args.output
    )
    match_id = reader.match_id_from_url(args.match_url)
    feed_error_streak_started_at: float | None = None
    stuck_alert_sent = False
    while True:
        try:
            snapshot = reader.fetch(args.match_url)
            output = pipeline.process(snapshot)
            feed_error_streak_started_at = None
            stuck_alert_sent = False
            print(
                json.dumps(
                    output
                    or {
                        "match_id": snapshot.match_id,
                        "verified_deliveries": len(snapshot.deliveries),
                        "score": snapshot.score,
                        "wickets": snapshot.wickets,
                        "prediction": "not due",
                    },
                    indent=2,
                )
            )
        except ToiFeedError as exc:
            # TOI's independent score, scorecard, and commentary endpoints can
            # cross briefly at a live boundary.  In continuous mode, retain the
            # verified pipeline state and retry quickly; never publish the
            # inconsistent snapshot.
            if args.once:
                raise
            feed_retry = (
                args.incomplete_retry_seconds
                if args.incomplete_retry_seconds is not None
                else args.poll_seconds
            )
            now = time.time()
            if feed_error_streak_started_at is None:
                feed_error_streak_started_at = now
            stuck_seconds = now - feed_error_streak_started_at
            print(
                json.dumps(
                    {
                        "status": "feed_sync_pending",
                        "detail": str(exc),
                        "retry_seconds": feed_retry,
                        "unreconciled_for_seconds": round(stuck_seconds, 1),
                    },
                    indent=2,
                )
            )
            if stuck_seconds >= args.stuck_alert_seconds and not stuck_alert_sent:
                stuck_alert_sent = True
                stuck_minutes = stuck_seconds / 60.0
                duration_text = (
                    f"{stuck_minutes:.1f} min"
                    if stuck_minutes >= 1
                    else f"{int(stuck_seconds)}s"
                )
                stuck_message = (
                    "⚠️ CricketBaba: this match's live feed has been "
                    f"inconsistent for over {duration_text} -- "
                    "predictions are paused until TOI's data reconciles. "
                    "No action needed; will resume automatically."
                )
                print(
                    json.dumps(
                        {"status": "feed_stuck", "detail": stuck_message},
                        indent=2,
                    )
                )
                if publisher is not None:
                    try:
                        publisher.publish(
                            f"stuck_alert:{match_id}:{int(feed_error_streak_started_at)}",
                            stuck_message,
                        )
                    except Exception:
                        pass
            time.sleep(max(1.0, feed_retry))
            continue
        except VerificationError as exc:
            # Never publish a disputed revision.  A continuous shadow reader
            # can safely rebase on the next fully reconciled snapshot while
            # retaining the same publisher (and its duplicate-message guard).
            if args.once:
                raise
            print(
                json.dumps(
                    {
                        "status": "verification_rebase",
                        "detail": str(exc),
                        "retry_seconds": 0.5,
                    },
                    indent=2,
                )
            )
            engine = make_engine()
            pipeline = VerifiedLivePredictionPipeline(
                engine=engine,
                publisher=publisher,
                output_file=args.output,
                restore_verified_state=False,
            )
            time.sleep(0.5)
            continue
        if args.once:
            break
        delay = args.poll_seconds
        if output is None and args.incomplete_retry_seconds is not None:
            delay = args.incomplete_retry_seconds
        time.sleep(max(1.0, delay))


if __name__ == "__main__":
    main()
