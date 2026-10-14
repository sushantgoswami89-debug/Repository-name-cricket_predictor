"""Is the live run forecast biased low / flat in a clean replay? (2026-10-13)

Live over-end forecasts so far average 5.9 runs against 8.1 actual (n=186) and barely move (std 1.6 vs 5.0). This replays recent
Cricsheet men's T20Is through the REAL live pipeline (the same path as the driver, full Cricsheet names) with every log in a temp
folder, then compares each over's forecast with what was scored. If the bias shows here too the MODEL is low; if not, the live
feed (names, venue, toss, context) is what differs.

    cd backend && ../.venv/bin/python experiment_live_bias.py [matches=40]
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
from pathlib import Path

from app.live.live_snapshot import LiveDelivery, LiveSnapshot
from app.live.pipeline import VerifiedLivePredictionPipeline
from app.ml.match_winner_ensemble_shadow_log import MatchWinnerEnsembleShadowLog
from app.ml.run_range_ensemble_shadow_log import RunRangeEnsembleShadowLog
from app.ml.wicket_ensemble_shadow_log import WicketEnsembleShadowLog
from app.ml.win_probability_shadow_log import WinProbabilityShadowLog

ROOT = Path(__file__).resolve().parents[1]


def pipeline_in(tmp: Path) -> VerifiedLivePredictionPipeline:
    return VerifiedLivePredictionPipeline(
        publisher=None,
        win_probability_shadow_log=WinProbabilityShadowLog(tmp / "wp.jsonl", tmp / "wp-o.jsonl", tmp / "wp.m"),
        wicket_ensemble_shadow_log=WicketEnsembleShadowLog(tmp / "wk.jsonl", tmp / "wk-o.jsonl", tmp / "wk.m"),
        run_range_ensemble_shadow_log=RunRangeEnsembleShadowLog(tmp / "rr.jsonl", tmp / "rr-o.jsonl", tmp / "rr.m"),
        match_winner_ensemble_shadow_log=MatchWinnerEnsembleShadowLog(tmp / "mw.jsonl", tmp / "mw-o.jsonl", tmp / "mw.m"),
    )


KNOW_BOWLER = "--bowler" in sys.argv
LAST_MEAN: list[float] = []


def _record_mean(original):
    def wrapper(probabilities, width):
        import numpy as np

        LAST_MEAN[:] = [float((np.asarray(probabilities)[0] * np.arange(np.asarray(probabilities).shape[1])).sum())]
        return original(probabilities, width)

    return wrapper


def replay(raw: dict, match_id: str, pipeline: VerifiedLivePredictionPipeline) -> list[tuple[int, int, float, float, int]]:
    """(innings, over, predicted expected_runs, point, actual) for every forecast that was later scored."""
    teams = raw["info"]["teams"]
    rows = []
    first_total = None
    for number, innings in enumerate([i for i in raw["innings"] if not i.get("super_over")], start=1):
        batting = innings["team"]
        bowling = next(t for t in teams if t != batting)
        deliveries, score, wickets, timestamp = [], 0, 0, 0
        target = first_total + 1 if number == 2 and first_total is not None else 0
        pending = None
        for source_over in innings["overs"]:
            over_number = int(source_over["over"]) + 1
            legal = 0
            for ball, d in enumerate(source_over["deliveries"], start=1):
                timestamp += 1000
                extras = d.get("extras", {}) or {}
                runs = int(d["runs"]["total"])
                score += runs
                wickets += int(bool(d.get("wickets")))
                legal += int("wides" not in extras and "noballs" not in extras)
                deliveries.append(LiveDelivery(
                    match_id=match_id, innings=number, over=over_number, ball=ball, bowler=d["bowler"], striker=d["batter"],
                    total_runs=runs, batter_runs=int(d["runs"]["batter"]), extras={k: int(v) for k, v in extras.items()},
                    wicket_kind=str(d["wickets"][0]["kind"]) if d.get("wickets") else None, feed_total=score, feed_wickets=wickets,
                    timestamp_ms=timestamp, commentary=""))
                nxt = None
                if KNOW_BOWLER and legal == 6 and source_over is not innings["overs"][-1]:
                    nxt = innings["overs"][innings["overs"].index(source_over) + 1]["deliveries"][0]["bowler"]
                snapshot = LiveSnapshot(
                    match_id=match_id, match_format="T20", batting_team=batting, bowling_team=bowling, innings=number, score=score,
                    wickets=wickets, overs=f"{over_number - 1}.{legal}", is_live=True, deliveries=tuple(deliveries), target=target,
                    scheduled_overs=20, venue_name=raw["info"].get("venue", ""),
                    announced_bowler=nxt or "", announced_bowler_over=over_number + 1 if nxt else 0)
                try:
                    out = pipeline.process(snapshot)
                except Exception:
                    continue
                if out:
                    review = out.get("previous_over_evaluation")
                    if review and pending:
                        rows.append((number, pending[0], pending[1], pending[2], int(review["actual_runs"]), pending[3], pending[4]))
                    pred = out["prediction"]
                    pending = (out["over"], float(pred.get("expected_runs", 0)), float(pred.get("point", pred.get("expected_runs", 0))), LAST_MEAN[0] if LAST_MEAN else 0.0, str(pred.get("expected_range", "")))
        if number == 1:
            first_total = score
    return rows


def main() -> None:
    # the shadow ensemble starts a subprocess for its neural net on every forecast (~seconds each); the forecast shown to the user
    # comes from the production engine, so the net is switched off here to keep the replay fast
    from app.ml.match_winner_ensemble_runtime import MatchWinnerEnsembleRuntime
    from app.ml.run_range_ensemble_runtime import RunRangeEnsembleRuntime
    from app.ml.wicket_ensemble_runtime import WicketEnsembleRuntime

    RunRangeEnsembleRuntime._nn_probabilities = lambda self, row: None
    WicketEnsembleRuntime._nn_probability = lambda self, row: None
    MatchWinnerEnsembleRuntime._nn_probability = lambda self, row: None
    import runtime_run_range_v3

    runtime_run_range_v3.select_inclusive_bands = _record_mean(runtime_run_range_v3.select_inclusive_bands)
    limit = int([a for a in sys.argv[1:] if not a.startswith("--")][0]) if [a for a in sys.argv[1:] if not a.startswith("--")] else 40
    files = []
    for path in (ROOT / "data" / "raw" / "cricsheet" / "t20i").glob("*.json"):
        info = json.loads(path.read_text(encoding="utf-8"))["info"]
        if info.get("gender") == "male" and str(info["dates"][0]) >= "2025-01-01":
            files.append((str(info["dates"][0]), path))
    files.sort()
    files = files[-limit:]
    rows = []
    with tempfile.TemporaryDirectory() as folder:
        pipeline = pipeline_in(Path(folder))  # built once: loading the models takes seconds; a new match/innings resets its scope
        for _date, path in files:
            raw = json.loads(path.read_text(encoding="utf-8"))
            rows += replay(raw, path.stem, pipeline)
    if not rows:
        raise SystemExit("no scored forecasts")
    pred = [r[2] for r in rows]
    act = [r[4] for r in rows]
    print(f"matches {len(files)}  forecasts scored {len(rows)}")
    print(f"mean forecast {statistics.mean(pred):.2f}  mean actual {statistics.mean(act):.2f}  bias {statistics.mean(pred) - statistics.mean(act):+.2f}")
    print(f"MAE {statistics.mean(abs(p - a) for p, a in zip(pred, act)):.2f}  std forecast {statistics.pstdev(pred):.2f}  std actual {statistics.pstdev(act):.2f}")
    hits = sum(1 for r in rows if r[6] and int(r[6].split('-')[0]) <= r[4] <= int(r[6].split('-')[1]))
    print(f"bowler known at the over end: {KNOW_BOWLER}   band hit rate {hits / len(rows):.1%}")
    mean_forecast = [r[5] for r in rows]
    print(f"distribution MEAN: forecast {statistics.mean(mean_forecast):.2f} vs actual {statistics.mean(act):.2f}  bias {statistics.mean(mean_forecast) - statistics.mean(act):+.2f}  "
          f"MAE {statistics.mean(abs(p - a) for p, a in zip(mean_forecast, act)):.2f}  std {statistics.pstdev(mean_forecast):.2f}")
    mm = statistics.mean(mean_forecast)
    covm = sum((p - mm) * (a - statistics.mean(act)) for p, a in zip(mean_forecast, act)) / len(act)
    print(f"  correlation of the distribution mean with actual {covm / (statistics.pstdev(mean_forecast) * statistics.pstdev(act)):.2f}")
    ordered = sorted(act)
    print(f"actual median {ordered[len(ordered) // 2]}  share of overs under 7: {sum(a < 7 for a in act) / len(act):.0%}")
    mp, ma = statistics.mean(pred), statistics.mean(act)
    cov = sum((p - mp) * (a - ma) for p, a in zip(pred, act)) / len(act)
    print(f"correlation forecast vs actual {cov / (statistics.pstdev(pred) * statistics.pstdev(act)):.2f}")
    for low, high in ((0, 5), (5, 6), (6, 7), (7, 8), (8, 20)):
        sub = [(p, a) for p, a in zip(pred, act) if low <= p < high]
        if sub:
            print(f"  forecast {low}-{high}: n {len(sub)}  mean forecast {statistics.mean(p for p, _ in sub):.2f}  mean actual {statistics.mean(a for _, a in sub):.2f}")
    for innings in (1, 2):
        sub = [r for r in rows if r[0] == innings]
        print(f"  innings {innings}: n {len(sub)}  forecast {statistics.mean(r[2] for r in sub):.2f}  actual {statistics.mean(r[4] for r in sub):.2f}")


if __name__ == "__main__":
    main()
