"""Score the live pre-match win % log against how the matches ended.

Usage: python score_prematch_log.py   (ICC top-10 sides only: see prematch_log.ICC_TOP_10)
Coin flip = 50% right, Brier 0.2500. The backtest to compare with (2021-25, fit on earlier years only):
all games ~67.7% right / 0.2056-0.2098; full-member games ~63.7% right / 0.2235-0.2270.
With only a handful of matches the live numbers are anecdotes: one match is a 30%-70% coin.
"""

from app.live.prematch_log import score


def report(fmt: str, title: str) -> None:
    s = score(fmt=fmt)
    print(f"{title}: matches with logged predictions: {s['matches_logged']}; finished matches recorded: {s['results']}; no result: {s['no_result']}")
    for stage in ("pre-toss", "after toss", "with XIs"):
        c = s["by_stage"].get(stage)
        if not c:
            print(f"  {stage:<11} no scored matches yet")
            continue
        print(f"  {stage:<11} n={c['n']:<3} right {c['right']}/{c['n']} ({c['right'] / c['n']:.0%})  Brier {c['brier']:.4f}")
        for match_id, a, b, p, winner in c["games"]:
            print(f"      {a} v {b}: said {a} {p:.0%} -> {winner} won")


def main() -> None:
    report("t20", "T20I")
    report("odi", "ODI (backtest 2019-25: ~64% right, Brier 0.220)")


if __name__ == "__main__":
    main()
