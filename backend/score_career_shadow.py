"""Score the silent career-strike-rate shadow against the live over outlook on the SAME real overs (user 2026-10-24).
Reads data/live/over-outlook-career-shadow.jsonl (written by the pipeline for every forecast where the next over's batters are known) and Cricbuzz's
official over summaries (over-summaries-<match>.jsonl) for what actually happened: runs and whether a wicket fell. Reports, for the live version and the
career version: bias and RMSE of the mean runs, Brier of the big / quiet / wicket chances, and how many batters were matched to a career record.
Usage: python score_career_shadow.py [match_id ...]      (no ids = every match in the shadow log)
"""
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

LIVE = Path(os.environ.get("CRICKETBABA_LIVE_DIR") or Path(__file__).resolve().parents[1] / "data" / "live")   # the env var lets a replay folder be scored
WICKET = re.compile(r"^(\d?W|W\d?)$")


def actual_overs(match_id: str) -> dict[tuple[int, int], tuple[int, bool]]:
    out = {}
    path = LIVE / f"over-summaries-{match_id}.jsonl"
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[(row["innings"], row["over"])] = (int(row["over_runs"]), any(WICKET.match(t) for t in str(row.get("summary", "")).split()))
    return out


def main(match_ids: list[str]) -> None:
    path = LIVE / "over-outlook-career-shadow.jsonl"
    if not path.exists():
        print("no shadow log yet (it is written during live matches)")
        return
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if not match_ids or r["match_id"] in match_ids:
                records[(r["match_id"], r["innings"], r["over"])] = r      # last write per over
    cache: dict[str, dict] = {}
    rows = []
    for (match_id, innings, over), r in sorted(records.items()):
        actual = cache.setdefault(match_id, actual_overs(match_id)).get((innings, over))
        if actual and r.get("live") and r.get("career"):
            rows.append((r, actual))
    print(f"{len(records)} shadow forecasts logged, {len(rows)} with a finished over to score, from {len({r['match_id'] for r, _ in rows})} match(es)")
    if not rows:
        return
    tiers = Counter(t for r, _ in rows for t in r["career"]["tiers"].values())
    print("batter matching:", dict(tiers))
    for label, keep in (("all overs", lambda r: True),
                        ("both batters matched", lambda r: "unmatched" not in r["career"]["tiers"].values())):
        sel = [(r, a) for r, a in rows if keep(r)]
        if not sel:
            continue
        runs = np.array([a[0] for _, a in sel], float)
        wicket = np.array([a[1] for _, a in sel], float)
        print(f"\n{label}: {len(sel)} overs   (actual mean {runs.mean():.2f} runs, big {np.mean(runs >= 10):.0%}, quiet {np.mean(runs <= 4):.0%}, wicket {wicket.mean():.0%})")
        print(f"{'':<14}{'bias':>7}{'RMSE':>7}{'bigBrier':>10}{'quietBrier':>12}{'wktBrier':>10}")
        for name in ("live", "career"):
            mean = np.array([r[name]["mean_runs"] for r, _ in sel])
            big = np.array([r[name]["p_big"] for r, _ in sel])
            quiet = np.array([r[name]["p_quiet"] for r, _ in sel])
            wkt = np.array([r[name]["p_wicket"] for r, _ in sel])
            print(f"{name:<14}{(mean - runs).mean():7.2f}{np.sqrt(((mean - runs) ** 2).mean()):7.2f}{((big - (runs >= 10)) ** 2).mean():10.4f}"
                  f"{((quiet - (runs <= 4)) ** 2).mean():12.4f}{((wkt - wicket) ** 2).mean():10.4f}")
    print("\nA few matches say little: the offline test (60,988 overs) found RMSE -0.07 and big-over AUC +0.02; wait for ~30 matches before a decision.")


if __name__ == "__main__":
    main(sys.argv[1:])
