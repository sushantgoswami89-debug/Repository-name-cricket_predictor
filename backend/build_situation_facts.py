"""Build models/situation_facts_v1.json: the historical record behind the chase Fact-o-Meter cards (user 2026-10-24: "these facts should have
come under Fact-o-Meter" -- e.g. teams that made 225+ in a men's T20 won 92% of 153 matches, the chasing side only 8%).

From the verified v3 over table (men's T20: internationals + IPL), matches since 2019, chasing side won = its final score above the first
innings total. Stores (a) [first innings total, chasing side won] per match and (b) for the chase checkpoints after 6, 10 and 15 overs
[required run rate, wickets down, chasing side won] per match. Usage: python build_situation_facts.py
"""
import json
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
CHECKPOINTS = (6, 10, 15)


def main() -> None:
    d = pd.read_csv(root / "data/candidates/v3/verified_training_overs.csv")
    d["year"] = pd.to_datetime(d["match_date"]).dt.year
    d = d[(d["year"] >= 2019) & d["innings"].isin([1, 2])]
    d["final"] = d["score_before_over"] + d["runs_in_over"]
    last = d.sort_values("over").groupby(["source_file", "innings"]).tail(1)
    totals = last.pivot(index="source_file", columns="innings", values="final")[[1, 2]].dropna()
    won = (totals[2] > totals[1]).astype(int)
    out = {"totals": [[int(t), int(w)] for t, w in zip(totals[1], won.loc[totals.index])], "chase": {}}
    for k in CHECKPOINTS:
        rows = d[(d["innings"] == 2) & (d["over"] == k + 1) & (d["source_file"].isin(totals.index))]   # state before over k+1 = after k overs
        out["chase"][str(k)] = [[round(float(r), 2), int(w), int(won.loc[f])]
                                for r, w, f in zip(rows["required_run_rate"], rows["wkts_down_before_over"], rows["source_file"])]
    out["matches"] = int(len(totals))
    (root / "models" / "situation_facts_v1.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print("matches", out["matches"], {k: len(v) for k, v in out["chase"].items()})


if __name__ == "__main__":
    main()
