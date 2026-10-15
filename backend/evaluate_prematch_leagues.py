"""Do extra T20 leagues improve the player ratings (and so the with-XIs win %)?

For each variant of (which competitions feed the player records, at what weight) the XI features
are rebuilt point-in-time, then the staged model is evaluated by rolling origin 2021-25: for each
year fit on earlier years only, predict that year, pool. Paired bootstrap of the Brier gain over
the baseline (T20Is + IPL only), on all games and on full-member games.
"""

from __future__ import annotations

import random
import sys

import numpy as np

from app.live.prematch_xi import DOMESTIC, FRANCHISE, build_dataset_xi
from train_prematch_stages import BLOCKS, fit_sequential

FULL = {"India", "Australia", "England", "South Africa", "New Zealand", "Pakistan", "Sri Lanka",
        "Bangladesh", "West Indies", "Afghanistan", "Zimbabwe", "Ireland"}
BASE_SOURCES = ("t20i", "ipl")
VARIANTS = {
    "T20I + IPL (current)": (BASE_SOURCES, {}),
    "+ franchise leagues, weight 1.0": (BASE_SOURCES + FRANCHISE, {}),
    "+ franchise leagues, weight 0.5": (BASE_SOURCES + FRANCHISE, {s: 0.5 for s in FRANCHISE}),
    "+ franchise 0.5 + domestic 0.25": (BASE_SOURCES + FRANCHISE + DOMESTIC,
                                         {**{s: 0.5 for s in FRANCHISE}, **{s: 0.25 for s in DOMESTIC}}),
    "+ everything, weight 1.0": (BASE_SOURCES + FRANCHISE + DOMESTIC, {}),
}


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict_final(model, rows):
    z = np.zeros(len(rows))
    for name, _ in BLOCKS:
        block = model[name]
        x = np.array([[r["x"][f] for f in block["features"]] for r in rows], float)
        z = z + (x / np.array(block["scale"])) @ np.array(block["weights"])
    return _sig(z)


def rolling(rows):
    preds, ys, fm = [], [], []
    for year in range(2021, 2026):
        train = [r for r in rows if r["date"] < f"{year}-01-01"]
        test = [r for r in rows if f"{year}-01-01" <= r["date"] < f"{year + 1}-01-01"]
        model = fit_sequential(train)
        preds += list(predict_final(model, test))
        ys += [r["y"] for r in test]
        fm += [r["a"] in FULL and r["b"] in FULL for r in test]
    return np.array(preds), np.array(ys, float), np.array(fm)


def boot(new, base, y, reps=2000):
    rng = random.Random(9)
    d = (base - y) ** 2 - (new - y) ** 2
    n = len(d)
    means = sorted(float(np.mean([d[rng.randrange(n)] for _ in range(n)])) for _ in range(reps))
    return float(d.mean()), means[int(0.025 * reps)], means[int(0.975 * reps)]


def main(only: list[str] | None = None) -> None:
    results = {}
    for name, (sources, weights) in VARIANTS.items():
        if only and not any(o in name for o in only):
            continue
        rows = build_dataset_xi(sources=sources, weights=weights)
        results[name] = rolling(rows)
        print(f"built {name}: {len(rows)} T20Is", flush=True)
    base_name = "T20I + IPL (current)"
    base_p, y, fm = results[base_name]
    for label, mask in (("ALL 2021-25", np.ones(len(y), bool)), ("FULL MEMBERS 2021-25", fm)):
        print(f"\\n{label} (n={int(mask.sum())})")
        for name, (p, _, _) in results.items():
            brier = np.mean((p[mask] - y[mask]) ** 2)
            acc = np.mean((p[mask] >= 0.5) == (y[mask] == 1))
            tail = "" if name == base_name else "  gain vs current %+.4f [%+.4f,%+.4f]" % boot(p[mask], base_p[mask], y[mask])
            print(f"  {name:<36} brier={brier:.4f} acc={acc:.3f}{tail}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
