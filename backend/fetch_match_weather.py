"""Pull historical weather for every men's T20I / IPL match city from Open-Meteo.

Open-Meteo's archive is free and keyless (https://open-meteo.com). For each city we
geocode once, then fetch hourly temperature, humidity, dew point, cloud cover and
rain for the calendar years in which matches were played there, and keep only a
small daily summary per match date (local time):

  evening (19:00-22:00): mean temperature, dew point, humidity  -> dew likelihood
  afternoon (12:00-18:00): mean cloud cover                      -> overcast
  whole day: total precipitation

Cricsheet has no match start times, so "evening" is assumed for every match; day
games are therefore noisy for dew and the model must learn the average effect.

Resumable: results are cached in data/external/weather/. Output files:
  geocode.json     city key -> {name, country, lat, lon}
  city_daily.json  city key -> date -> summary
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

from app.live.match_weather import HOURLY_VARS, summarise
from app.live.prematch_model import ipl_rows, t20i_rows

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external" / "weather"
GEOCODE = OUT_DIR / "geocode.json"
DAILY = OUT_DIR / "city_daily.json"
_NOISE = {"international", "cricket", "stadium", "ground", "oval", "club", "sports", "complex"}


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _get(url: str, params: dict, tries: int = 4) -> dict | None:
    for attempt in range(tries):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 429:
                time.sleep(20 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
    return None


# Cricsheet "city" values the geocoder does not know -> a nearby place it does.
ALIASES = {
    "kigali city": "Kigali", "ilfov county": "Bucharest", "brondby": "Copenhagen",
    "ishoj": "Copenhagen", "new chandigarh": "Chandigarh", "st lucia": "Gros Islet",
    "st saviour": "Saint Helier", "port  soif": "Guernsey",
    "st georges quilmes": "Buenos Aires",
}


def geocode(city: str) -> dict | None:
    candidates = [ALIASES.get(city, city)]
    if city not in candidates:
        candidates.append(city)
    first = city.split()[0] if city.split() else city
    if any(word in _NOISE for word in city.split()) and first != city:
        candidates.append(first)
    for name in candidates:
        data = _get(
            "https://geocoding-api.open-meteo.com/v1/search",
            {"name": name, "count": 5, "language": "en", "format": "json"},
        )
        results = (data or {}).get("results") or []
        if results:
            best = max(results, key=lambda r: r.get("population") or 0)
            return {
                "query": city,
                "name": best.get("name"),
                "country": best.get("country"),
                "lat": best["latitude"],
                "lon": best["longitude"],
            }
        time.sleep(0.2)
    return None


def _clusters(dates: list[str], gap_days: int = 20) -> list[tuple[str, str]]:
    """Group sorted ISO dates into windows so one request covers a whole season."""
    from datetime import date

    out: list[tuple[str, str]] = []
    for day in dates:
        if out and (date.fromisoformat(day) - date.fromisoformat(out[-1][1])).days <= gap_days:
            out[-1] = (out[-1][0], day)
        else:
            out.append((day, day))
    return out


def main() -> int:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    needed: dict[str, set[str]] = defaultdict(set)
    for row in list(t20i_rows()) + list(ipl_rows()):
        needed[row["city"]].add(row["date"])

    geo = _load(GEOCODE)
    daily = _load(DAILY)
    cities = sorted(needed, key=lambda c: -len(needed[c]))
    for city in cities:
        if not geo.get(city):
            geo[city] = geocode(city)
            GEOCODE.write_text(json.dumps(geo, indent=1))
            time.sleep(0.15)
    skipped = [c for c in cities if not geo.get(c)]
    print(f"geocoded {len(cities) - len(skipped)}/{len(cities)} cities; unresolved: {skipped}", flush=True)

    tasks = []
    for city in cities:
        if not geo.get(city):
            continue
        have = daily.setdefault(city, {})
        for first, last in _clusters(sorted(d for d in needed[city] if d not in have)):
            tasks.append((city, first, last))
    print(f"{len(tasks)} date windows to fetch", flush=True)

    lock = threading.Lock()

    def fetch(task):
        city, first, last = task
        place = geo[city]
        data = _get(
            "https://archive-api.open-meteo.com/v1/archive",
            {"latitude": place["lat"], "longitude": place["lon"], "start_date": first,
             "end_date": last, "hourly": HOURLY_VARS, "timezone": "auto"},
        )
        return task, (summarise(data["hourly"]) if data and "hourly" in data else None)

    done = 0
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(fetch, t) for t in tasks]
        for future in as_completed(futures):
            (city, first, last), summary = future.result()
            with lock:
                done += 1
                if summary:
                    for day in needed[city]:
                        if first <= day <= last and day in summary:
                            daily[city][day] = summary[day]
                else:
                    print(f"no data: {city} {first}..{last}", flush=True)
                if done % 50 == 0:
                    DAILY.write_text(json.dumps(daily))
                    print(f"{done}/{len(tasks)} windows", flush=True)
    DAILY.write_text(json.dumps(daily))
    covered = sum(len(v) for v in daily.values())
    total = sum(len(v) for v in needed.values())
    print(f"done: {covered}/{total} city-dates have weather")
    return 0


if __name__ == "__main__":
    sys.exit(main())
