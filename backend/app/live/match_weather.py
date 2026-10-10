"""Match-day weather from Open-Meteo (free, no key) for the live briefing.

One forecast call for the match city and date; returns the same daily summary the
historical backfill (fetch_match_weather.py) stores, so the model sees identical
features live and in training. Never raises: any failure returns None and the
briefing simply omits the conditions line.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[3]
GEOCODE_CACHE = ROOT / "data" / "external" / "weather" / "geocode.json"
HOURLY_VARS = "temperature_2m,relative_humidity_2m,dew_point_2m,cloud_cover,precipitation"


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 2) if values else None


def summarise(hourly: dict) -> dict[str, dict]:
    """Hourly arrays (local time) -> per-day summary: evening 19-22 temp/dew
    point/humidity, afternoon 12-18 cloud, total rain."""
    by_day: dict[str, dict[int, tuple]] = defaultdict(dict)
    for i, stamp in enumerate(hourly["time"]):
        by_day[stamp[:10]][int(stamp[11:13])] = (
            hourly["temperature_2m"][i],
            hourly["relative_humidity_2m"][i],
            hourly["dew_point_2m"][i],
            hourly["cloud_cover"][i],
            hourly["precipitation"][i],
        )
    out = {}
    for day, hours in by_day.items():
        evening = [hours[h] for h in range(19, 23) if h in hours]
        afternoon = [hours[h] for h in range(12, 19) if h in hours]
        match_hours = [hours[h] for h in range(17, 24) if h in hours]  # a T20 runs roughly 18:00-23:00
        out[day] = {
            "eve_rain_mm": round(sum(x[4] or 0 for x in match_hours), 2),
            "eve_rain_hours": sum(1 for x in match_hours if (x[4] or 0) >= 0.1),
            "eve_temp": _mean(x[0] for x in evening),
            "eve_humidity": _mean(x[1] for x in evening),
            "eve_dew_point": _mean(x[2] for x in evening),
            "aft_cloud": _mean(x[3] for x in afternoon),
            "eve_cloud": _mean(x[3] for x in evening),
            "rain_mm": round(sum(x[4] or 0 for x in hours.values()), 2),
        }
    return out


def _place(city: str) -> dict | None:
    try:
        cached = json.loads(GEOCODE_CACHE.read_text(encoding="utf-8")).get(city)
        if cached:
            return cached
    except (OSError, ValueError):
        pass
    try:
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 5, "language": "en", "format": "json"},
            timeout=10,
        )
        results = response.json().get("results") or []
        if results:
            best = max(results, key=lambda r: r.get("population") or 0)
            return {"lat": best["latitude"], "lon": best["longitude"], "name": best.get("name")}
    except (requests.RequestException, ValueError, KeyError):
        pass
    return None


def match_day_weather(city: str, day_iso: str | None = None, attempts: int = 3) -> dict | None:
    """Forecast summary for `city` on `day_iso` (YYYY-MM-DD, the match's local date);
    with no date, the venue's local "today" (match day when run on match day).
    Retries a transient failure (seen once on 2026-10-10: the briefing silently lost its
    weather and dew), then gives up with None."""
    import time as _time

    for attempt in range(attempts):
        result = _match_day_weather_once(city, day_iso)
        if result is not None:
            return result
        if attempt + 1 < attempts:
            _time.sleep(1.0 * (attempt + 1))
    return None


def _match_day_weather_once(city: str, day_iso: str | None) -> dict | None:
    place = _place(city.strip().lower())
    if not place:
        return None
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["lat"],
                "longitude": place["lon"],
                "hourly": HOURLY_VARS,
                "daily": "sunset",
                **({"start_date": day_iso, "end_date": day_iso} if day_iso else {"forecast_days": 2}),
                "timezone": "auto",
            },
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        if not day_iso:
            from datetime import datetime, timedelta, timezone

            local_now = datetime.now(timezone.utc) + timedelta(seconds=payload.get("utc_offset_seconds", 0))
            day_iso = local_now.date().isoformat()
        day = summarise(payload["hourly"]).get(day_iso)
        if day is None:
            return None
        # Hour-by-hour detail for the live dew outlook (not stored in the historical backfill).
        hours = {}
        for i, stamp in enumerate(payload["hourly"]["time"]):
            if stamp[:10] == day_iso and None not in (
                payload["hourly"]["temperature_2m"][i], payload["hourly"]["dew_point_2m"][i]
            ):
                hours[int(stamp[11:13])] = {
                    "temp": payload["hourly"]["temperature_2m"][i],
                    "dew_point": payload["hourly"]["dew_point_2m"][i],
                    "humidity": payload["hourly"]["relative_humidity_2m"][i],
                }
        sunset = None
        try:
            stamp = payload["daily"]["sunset"][payload["daily"]["time"].index(day_iso)]
            sunset = int(stamp[11:13]) + int(stamp[14:16]) / 60.0
        except (KeyError, ValueError, IndexError, TypeError):
            pass
        return {**day, "hours": hours, "sunset_h": sunset, "utc_offset_s": payload.get("utc_offset_seconds", 0)}
    except (requests.RequestException, ValueError, KeyError):
        return None
