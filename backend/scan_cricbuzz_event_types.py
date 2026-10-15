"""Scan many real completed Cricbuzz matches' cached commentary windows
for the event types still unconfirmed against real data (docs/
live_field_test_2026-09-15.md item 5): NO-BALL, BYE, LEG-BYE, and every
wicket kind except caught/bowled/run-out (lbw, stumped, hit-wicket,
caught and bowled).

Each match's page only carries a small rolling window of recent balls
(confirmed 2026-09-18, even for completed matches -- Cricbuzz has no
retrievable full ball-by-ball archive), so this samples many matches'
tail-ends rather than one full match, to raise the odds of catching each
event type somewhere.

Usage:
    .venv/bin/python3 backend/scan_cricbuzz_event_types.py
"""

import sys
import time

import requests

sys.path.insert(0, ".")
from app.live.cricbuzz_discovery import CricbuzzMatchDiscovery
from app.live.cricbuzz_reader import (
    _classify_wicket,
    _delivery_objects,
    _extract_flight_chunks,
    _runs_and_extras,
)

TARGETS = {"noballs", "byes", "legbyes"}
WICKET_TARGETS = {"lbw", "stumped", "hit wicket", "caught and bowled"}

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; CricketBaba/3.0)"})

discovery = CricbuzzMatchDiscovery()
matches = discovery.matches()
completed = [m for m in matches if m.state.casefold() == "complete"]
print(f"Scanning {len(completed)} completed matches' cached windows...")

found_extras: dict[str, str] = {}
found_wickets: dict[str, str] = {}

for match in completed:
    try:
        response = session.get(f"{match.url}?_cb={int(time.time()*1000)}", timeout=12)
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"  {match.label}: fetch failed ({error})")
        continue
    chunks = _extract_flight_chunks(response.text)
    objs = _delivery_objects(chunks, match.match_id)
    for obj in objs:
        comm_text = obj.get("commText", "")
        events = [str(e) for e in obj.get("event", [])]
        _, _, extras = _runs_and_extras(comm_text, events)
        for extra_key in extras:
            if extra_key in TARGETS and extra_key not in found_extras:
                found_extras[extra_key] = f"{match.label}: {comm_text!r}"
                print(f"FOUND extra={extra_key} in {match.label}")
        if "wicket" in events:
            kind = _classify_wicket(comm_text)
            if kind in WICKET_TARGETS and kind not in found_wickets:
                found_wickets[kind] = f"{match.label}: {comm_text!r}"
                print(f"FOUND wicket={kind} in {match.label}")
    if len(found_extras) == len(TARGETS) and len(found_wickets) == len(WICKET_TARGETS):
        break

print("\n=== Summary ===")
print("Extras found:", list(found_extras))
print("Extras still missing:", TARGETS - set(found_extras))
print("Wicket kinds found:", list(found_wickets))
print("Wicket kinds still missing:", WICKET_TARGETS - set(found_wickets))
print()
for key, example in {**found_extras, **found_wickets}.items():
    print(f"{key}: {example}")
