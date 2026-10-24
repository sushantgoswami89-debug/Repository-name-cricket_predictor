"""Read EVERY page Cricbuzz has for a match and list the data blocks each one carries.

Written 2026-10-10 after the scorecard was found to be a separate page (the match page only links to it),
which made the earlier page inventory wrong. For each page of a match this fetches the HTML, decodes the
Next.js flight data and lists the named JSON blocks (key, size, first-level fields), skipping UI plumbing.
Usage: python inspect_cricbuzz_pages.py <match_id> [--save]    (--save stores the raw pages, gzip)
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import requests

from app.live.cricbuzz_reader import _extract_flight_chunks

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CricketBaba/3.0)"}
ROOT = Path(__file__).resolve().parents[1]
DECODER = json.JSONDecoder()
UI_KEYS = {"children", "className", "props", "style", "href", "src", "alt", "id", "type", "key", "ref", "content",
           "title", "name", "value", "data", "dangerouslySetInnerHTML", "__html", "onClick", "width", "height", "rel"}
MIN_BYTES = 400


def match_pages(match_id: str) -> list[tuple[str, str]]:
    html = requests.get(f"https://www.cricbuzz.com/live-cricket-scores/{match_id}/x", headers=HEADERS, timeout=20).text
    links = sorted(set(re.findall(rf'href="(/[a-z-]+/{match_id}/[^"]*)"', html)))
    pages, seen = [], set()
    for href in links:
        kind = href.split("/")[1]
        if kind not in seen:
            seen.add(kind)
            pages.append((kind, "https://www.cricbuzz.com" + href))
    return pages


def blocks(html: str) -> dict[str, tuple[int, str]]:
    """key -> (largest size in bytes, first-level fields) for named JSON blocks in the flight data."""
    found: dict[str, tuple[int, str]] = {}
    for chunk in _extract_flight_chunks(html):
        for m in re.finditer(r'"([A-Za-z][A-Za-z0-9_]{2,40})":([\[{])', chunk):
            key = m.group(1)
            if key in UI_KEYS:
                continue
            try:
                value, end = DECODER.raw_decode(chunk[m.start(2):])
            except json.JSONDecodeError:
                continue
            size = end
            if size < MIN_BYTES:
                continue
            if isinstance(value, dict):
                fields = ",".join(list(value)[:8])
            elif value and isinstance(value[0], dict):
                fields = f"list[{len(value)}] of {{" + ",".join(list(value[0])[:8]) + "}"
            else:
                fields = f"list[{len(value)}]"
            if key not in found or size > found[key][0]:
                found[key] = (size, fields)
    return found


def main() -> None:
    match_id = sys.argv[1]
    save = "--save" in sys.argv
    for kind, url in match_pages(match_id):
        try:
            response = requests.get(url, headers=HEADERS, timeout=25)
        except requests.RequestException as error:
            print(f"\n## {kind}: fetch failed ({error})")
            continue
        found = blocks(response.text)
        print(f"\n## {kind}  ({response.status_code}, {len(response.text) // 1024} KB, {len(found)} data blocks)  {url}")
        for key, (size, fields) in sorted(found.items(), key=lambda kv: -kv[1][0])[:25]:
            print(f"   {key:<28} {size:>7} B  {fields[:110]}")
        if save:
            folder = ROOT / "data" / "raw" / "cricbuzz-pages" / match_id / "reference"
            folder.mkdir(parents=True, exist_ok=True)
            with gzip.open(folder / f"{kind}.html.gz", "wt", encoding="utf-8") as handle:
                handle.write(response.text)


if __name__ == "__main__":
    main()
