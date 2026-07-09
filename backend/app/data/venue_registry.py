"""
Venue Registry.

Stores static venue information used by the Match Context Engine.
"""

from __future__ import annotations

from typing import Dict

VENUE_REGISTRY: Dict[str, Dict[str, object]] = {
    "MCG": {
        "full_name": "Melbourne Cricket Ground",
        "city": "Melbourne",
        "country": "Australia",
        "pitch_type": "Balanced",
        "average_first_innings": 292,
        "average_second_innings": 247,
        "boundary_size": "Large",
        "dew": "Medium",
        "timezone": "Australia/Melbourne",
    },
    "Eden Gardens": {
        "full_name": "Eden Gardens",
        "city": "Kolkata",
        "country": "India",
        "pitch_type": "Batting",
        "average_first_innings": 287,
        "average_second_innings": 241,
        "boundary_size": "Medium",
        "dew": "High",
        "timezone": "Asia/Kolkata",
    },
    "Wankhede": {
        "full_name": "Wankhede Stadium",
        "city": "Mumbai",
        "country": "India",
        "pitch_type": "Batting",
        "average_first_innings": 301,
        "average_second_innings": 268,
        "boundary_size": "Small",
        "dew": "High",
        "timezone": "Asia/Kolkata",
    },
}
