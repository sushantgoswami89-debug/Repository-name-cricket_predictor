"""
Venue Service.

Provides venue lookup functionality.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from app.data.venue_registry import VENUE_REGISTRY

logger = logging.getLogger(__name__)


class VenueService:
    """Venue lookup service."""

    @staticmethod
    def get_venue(name: str) -> Dict[str, Any]:
        """
        Retrieve venue information.

        Parameters
        ----------
        name : str
            Venue name.

        Returns
        -------
        Dict[str, Any]
            Venue information.

        Raises
        ------
        ValueError
            If venue does not exist.
        """

        logger.info("Looking up venue '%s'", name)

        if name not in VENUE_REGISTRY:
            raise ValueError(f"Unknown venue: {name}")

        return VENUE_REGISTRY[name]
