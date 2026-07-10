"""
Context Builder.

Responsible for constructing a fully validated and enriched
MatchContext from raw match data.
"""

from __future__ import annotations

import logging
from typing import Any

from app.models.match_context import MatchContext
from app.services.match_context_service import MatchContextService

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    High-level builder responsible for creating MatchContext objects.

    This class acts as the single entry point for constructing
    a complete MatchContext from raw match information.
    """

    @staticmethod
    def build(match_data: dict[str, Any]) -> MatchContext:
        """
        Build a complete MatchContext.

        Parameters
        ----------
        match_data : dict[str, Any]
            Raw match information.

        Returns
        -------
        MatchContext
            Fully validated and enriched MatchContext.
        """

        logger.info("Building complete MatchContext")

        context = MatchContextService.build(match_data)

        logger.info("ContextBuilder completed successfully")

        return context
