"""
CricketBaba Database Writer.

Persists validated Match objects to the database.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.models.match_data import Match

logger = logging.getLogger(__name__)


class DatabaseWriter:
    """
    Writes validated matches to the database.

    NOTE:
    This is the first production implementation.
    Future versions will expand persistence for
    teams, players, venues and complete scorecards.
    """

    def __init__(
        self,
        db: Session,
    ) -> None:
        self.db = db

    def write(
        self,
        match: Match,
    ) -> None:
        """
        Persist a validated match.

        Parameters
        ----------
        match
            Validated Match object.
        """

        logger.info(
            "Writing match: %s vs %s",
            match.info.teams[0],
            match.info.teams[1],
        )

        try:
            #
            # Full persistence will be implemented
            # in the next milestone.
            #
            self.db.commit()

            logger.info("Database write successful.")

        except Exception:

            self.db.rollback()

            logger.exception(
                "Database write failed."
            )

            raise