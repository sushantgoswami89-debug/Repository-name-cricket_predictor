"""
CricketBaba JSON Loader.

Loads Cricsheet JSON files into memory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JsonLoader:
    """
    Loads cricket JSON files.
    """

    def load(
        self,
        path: str | Path,
    ) -> dict[str, Any]:
        """
        Load a JSON file.

        Parameters
        ----------
        path
            Path to the JSON file.

        Returns
        -------
        dict[str, Any]
            Parsed JSON.
        """

        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        logger.info("Loading JSON file: %s", file_path)

        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)
