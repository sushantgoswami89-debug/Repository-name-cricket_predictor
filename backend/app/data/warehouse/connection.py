"""
CricketBaba Data Warehouse Connection.

Provides a singleton SQLite connection for the CricketBaba
Data Warehouse.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class WarehouseConnection:
    """
    Singleton SQLite warehouse connection.
    """

    _connection: sqlite3.Connection | None = None

    @classmethod
    def get_connection(cls) -> sqlite3.Connection:
        if cls._connection is None:
            project_root = Path(__file__).resolve().parents[4]

            warehouse_dir = project_root / "data" / "warehouse"

            warehouse_dir.mkdir(parents=True, exist_ok=True)

            database_path = warehouse_dir / "cricketbaba.db"

            logger.info(
                "Opening warehouse database: %s",
                database_path,
            )

            cls._connection = sqlite3.connect(database_path)

            cls._connection.row_factory = sqlite3.Row

        return cls._connection

    @classmethod
    def close(cls) -> None:
        if cls._connection is not None:
            logger.info("Closing warehouse connection")

            cls._connection.close()

            cls._connection = None
