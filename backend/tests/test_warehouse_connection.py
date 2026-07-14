"""
Tests for WarehouseConnection.
"""

from app.data.warehouse.connection import WarehouseConnection


def test_connection() -> None:
    """
    Ensure warehouse connection opens.
    """

    connection = WarehouseConnection.get_connection()

    assert connection is not None

    WarehouseConnection.close()
