"""
Prediction AI - Database Connection

Creates and manages the SQLite database for CricketBaba.
"""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Database directory
DB_DIR = PROJECT_ROOT / "data"

# Create directory if it doesn't exist
DB_DIR.mkdir(parents=True, exist_ok=True)

# SQLite database file
DATABASE_PATH = DB_DIR / "cricketbaba.db"

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# SQLAlchemy Engine
engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
)

# Session Factory
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

# Base class for all tables
Base = declarative_base()


def get_db():
    """
    Returns a database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
