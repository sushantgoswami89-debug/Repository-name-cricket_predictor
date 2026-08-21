from .database import Base, engine
from .schema import (
    Team,
    Player,
    Venue,
    Match,
    Innings,
    Over,
    Ball,
)


def create_database():
    """
    Create all database tables.
    """
    Base.metadata.create_all(bind=engine)
