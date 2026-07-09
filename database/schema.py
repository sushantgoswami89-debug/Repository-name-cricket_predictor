"""
Prediction AI - SQLite Database Schema
Core database tables.
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    Date,
    ForeignKey,
)

from database.database import Base


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    short_name = Column(String)
    country = Column(String)


class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    batting_style = Column(String)
    bowling_style = Column(String)
    role = Column(String)


class Venue(Base):
    __tablename__ = "venues"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    city = Column(String)
    country = Column(String)


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    cricsheet_id = Column(String, unique=True)

    match_date = Column(Date)
    season = Column(String)
    format = Column(String)

    venue_id = Column(Integer, ForeignKey("venues.id"))

    team1_id = Column(Integer, ForeignKey("teams.id"))
    team2_id = Column(Integer, ForeignKey("teams.id"))

    toss_winner_id = Column(Integer, ForeignKey("teams.id"))
    toss_decision = Column(String)

    winner_id = Column(Integer, ForeignKey("teams.id"))

    result = Column(String)


class Innings(Base):
    __tablename__ = "innings"

    id = Column(Integer, primary_key=True)

    match_id = Column(Integer, ForeignKey("matches.id"))

    innings_number = Column(Integer)

    batting_team_id = Column(Integer, ForeignKey("teams.id"))
    bowling_team_id = Column(Integer, ForeignKey("teams.id"))


class Over(Base):
    __tablename__ = "overs"

    id = Column(Integer, primary_key=True)

    innings_id = Column(Integer, ForeignKey("innings.id"))

    over_number = Column(Integer)

    runs = Column(Integer)

    wickets = Column(Integer)


class Ball(Base):
    __tablename__ = "balls"

    id = Column(Integer, primary_key=True)

    innings_id = Column(Integer, ForeignKey("innings.id"))

    over_number = Column(Integer)

    ball_number = Column(Float)

    striker_id = Column(Integer, ForeignKey("players.id"))
    non_striker_id = Column(Integer, ForeignKey("players.id"))
    bowler_id = Column(Integer, ForeignKey("players.id"))

    batter_runs = Column(Integer)
    extras = Column(Integer)
    total_runs = Column(Integer)

    wicket = Column(Boolean)

    dismissal_type = Column(String)