"""
Database Configuration
=======================
SQLAlchemy engine, session factory, and base model.
Supports SQLite (dev) and PostgreSQL (prod) via DATABASE_URL.
"""

import os
import logging
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase

logger = logging.getLogger(__name__)

# Load DATABASE_URL from environment or .env file
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./quant_engine.db")

# SQLite needs special handling for foreign keys
is_sqlite = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if is_sqlite else {},
    pool_pre_ping=True,
)

# Enable foreign key enforcement for SQLite
if is_sqlite:
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


def get_db():
    """FastAPI dependency — yields a DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables (for dev — use Alembic for prod migrations)."""
    from db.models import User, Portfolio, Trade, Alert, AuditLog  # noqa
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database initialized: {DATABASE_URL}")
