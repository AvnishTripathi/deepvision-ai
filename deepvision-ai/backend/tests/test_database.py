"""
Tests for SQLAlchemy Database Connectivity and Engine Setup.
"""

import os
from sqlalchemy import text
from app.core.database import (
    init_db,
    check_db_connection,
    get_db,
    get_database_url,
    engine,
)
from app.models.base import Base


def test_database_initialization():
    """Verify init_db initializes metadata without errors."""
    init_db()
    assert isinstance(Base.metadata.tables, dict)


def test_database_connection_probe():
    """Verify check_db_connection returns connected status and latency."""
    res = check_db_connection()
    assert res["status"] == "connected"
    assert "latency_ms" in res
    assert res["latency_ms"] >= 0


def test_session_lifecycle():
    """Verify get_db dependency yields a valid session and executes queries."""
    db_gen = get_db()
    session = next(db_gen)
    try:
        val = session.execute(text("SELECT 1")).scalar()
        assert val == 1
    finally:
        session.close()


def test_postgresql_url_mapping(monkeypatch):
    """Verify postgresql:// URLs automatically map to modern psycopg 3 dialect."""
    test_pg_url = "postgresql://user:secret@localhost:5432/testdb"
    monkeypatch.setenv("DATABASE_URL", test_pg_url)
    
    mapped = get_database_url()
    assert mapped == "postgresql+psycopg://user:secret@localhost:5432/testdb"
