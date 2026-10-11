"""
DeepVision AI — Database Engine & Session Management
SQLAlchemy 2.0 foundation supporting PostgreSQL with connection pooling
and graceful SQLite fallback for local development.
"""

import time
import logging
from typing import Generator, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings
from app.models.base import Base

logger = logging.getLogger("deepvision.database")


def get_database_url() -> str:
    """
    Normalizes the database URL.
    Ensures PostgreSQL URLs use psycopg 3 (postgresql+psycopg://).
    """
    import os
    raw_url = os.environ.get("DATABASE_URL") or settings.DATABASE_URL or "sqlite:///./deepvision.db"
    
    # Automatically map postgresql:// or postgres:// to modern psycopg 3 dialect
    if raw_url.startswith("postgresql://"):
        return raw_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if raw_url.startswith("postgres://"):
        return raw_url.replace("postgres://", "postgresql+psycopg://", 1)
        
    return raw_url


DATABASE_URL = get_database_url()
is_sqlite = DATABASE_URL.startswith("sqlite")

# Engine options configured for enterprise PostgreSQL workloads
engine_kwargs: Dict[str, Any] = {
    "pool_pre_ping": True,
}

if is_sqlite:
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # PostgreSQL pooling parameters
    engine_kwargs.update({
        "pool_size": 10,
        "max_overflow": 20,
        "pool_recycle": 1800,
        "pool_timeout": 30,
    })

engine = create_engine(DATABASE_URL, **engine_kwargs)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding scoped database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initializes database tables.
    Called on application startup. Safe across concurrent workers.
    """
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
        logger.info("Database metadata initialized successfully (Dialect: %s)", engine.dialect.name)
    except Exception as exc:
        # If running with multiple workers (e.g. workers=2 on Render) or tables exist
        err_msg = str(exc).lower()
        if "already exists" in err_msg:
            logger.info("Database tables already initialized: %s", exc)
        else:
            logger.warning("Database initialization note (non-fatal): %s", exc)


def check_db_connection() -> Dict[str, Any]:
    """
    Tests database connectivity and latency.
    Useful for health checks and readiness probes.
    """
    start_time = time.perf_counter()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "status": "connected",
            "dialect": engine.dialect.name,
            "latency_ms": latency_ms
        }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.warning("Database connectivity check failed: %s", exc)
        return {
            "status": "disconnected",
            "dialect": engine.dialect.name,
            "error": str(exc),
            "latency_ms": latency_ms
        }
