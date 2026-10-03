import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy Engine supporting both PostgreSQL and SQLite
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_all_tables():
    """Create all configured database tables."""
    Base.metadata.create_all(bind=engine)

def check_database_connection() -> dict:
    """Test database connectivity for health checks."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "connected", "type": "postgresql" if "postgresql" in settings.DATABASE_URL else "sqlite"}
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return {"status": "disconnected", "error": str(e)}
