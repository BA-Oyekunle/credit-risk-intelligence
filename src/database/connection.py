import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import declarative_base, sessionmaker


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

# Render provides DATABASE_URL.
# Local development uses the individual PG* variables.

render_database_url = os.getenv("DATABASE_URL")

if render_database_url:

    # Parse the supplied URL and ensure SQLAlchemy uses
    # the psycopg 3 driver installed in requirements.txt.
    DATABASE_URL = make_url(render_database_url).set(
        drivername="postgresql+psycopg"
    )

else:

    # Local PostgreSQL configuration.
    DATABASE_URL = URL.create(
        drivername="postgresql+psycopg",
        username=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        database=os.getenv("PGDATABASE"),
    )


# ============================================================
# VALIDATE DATABASE CONFIGURATION
# ============================================================

if not DATABASE_URL.username:
    raise ValueError(
        "Database username is missing. "
        "Configure DATABASE_URL or PGUSER."
    )

if not DATABASE_URL.password:
    raise ValueError(
        "Database password is missing. "
        "Configure DATABASE_URL or PGPASSWORD."
    )

if not DATABASE_URL.database:
    raise ValueError(
        "Database name is missing. "
        "Configure DATABASE_URL or PGDATABASE."
    )


# ============================================================
# SQLALCHEMY ENGINE
# ============================================================

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=False,
)


# ============================================================
# SESSION CONFIGURATION
# ============================================================

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


# ============================================================
# DECLARATIVE BASE
# ============================================================

Base = declarative_base()


# ============================================================
# DATABASE SESSION DEPENDENCY
# ============================================================

def get_db():
    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()
