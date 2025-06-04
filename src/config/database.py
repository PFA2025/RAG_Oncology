from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Database configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up to project root
DB_DIR = PROJECT_ROOT / 'data'
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / 'user_memories.db'
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH.absolute()}"

# Create database engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Dependency to get DB session
def get_db():
    """Dependency that provides a DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
