# import os
# #import sqlite3
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from models import Base

# # SQLite database file location
# #DATABASE_URL = "sqlite:////data/users_v3.db"
# DATABASE_URL = os.getenv(
#     "DATABASE_URL",
#     "postgresql+psycopg2://neondb_owner:npg_vjxLQFyk89fD@ep-super-glitter-adzibbu2-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
# )



# # SQLAlchemy engine & session
# #engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
# engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# def init_db():
#     """Create all tables and print Users table schema."""
#     Base.metadata.create_all(bind=engine)
#     db_path = DATABASE_URL.replace("sqlite:///", "")
#     if os.path.exists(db_path):
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()
#         cursor.execute("PRAGMA table_info(users);")
#         columns = cursor.fetchall()
#         print("\n[DB SCHEMA] Users table columns:")
#         for col in columns:
#             print(f" - {col[1]} ({col[2]})")
#         conn.close()
#     else:
#         print(f"[DB SCHEMA] Database file not found at {db_path}")

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

# ------------------ DATABASE URL ------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://neondb_owner:npg_vjxLQFyk89fD@ep-super-glitter-adzibbu2-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
)

# ------------------ ENGINE & SESSION ------------------
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True  # prevents stale connection issues
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# ------------------ INIT FUNCTION ------------------
def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    print("[INFO] Database tables created/verified.")

# ------------------ DEPENDENCY ------------------
def get_db():
    """FastAPI dependency to get DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

