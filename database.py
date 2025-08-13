from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base
import os

#DATABASE_URL = "sqlite:///./chat.db"
DATABASE_URL = "sqlite:////data/users_v3.db"
# print("Current working directory:", os.getcwd())
# print("Resolved DB path:", os.path.abspath(DATABASE_URL.replace("sqlite:///", "")))


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
    db_path = DATABASE_URL.replace("sqlite:///", "")
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(users);")
        columns = cursor.fetchall()
        print("\n[DB SCHEMA] Users table columns:")
        for col in columns:
            print(f" - {col[1]} ({col[2]})")  # col[1]=name, col[2]=type
        conn.close()
    else:
        print(f"[DB SCHEMA] Database file not found at {db_path}")

