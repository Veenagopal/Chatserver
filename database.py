from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

#DATABASE_URL = "sqlite:///./chat.db"
DATABASE_URL = "sqlite:////data/users_v3.db"
# print("Current working directory:", os.getcwd())
# print("Resolved DB path:", os.path.abspath(DATABASE_URL.replace("sqlite:///", "")))


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
