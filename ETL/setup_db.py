from sqlalchemy import create_engine
from sqlalchemy.sql import text
from utils.constants import SQL_DIR, TABLE_NAMES
from utils.cfg import DB_CONNSTR


engine = create_engine(DB_CONNSTR)


def create_tables():
    """Create db tables"""
    with engine.connect() as con:
        for file in TABLE_NAMES:
            with open(SQL_DIR / f"{file}.sql") as f:
                query = text(f.read())
            con.execute(query)


if __name__ == "__main__":
    create_tables()
    print("Tables created")
    