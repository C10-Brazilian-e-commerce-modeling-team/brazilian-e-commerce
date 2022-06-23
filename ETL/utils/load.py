from sqlalchemy import create_engine
from cfg import DB_CONNSTR

engine = create_engine(DB_CONNSTR)

class Loader:
    def load_table(self, name, df):
        df.to_sql(name, con=engine, index=False, if_exists="append")
