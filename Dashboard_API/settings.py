from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, create_database
import logging

import sys
sys.path.append("../Data_analysis")
from local_settings import postgresql as settings


log = logging.getLogger(__name__)

def get_engine(user, passwd, host, port, db):
    url = f"postgresql://{user}:{passwd}@{host}:{port}/{db}"
    if not database_exists(url):
        create_database(url)
    engine = create_engine(url, pool_size=50, echo=False)
    return engine

engine = get_engine(settings['pguser'],
                    settings['pgpasswd'],
                    settings['pghost'],
                    settings['pgport'],
                    settings['pgdb'])
print(engine.url)