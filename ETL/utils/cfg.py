from decouple import AutoConfig
from constants import ROOT_DIR

config = AutoConfig(search_path=ROOT_DIR)

USER = config('USER')
PASSWORD = config('PASSWORD')
HOST = config('HOST')
PORT = config('PORT')
DATABASE = config('DATABASE')

DB_CONNSTR = f'postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}'

CUSTOMERS_RAW_DATA_URL = config('CUSTOMERS_RAW_DATA_URL')
GEOLOCATION_RAW_DATA_URL = config('GEOLOCATION_RAW_DATA_URL')
ORDER_ITEMS_RAW_DATA_URL = config('ORDER_ITEMS_RAW_DATA_URL')
ORDER_REVIEWS_RAW_DATA_URL = config('ORDER_REVIEWS_RAW_DATA_URL')
ORDERS_RAW_DATA_URL = config('ORDERS_RAW_DATA_URL')
PAYMENTS_RAW_DATA_URL = config('PAYMENTS_RAW_DATA_URL')
PRODUCTS_RAW_DATA_URL = config('PRODUCTS_RAW_DATA_URL')
SELLERS_RAW_DATA_URL = config('SELLERS_RAW_DATA_URL')

RAW_DATA_URL_DICT = {
    'customers': CUSTOMERS_RAW_DATA_URL,
    'geolocation': GEOLOCATION_RAW_DATA_URL,
    'order_items': ORDER_ITEMS_RAW_DATA_URL,
    'order_reviews': ORDER_REVIEWS_RAW_DATA_URL,
    'orders': ORDERS_RAW_DATA_URL,
    'payments': PAYMENTS_RAW_DATA_URL,
    'products': PRODUCTS_RAW_DATA_URL,
    'sellers': SELLERS_RAW_DATA_URL,
}


