from utils.cfg import RAW_DATA_URL_DICT
from utils.extract import Extract_csv, Extract_enrich_order
from utils.load import Loader
import pandas as pd
from utils.constants import TABLE_NAMES


extract_dict = {
    'customers': Extract_csv('customers', RAW_DATA_URL_DICT['customers']),
    'geolocation': Extract_csv('geolocation', RAW_DATA_URL_DICT['geolocation']),
    'order_items': Extract_csv('order_items', RAW_DATA_URL_DICT['order_items']),
    'order_reviews': Extract_csv('order_reviews', RAW_DATA_URL_DICT['order_reviews']),
    'orders': Extract_enrich_order('orders', RAW_DATA_URL_DICT['orders']),
    'payments': Extract_csv('payments', RAW_DATA_URL_DICT['payments']),
    'products': Extract_csv('products', RAW_DATA_URL_DICT['products']),
    'sellers': Extract_csv('sellers', RAW_DATA_URL_DICT['sellers'])
}


def extract_raws() -> list[str]:
    """Extract raw data from url and return list of paths

    Returns:
        list[str]: list of stored data file paths
    """
    file_paths = dict()
    for name, extractor in extract_dict.items():
        file_path = extractor.extract()
        df = pd.read_csv(file_path)
        file_path = extractor.tranform(df)
        file_paths[name] = file_path
    return file_paths

def transform():
    pass

def load():
    load_ = Loader()

    for name in TABLE_NAMES:
        path = f'data/{name}.csv'
        df = pd.read_csv(path)
        load_.load_table(name, df)
    

def run_pipeline():
    # Extract raw data and store path in file_paths dict
    file_paths = extract_raws()
    print('Extract done!')

    load()
    print('Load Done!')

if __name__ == "__main__":

    run_pipeline()

