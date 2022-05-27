import json
import random
import pandas as pd
from utils.scraper import scraping_reviews
from utils.scraper import scraping_categories


def create_dataset(products: list, path: str) -> dict:
    try:
        dict_reviews = []
        idxs = [random.randrange(0, len(products)) for _ in range(110)]
        for idx in idxs:
            reviews = scraping_reviews(products[idx])
            # print(reviews)
            dict_reviews += reviews 

        with open(path, "w") as f:
            json.dump(dict_reviews, f)
            f.close()

        return "Json file created"
    
    except Exception as e:
        print(e)
    


if __name__ == '__main__':

    # category_url = "https://lista.mercadolivre.com.br/casa-moveis-decoracao/_Deal_casaedecoracao-decor#deal_print_id=e97f30c0-d00a-11ec-88f8-6dbc6c3a0c56&c_id=special-normal&c_element_order=6&c_campaign=LABEL&c_uid=e97f30c0-d00a-11ec-88f8-6dbc6c3a0c56"
    # products = scraping_categories(category_url)
    # print(create_dataset(products, "./reviews/beleza_saude.json"))

    with open('categories/brazil.json', 'r') as f:
        categories = json.load(f)
        f.close()
    
    for key in categories:
        category = categories[key]
        products = scraping_categories(category)
        print("Category: {} found {} articles".format(key, len(products)))
        print(create_dataset(products, "./reviews/{}.json".format(key)))



