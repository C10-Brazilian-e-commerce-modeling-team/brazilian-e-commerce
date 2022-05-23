import json
import pandas as pd
from utils.scraper import scraping_reviews
from utils.scraper import scraping_products


def create_dataset(products: list, path: str) -> dict:
    try:
        dict_reviews = []
        for product in products:
            reviews = scraping_reviews(product)
            # print(reviews)
            dict_reviews += reviews

        with open(path, "w") as f:
            json.dump(dict_reviews, f)
            f.close()

        return "Json file created"
    
    except Exception as e:
        print(e)
    


if __name__ == '__main__':

    category_url = "https://lista.mercadolivre.com.br/casa-moveis-decoracao/_Deal_casaedecoracao-decor#deal_print_id=e97f30c0-d00a-11ec-88f8-6dbc6c3a0c56&c_id=special-normal&c_element_order=6&c_campaign=LABEL&c_uid=e97f30c0-d00a-11ec-88f8-6dbc6c3a0c56"
    products = scraping_products(category_url)
    print("products found: ", len(products))
    print(create_dataset(products, "./reviews/moveis_decoracao.json"))

    # with open('categories/brazil.json', 'r') as f:
    #     categories = json.load(f)
    #     f.close()
    
    # for key in categories:
    #     category = categories[key]
    #     products = scraping_products(category)
    #     print("Category: {} found {} articles".format(key, len(products)))
    #     print(create_dataset(products, "./reviews/{}.json".format(key)))



