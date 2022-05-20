import json
import pandas as pd
from utils.scraper import scraping_reviews
from utils.scraper import scraping_categories


def create_dataset(products: list, route: str) -> dict:
    try:
        dict_reviews = []
        for product in products:
            reviews = scraping_reviews(product)
            # print(reviews)
            dict_reviews.append(reviews)

        with open(route, "w") as f:
            json.dump(dict_reviews, f)
            f.close()

        return "Json file created"
    
    except Exception as e:
        print(e)
    


if __name__ == '__main__':
    
    category_url = "https://lista.mercadolivre.com.br/casa-moveis-decoracao/"
    products = scraping_categories(category_url)
    
    print(create_dataset(products, "./reviews/first_test.json"))
