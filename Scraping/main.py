import pandas as pd
from utils.scraper import scraping_reviews
from utils.scraper import scraping_categories


def create_dataset(products: list) -> pd.DataFrame:
    try:
        list_reviews = []
        for product in products:
            reviews = scraping_reviews(product)
            print(reviews)
            if reviews["reviews"][0]["rating"] and reviews["reviews"][0]["title"]["text"] and reviews["reviews"][0]["comment"]["content"]["text"]:
                list_reviews.append([reviews["reviews"][0]["rating"], 
                                    reviews["reviews"][0]["title"]["text"], 
                                    reviews["reviews"][0]["comment"]["content"]["text"]]) # rating, title,  comment

        return pd.DataFrame(list_reviews)
    
    except Exception as e:
        print(e)
    


if __name__ == '__main__':
    
    category_url = "https://lista.mercadolivre.com.br/casa-moveis-decoracao/"
    products = scraping_categories(category_url)
    
    print(create_dataset(products))
