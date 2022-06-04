import json
import random
import pandas as pd
from utils.scraper import scraping_reviews, scraping_categories
from utils.data_process import read_categories, features, create_csv


def create_dataset(products: list, path: str) -> dict:
    
    """
    This script create a json format dataset of a list of products.
    
    Paramters
    ----------
    products: list
        List of products to scrape reviews
    path: str
        Path to save the created file

    Returns:
    ----------
        Json file with all products reviews in the specified dir of path
    """
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

    """
    Script for scrape just once category

    category_url = "category url"
    products = scraping_categories(category_url)
    print(create_dataset(products, path))
    """

    with open('categories/brazil.json', 'r') as f:
        categories = json.load(f)
        f.close()
    
    # This block execute the code for scrape all reviews in the variable "categories"
    # Returns a json files for each category
    for key in categories:
        category = categories[key]
        products = scraping_categories(category)
        print("Category: {} found {} articles".format(key, len(products)))
        print(create_dataset(products, "./reviews/{}.json".format(key)))
    
    # This block takes reviews and ranking of products from json files 
    # create .csv file with reviews for each category
    for key in categories:
        reviews_json = read_categories(key)
        df_reviews = features(reviews_json)
        print(f"Category {key}: {len(df_reviews)}")
        create_csv(df=df_reviews, path='./External_data/{}.csv'.format(key))
        print("csv created")






