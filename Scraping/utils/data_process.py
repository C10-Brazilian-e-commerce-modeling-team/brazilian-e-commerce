import json
import pandas as pd


def read_categories(cat_name: str) -> json:

    with open("./reviews/{}.json".format(cat_name), "r") as f:
        cat_json = json.load(f)
        f.close()
    
    return cat_json


def features(reviews_category: json) -> pd.DataFrame:

    reviews = []
    for i in range(len(reviews_category)):
        score = int(reviews_category[i]["reviews"][0]["rating"])
        coment = reviews_category[i]["reviews"][0]["comment"]["content"]["text"]
        reviews.append([score, coment])

    return pd.DataFrame(reviews, columns=["score", "comment"])


def create_csv(df:pd.DataFrame, path:str):
    return df.to_csv(path)




