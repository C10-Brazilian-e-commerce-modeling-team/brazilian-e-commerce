import re
import json
import requests
import lxml.html as html


def scraping_categories(category_url: str) -> list:

    try:
        all_links = []
        # while category_url:
        response = requests.get(category_url)
        if response.status_code == 200:
            content = response.content.decode('utf-8')
            parsed = html.fromstring(content)
            links_products = parsed.xpath('//*[@id="root-app"]/div/div/section/ol/li/div/div/a/@href')
            all_links += links_products
            #next_page = parsed.xpath('//li[@class="andes-pagination__button andes-pagination__button--next"]/a/@href')
            # category_url = next_page[0]
        print(len(all_links))
        return all_links

    except Exception as e:
        print(e)


def get_reviews(request) -> json:
    
    product_content = request.content.decode('utf-8')
    product_parsed = html.fromstring(product_content)
    reviews = product_parsed.xpath('/html/body/p/text()')
    reviews_json = json.loads(reviews[0])
    
    return reviews_json


def scraping_reviews(url_product: str) -> dict:
    
    page_reviews = 'https://www.mercadolivre.com.br/noindex/catalog/reviews/MLB{}/scroll?siteId=MLB&type=all&isItem=true&offset={}&limit=1'
    altern_page_reviews = 'https://www.mercadolivre.com.br/noindex/catalog/reviews/MLB{}/scroll?siteId=MLB&type=all&isItem=false&offset={}&limit=1'
    
    try:
        counter = 0
        product = url_product
        product_id = re.search("/MLB-?(\d{4,})", product)
        while True:
            
            product_reviews = page_reviews.format(product_id.group(1), counter)
            product_request = requests.get(product_reviews)

            if product_request.status_code != 200:
                product_reviews = altern_page_reviews.format(product_id.group(1), counter)
                product_request = requests.get(product_reviews)
                reviews = get_reviews(product_request)
                print(reviews)
            
                if len(reviews["reviews"][0]) == 1:
                    break
                else:
                    return reviews

            else:
                product_request = requests.get(product_reviews)
                reviews = get_reviews(product_request)
                print(reviews)

                if len(reviews["reviews"][0]) == 1:
                    break
                else:
                    return reviews
            
        counter += 1
        print("Reviews scraped: ".format(counter))
        
    except:
        next


