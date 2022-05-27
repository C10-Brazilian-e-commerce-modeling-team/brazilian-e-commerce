import re
import json
import requests
import lxml.html as html


def scraping_categories(category_url: str) -> list:
    """
    Web Scraping to Mercado Libre categories

    Parameters
    ----------
    category_url: str
        Url from a specific category, its contains products list.

    Returns
    ---------
        all_links: list
            List with all products of category
    """

    all_links = []
    try:
        while category_url:
            # Make request to the server
            response = requests.get(category_url)
            if response.status_code == 200:
                content = response.content.decode('utf-8')
                parsed = html.fromstring(content)
                # Scrap products link
                links_products = parsed.xpath('//*[@id="root-app"]/div/div/section/ol/li/div/div/a/@href')
                if len(links_products) == 0:
                    # Alrtern Xpath in someones categories
                    links_products = parsed.xpath('//*[@id="root-app"]/div/div/section/ol/li/div/div/div/a/@href')
                all_links += links_products
                # Xpath loop for scrape next page
                next_page = parsed.xpath('//li[@class="andes-pagination__button andes-pagination__button--next"]/a/@href')
                category_url = next_page[0]

    except Exception as e:
        print("Products scraped")
        print("products found:", len(all_links))

    return all_links



def get_reviews(request: requests.models.Response) -> json:
    """
    This function scrape the reviews of some product

    Parameters:
    ----------
    request: requests.models.Response
        Response to HTTP request

    Returns:
    ----------
    reviews_json: json
        Dict with the reviews of the product in json format
    """
    # Make request to the server
    product_content = request.content.decode('utf-8')
    product_parsed = html.fromstring(product_content)
    reviews = product_parsed.xpath('/html/body/p/text()')
    reviews_json = json.loads(reviews[0])
    
    return reviews_json


def scraping_reviews(url_product: str) -> list:
    """
    This function iterate and found all reviews of a product

    Parameter:
    ----------
    url_product: str
        Url of the product wich scrape reviews

    Returns:
    ----------
    list_reviews: list
        The function returns a list with all reviews of a product
    """
    
    # Exist two links option for get reviews
    page_reviews = 'https://www.mercadolivre.com.br/noindex/catalog/reviews/MLB{}/scroll?siteId=MLB&type=all&isItem=true&offset={}&limit=1'
    altern_page_reviews = 'https://www.mercadolivre.com.br/noindex/catalog/reviews/MLB{}/scroll?siteId=MLB&type=all&isItem=false&offset={}&limit=1'
    
    list_reviews = []   # Define empty list for storage the reviews
    
    try:
        counter = 0   # Counter for increase the number of reviews
        product = url_product
        # Found the code of the prooduct for with Regex
        product_id = re.search("/MLB-?(\d{4,})", product) 
        print(product_id)
        # Loop for get each review
        while True:
            
            product_reviews = page_reviews.format(product_id.group(1), counter)
            product_request = requests.get(product_reviews)
            # This code block decide use between the two links option
            if product_request.status_code != 200:
                product_reviews = altern_page_reviews.format(product_id.group(1), counter)
                product_request = requests.get(product_reviews)
                reviews = get_reviews(product_request)
                print(reviews)
                # No append products without reviews
                if len(reviews["reviews"][0]) == 1 and reviews["message"]:
                    print("No reviews")
                # Break the loop when its gets at the end
                elif len(reviews["reviews"][0]) == 1:
                    break
                else:
                    list_reviews.append(reviews)

            else:
                product_request = requests.get(product_reviews)
                reviews = get_reviews(product_request)
                print(reviews)
                # No append products without reviews
                if len(reviews["reviews"][0]) == 1 and reviews["message"]:
                    print("No reviews")
                # Break the loop when its gets at the end
                elif len(reviews["reviews"][0]) == 1:
                    break
                else:
                    list_reviews.append(reviews)

            counter += 1
            print(f"Scraping review no.{counter}")

        
    except Exception as e:
        print(e)


    return list_reviews


