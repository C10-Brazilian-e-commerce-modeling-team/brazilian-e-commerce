# Brazilian E-Commerce Modeling Team: Project Technology

👋🏼 Greatings! at this file you will find al the technologies we chose to face our first Platzi Master Squad Project 🚀.

## Table of Contents
1. [DataBases](#databases)
2. [Data Analysis](#dataanalysis)
3. [Data Viz](#dataviz)
4. [Cloud](#cloud)

## Components and Technologies: 

### 🗄 DataBases: How to organize the information and it's interaction. <a name="databases"></a>
#### 📁 Given Data
- Kaggle Data: PostgreSQL.
All the data is processed and loaded with PostgreSQL to the database server.

#### 🕷 External data Extraction: New sources of data to enrich the model.
- Web Scrapping: we'll use Selenium, Scrapy and BeautifulSoup to create scripts to optain external information in order to relate with the given data and enrich the data analysis.

### 🎛 Entity Relation (ER) Diagram 
For the Kaggle's Given Data we create the ER Diagram ilustrated in this graph:
![Entity Relation Diagram](https://user-images.githubusercontent.com/69657612/161407546-1049e279-b928-47a4-8843-b057c0c8198c.jpeg)

#### **Changes**
In order to have a clean and better practice of performance, we have made the next changes:
- All tables and its columns were renamed to avoid redundancies.
- All indexes are transformed to type int to improve query performance.

#### **olist_customers_dataset ->  customers**
- Remove the customer_id column to normalize the table.
- Convert customer_unique_id column to id column of type int.
- Add the name and last_name columns to represent a more realistic environment.

#### **olist_order_items_dataset ->  order_items**
- Remove order_item_id column and add quantity column for normalization.
- All values in the freight_value column should be updated by quantity * old value of freight_value.
 
#### 🔩**Database Enhancement**
- Replace the name_length and description_length columns with name and description respectively in the products table and generate random values to simulate a more realistic environment.
- Aggregate total_column for each order.



### 👩🏼‍💻 Data Analysis: Understanding what our data have to say. <a name="dataanalysis"></a>
- **Language:** Python
- **Libraries:** Numpy, Pandas, matplotlib, seaborn, SkLearn.

### **📊 Data Visualizations: Dashboard of results and database transactions** <a name="dataviz"></a>
We need data visualization providers for the data analysis results and the project's requirement of the summary of transactions of the database.
- Google Data Studio
- PowerBI

### ☁ Cloud Provider: Where our project will live. <a name="cloud"></a>
In order to have online our project with all it's features we will use the free tier of AWS services for the database hosting, its connection for the data analysis queries and dashboard visualizations.

## ETL Pipeline

![](https://drive.google.com/uc?id=1kcFl_Qz_K1WvuMY0wPUoA51tfOWEIv58)

[Watch our interactive Data Model and Data Pipeline](https://miro.com/app/board/uXjVO-rg3Ls=/)

