#data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px


#dashboard
import streamlit as st
import plost
from PIL import Image

# Page setting
st.set_page_config(page_title="Brazilian E-Commerce Dashboard",page_icon="📊",layout="wide")

# create the dataframes with the given csv files
# Using a dict structure to store and generate the dataframes
file_name ={'orders':'orders', 'order_items':'order_items', 'products':'products',
            'product_category_english':'product_category_name_translation',
            'customers': 'customers','sellers': 'sellers',
            'payments':'payments', 'order_reviews':'order_reviews', 'geolocation':'geolocation'}
for name in file_name.keys():
    locals()[name] = pd.read_csv('Data_analysis/datasets/'+file_name[name]+'.csv')

dataset_list = file_name.keys()

# let's check the columns of the dataframes
df_columns_name = pd.DataFrame([globals()[i].columns for i in dataset_list], index = dataset_list).T

def time_periods(x):
    if x>=5 and x<12:
        return "Morning"
    elif x>=12 and x<17:
        return "Afternoon"
    elif x>=17 and x<21:
        return "Evening"
    else:
        return "Night"

# clean the dataset with the neccessary columns:
orders_df = orders[['order_id','customer_id','order_status','order_purchase_timestamp','order_estimated_delivery_date',
                    'order_delivered_customer_date','order_delivered_carrier_date']]

# add the date columns
orders_df = orders_df.assign(
    purchase_date = pd.to_datetime(orders_df['order_purchase_timestamp']).dt.date,
    purchase_year = pd.to_datetime(orders_df['order_purchase_timestamp']).dt.year,
    purchase_month = pd.to_datetime(orders_df['order_purchase_timestamp']).dt.month,
    purchase_MMYYYY= pd.to_datetime(orders_df['order_purchase_timestamp']).dt.strftime('%b-%y'),
    purchase_day = pd.to_datetime(orders_df['order_purchase_timestamp']).dt.day_name(),
    purchase_hour = pd.to_datetime(orders_df['order_purchase_timestamp']).dt.hour)

#add the time_periods
orders_df["purchase_time"]= orders_df["purchase_hour"].apply(time_periods)


###calculate %orders by status for each year###

# Pivot table for counting orders by status and year
ord_sy = orders_df.pivot_table(values = 'order_id', index='order_status'
                                , columns='purchase_year', aggfunc= 'count')
ord_sy.fillna(0, inplace=True) # replace NaN with 0

# Add the percentage of order by status for each year:
ord_sy = ord_sy.apply(lambda x: ((x*100)/x.sum()).round(2), axis=0).T

# Show the percentage of orders by status for each year:
perc_ord_status=ord_sy.style.set_caption('Percentage of Orders by Status')

### Merging the datasets
detail_df= (((order_items.merge(orders_df, how="left",on='order_id'))
                 .merge(products, how="left",on='product_id'))
            .merge(product_category_english, how='left', on='product_category_name')).merge(customers, how="left", on="customer_id")

# Add column with condition for the weekend days of the purchase
conditions = [(detail_df['purchase_day'] == 'Saturday'),(detail_df['purchase_day'] == 'Sunday')]
choices = ['weekends', 'weekends']
detail_df['weekday'] = np.select(conditions, choices, default='workdays')

#filter order_status == "delivered" (because only analyze delivered orders):
detail_df= detail_df[detail_df['order_status']=='delivered']

# Create a new product dataframe with the columns of the detailed df
products_df = detail_df[['order_id', 'product_id','price', 'order_status', 'purchase_date','purchase_MMYYYY', 'purchase_year','purchase_month','purchase_day','purchase_time','weekday','product_category_name_english', 'customer_unique_id', 'customer_state'
                         , 'order_delivered_customer_date', 'order_estimated_delivery_date','order_delivered_carrier_date','shipping_limit_date', 'seller_id']]

# Pivot table with the sum of delivered orders of every month
ord_by_M=products_df.pivot_table(values = ['order_id', 'price']
                              , index=['purchase_year','purchase_month','purchase_MMYYYY']
                              , aggfunc={'order_id':'nunique','price':'sum'})

# Sort data by month-year column
ord_by_M = ord_by_M.sort_index(ascending=[1,1,1])
ord_by_M.reset_index(inplace = True)
del ord_by_M['purchase_year']
del ord_by_M['purchase_month']
ord_by_M.set_index('purchase_MMYYYY', inplace=True)
ord_by_M['revenue($R1000)']=ord_by_M['price']/1000
del ord_by_M['price']

# Revenue and Orders by Month (2016-2018)
N_ord_by_M=products_df.pivot_table(values = ['order_id', 'price']
                              , index=['purchase_year','purchase_month']
                              , aggfunc={'order_id':'nunique','price':'sum'})
N_ord_by_M.reset_index(inplace=True)
N_ord_by_M['date'] = pd.to_datetime(N_ord_by_M['purchase_year'].map(str) + '-' + N_ord_by_M['purchase_month'].map(str)
                                    ).dt.strftime("%Y-%m")

del N_ord_by_M['purchase_year']
del N_ord_by_M['purchase_month']
N_ord_by_M.set_index('date', inplace=True)
N_ord_by_M['revenue($R1000)']=N_ord_by_M['price']/1000
del N_ord_by_M['price']


#Calculating the growth for the Maximun historical compared with the previous year
products_Maximun=products_df[products_df['purchase_year'].isin([2017,2018])]
products_Maximun=products_Maximun[products_Maximun['purchase_month']<=8]
products_Maximun['revenue($R1000)']=products_Maximun['price']/1000
products_Maximun=products_Maximun.pivot_table(values='revenue($R1000)', columns='purchase_year', aggfunc ='sum')
products_Maximun.reset_index(inplace=True)
products_Maximun['%Growth']=(products_Maximun[2018]/products_Maximun[2017]-1)*100
products_Maximun.columns.names = ['']
products_Maximun.index.names = ['']

# Pivot table with the orders of nov 2017
ord_Nov_17=products_df[products_df['purchase_MMYYYY']== 'Nov-17'].pivot_table(values = ['order_id', 'price']
                              , index=['purchase_date']
                              , aggfunc={'order_id':'nunique','price':'sum'})

# date transfomed into string in order to draw the columns
ord_Nov_17.sort_index(ascending=True, inplace=True)
ord_Nov_17.reset_index(inplace = True)
ord_Nov_17 = ord_Nov_17.astype({"purchase_date": str}, errors='raise') 
ord_Nov_17.set_index('purchase_date', inplace=True)
ord_Nov_17['revenue($R1000)']=ord_Nov_17['price']/1000 # adjust the price to $R1000 scale

# Pivot table with the sum of orders, revenue and order size by date
daily_ord = products_df.pivot_table(values=["order_id","price"], index=['purchase_date','weekday']
                                ,aggfunc={"order_id":'nunique', "price":"sum"})
daily_ord['order_size'] = daily_ord['price']/daily_ord['order_id']


#ORDERS BEHAVIOUR THROUGH DAY AND TIME
# Pivot table that sums the orders number, size by day and time
ord_day_time = products_df.pivot_table(values=["order_id","price"]
                                      , index=['purchase_date','purchase_day','purchase_time']
                                      ,aggfunc={"order_id":'nunique', "price":"sum"})
ord_day_time.fillna(0, inplace = True)
ord_day_time.reset_index(level=['purchase_date','purchase_day','purchase_time'], inplace=True)
ord_day_time["order_size"]=(ord_day_time["price"]/ord_day_time["order_id"])
ord_day_time.rename(columns={'order_id':'no_of_orders'}, inplace=True)

ord_daytime = ord_day_time.pivot_table(values=['no_of_orders','order_size'], index='purchase_time'
                                      , columns='purchase_day'
                                      , aggfunc='mean').astype(int)

# columns by day reindex
day_of_week = ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday']
ord_daytime = ord_daytime.reindex(index= ['Morning', 'Afternoon','Evening','Night'])
ord_daytime = ord_daytime.reindex(columns= day_of_week, level = 'purchase_day')


# Pivot table with the mean of daily # and size of the orders by workdays and weekends
ord_wd = daily_ord.pivot_table(values=['order_id','order_size'], index='weekday', aggfunc='mean')
ord_wd=ord_wd.reindex(index= ['workdays', 'weekends'])
ord_wd['indice'] = ord_wd.index

# pivot table with the mean of daily # and size of the orders by purchase date and time
ord_time=products_df.pivot_table(values="order_id", index=['purchase_date','purchase_time'],aggfunc="nunique")
ord_time.reset_index(level="purchase_date", inplace=True)
ord_time.reset_index(level="purchase_time", inplace=True)


# Top Products
# Pivot table with the sum of revenue of each category, # and size of orders
prod_rank=products_df.pivot_table(values=['price', 'order_id'], index=['product_category_name_english']
                          , aggfunc={'price': 'sum', 'order_id': 'nunique'})
prod_rank["ord_size($R)"]=prod_rank["price"]/prod_rank["order_id"]
prod_rank["price"]=prod_rank["price"]/1000
prod_rank.sort_values(by='price', ascending = False, inplace = True)
prod_rank_top20=prod_rank.rename(columns={'order_id':'no_of_order','price':"revenue($R1000)"}).head(20)

# Delivery Performance
# add delivery related columns with date format
products_df = products_df.assign(
    order_delivered_customer_date = pd.to_datetime(products_df["order_delivered_customer_date"]).dt.date,
    order_delivered_carrier_date = pd.to_datetime(products_df["order_delivered_carrier_date"]).dt.date,
    order_estimated_delivery_date = pd.to_datetime(products_df["order_estimated_delivery_date"]).dt.date,
    shipping_limit_date = pd.to_datetime(products_df["shipping_limit_date"]).dt.date)

# calculate delivery delay: 
    # delivary days: delivery date - purchase date
    # delivary estimated days vs real deliver: estimated date - delivery customer date
    # limit vs carrier: shipping limit date - carrier date
products_df = products_df.assign(delivered_days= (products_df['order_delivered_customer_date'] - products_df['purchase_date']).dt.days
                                 ,days_est_vs_deliver= (products_df['order_estimated_delivery_date'] - products_df['order_delivered_customer_date']).dt.days
                                 ,days_limit_vs_deliver_carrier= (products_df['shipping_limit_date'] - products_df['order_delivered_carrier_date']).dt.days)

# Filtering only orders that are delivered at the moment of the historical data
products_df=products_df.assign(seller_to_carrier=np.where(products_df['days_limit_vs_deliver_carrier']<0,'late deliver to carrier','in time deliver to carrier'))
products_df['est_to_deliver'] = np.where(products_df['days_est_vs_deliver']<0, 'late deliver', 'on time deliver')

#delays of over 10, from 5 to 10 and under 5 days
conditions = [(products_df['days_est_vs_deliver'] < -10),
              (products_df['days_est_vs_deliver'] <= -5),
              (products_df['days_est_vs_deliver'] < 0)]
choices = ['late over 10 days', 'late from 5 days to 10 days','late under 5 days']
products_df['est_to_deliver_detail'] = np.select(conditions, choices, default='on time deliver')

# pivot table with the mean review score by order_id
reviews_unique = order_reviews.pivot_table(values='review_score', index='order_id', aggfunc = 'mean')
reviews_unique.reset_index(inplace=True)

# merging product and reviews dataframes
products_df=products_df.merge(reviews_unique[['order_id','review_score']], how="left", on ='order_id')

#remove duplicates for same products in 1 order to avoid errors
deliver_df=products_df.drop_duplicates(keep=False,inplace=False)

#Summary statistics
sum_nan = deliver_df.isna().sum()

# Unique order with its delivered day
deliver_ord=deliver_df[['order_id','delivered_days']].drop_duplicates(keep=False)


# unique order of delivered day and categories
deliver_uni_ord=deliver_df[['order_id','delivered_days','product_category_name_english']].drop_duplicates(keep=False)
deli_top_5= deliver_uni_ord[deliver_uni_ord['product_category_name_english'].isin(['bed_bath_table', 'health_beauty', 'watches_gifts','sports_leisure','computers_accessories'])]

# Real Delivery vs Estimation
delivery_df=deliver_df[['order_id','est_to_deliver']]
deliver=delivery_df.pivot_table(values='order_id',index='est_to_deliver', aggfunc='nunique')
deliver.sort_values(by='order_id', ascending=False, inplace=True)
deliver['status'] = deliver.index

# Categorization of late deliveries
late_deliver_df=deliver_df[deliver_df['est_to_deliver']=='late deliver']
late_deli_status=late_deliver_df.pivot_table(values='order_id',index='est_to_deliver_detail', aggfunc='nunique')
late_deli_status.sort_values(by='order_id', ascending=False, inplace=True)
late_deli_status['status']=late_deli_status.index

# 🔟 Deliver status and review score
# Filtering orders with review and removing duplicates
review_score = deliver_df[~deliver_df['review_score'].isna()]
uni_review = review_score[['order_id','est_to_deliver','review_score']].drop_duplicates()

# Pivot table with the average review score of in time and late deliver orders
uni_review_pt = uni_review.pivot_table(values='review_score', index='est_to_deliver', aggfunc='mean')
uni_review_pt.sort_values(by=['review_score'], ascending=False, inplace=True)
uni_review_pt['indice'] = uni_review_pt.index

#🚚 Late Delivers by Carrier
# Unique orders with late deliver
late_deliver=late_deliver_df.pivot_table(values='order_id',index='est_to_deliver', aggfunc='nunique')
late_deliver.rename(columns={'order_id':'unique_orders'}, inplace=True)

# Unique orders with late deliver by carrier
late_to_carrier=deliver_df[deliver_df['est_to_deliver']=='late deliver'].pivot_table(values='order_id',index='est_to_deliver', columns = 'seller_to_carrier', aggfunc='nunique')
deliver_to_carrier = late_deliver.merge(late_to_carrier, how = "left", on = 'est_to_deliver')

# Recalculat the # of orders because the products of a order may be delivered by different sellers and corresponding carriers
deliver_to_carrier.drop(columns='in time deliver to carrier')
deliver_to_carrier['in time deliver to carrier'] = deliver_to_carrier['unique_orders'] - deliver_to_carrier['late deliver to carrier']
deliver_to_carrier.drop(columns='unique_orders',inplace=True)
deli_to_carrier=deliver_to_carrier.T
deli_to_carrier['status']=deli_to_carrier.index
# Unique orders by seller
seller=deliver_df[['seller_id','seller_to_carrier','order_id']]
seller_pv=seller.pivot_table(values='order_id',index='seller_id', aggfunc='nunique')
seller_pv.reset_index(inplace=True)
seller_pv.rename(columns={"order_id": "unique_order"}, inplace=True)

# Late orders by seller
seller_late_deliver=seller[seller.seller_to_carrier=='late deliver to carrier']
seller_late_deli = seller_late_deliver.drop_duplicates() #remove duplicates
seller_late=seller_late_deli.pivot_table(values='order_id',index='seller_id', aggfunc='nunique')
seller_late.reset_index(inplace=True)
seller_late.rename(columns={"order_id": "late_order"}, inplace=True)

# Percentage of late orders by seller
seller_summary=seller_pv.merge(seller_late, how="left", on='seller_id')
seller_summary.fillna(0, inplace=True) #fill the missing values with 0
seller_summary["percent_late_order"]=seller_summary.late_order*100/seller_summary.unique_order
seller_summary.sort_values("percent_late_order", ascending=False, inplace=True)
seller_summary.head(20) #Top 20 sellers with the highest percentage of late orders

# Average orders per seller
avg_ord_per_seller = pd.DataFrame(seller_pv.mean(numeric_only=True))
avg_ord_per_seller.reset_index(inplace=True)

# Seller with most late order: # of orders >= avg orders/seller & highest percentage of late orders
seller_top=seller_summary[seller_summary['unique_order']>=avg_ord_per_seller.iloc[0,1]]
seller_top = seller_top.sort_values("percent_late_order", ascending=False)
seller_top_10 = seller_top.head(10)
seller_top_10.set_index('seller_id', inplace=True)

#Churn analysis
# Dataframe to count how many times a customer shop 
df_order = products_df.groupby(['order_id','purchase_MMYYYY','purchase_year','customer_unique_id'], as_index=False).sum().loc[:, ['order_id','customer_unique_id','purchase_MMYYYY','purchase_year', 'price']].sort_values(by='purchase_MMYYYY', ascending=True)
df_order['time_to_shop'] = 1
df_order['time_to_shop']=df_order.groupby(['customer_unique_id']).cumcount() + 1 #cumcount() starts at 0, add 1 so that it starts at 1

df_order_2016 = df_order[df_order['purchase_year']==2016]
df_order_2017 = df_order[df_order['purchase_year']==2017]
df_order_2018 = df_order[df_order['purchase_year']==2018]

customer_counter = df_order.groupby(['customer_unique_id']).count().reset_index()
customer_counter["order_count"] = customer_counter["order_id"]


customer_counter = customer_counter.drop(["order_id", "purchase_MMYYYY", "price", "time_to_shop","purchase_year"], axis=1)
customer_counter = customer_counter.groupby(["order_count"]).count().reset_index().rename(columns={"customer_unique_id": "num_customer"})
customer_counter["percentage_customer"] = 100.0 * customer_counter["num_customer"] / customer_counter["num_customer"].sum()


#Payments
payments = pd.read_csv('Data_analysis/datasets/payments.csv')
# Inspect the payment type
data = payments['payment_type'].value_counts()

payments_df = pd.DataFrame(data=data)
payments_df.rename(columns = {'payment_type':'count'}, inplace = True)
payments_df['payment_type'] = payments_df.index


#--------------------------------------------------------------------------------------------------------------------------------------------------
import plotly.graph_objects as go

#Graph 1
products_df1 = detail_df[['order_id', 'product_id','price', 'order_status', 'purchase_date','purchase_MMYYYY', 'purchase_year','purchase_month','purchase_day','purchase_time','weekday','product_category_name_english', 'customer_unique_id', 'customer_state'
                         , 'order_delivered_customer_date', 'order_estimated_delivery_date','order_delivered_carrier_date','shipping_limit_date', 'seller_id', 'customer_city']]


sao = products_df1[(products_df1['customer_city'] == 'sao paulo') | (products_df1['customer_city'] == 'rio de janeiro') | (products_df1['customer_city'] == 'belo horizonte') ]

srb = sao.pivot_table(values = ['order_id', 'price']
                                , index=['purchase_year', 'customer_city']
                                , columns= None
                                , aggfunc={'order_id': 'nunique', 'price': 'sum'})
srb.reset_index(inplace=True)
srb['date'] = pd.to_datetime(srb['purchase_year'].map(str) 
                                ).dt.strftime("%Y")
del srb['purchase_year']
srb['revenue($R1000)'] = srb['price']/1000
del srb['price']

srb.drop([0, 1, 2], inplace=True)

#Graph 2
total_sales_categories_year = products_df[['order_id','purchase_year','price','product_category_name_english']]
top5_cat_year = total_sales_categories_year.groupby(['purchase_year','product_category_name_english'],as_index=False).agg({'price':'sum'}).sort_values(by=['purchase_year','price'], ascending=False)
top5_cat_year = top5_cat_year.groupby('purchase_year').head(5).reset_index(drop=True)

#Graph 3
df_state_treemap = products_df[['order_id','purchase_year','price','product_category_name_english','customer_state']]
df_state_treemap = df_state_treemap.groupby(['customer_state','product_category_name_english'],as_index=False).agg({'price':['sum','count']})
df_state_treemap = df_state_treemap.groupby('customer_state').head(5).reset_index(drop=True)
df_state_treemap.columns =list(map(''.join, df_state_treemap.columns.values)) # to integrate the pricesum and pricecount columns from multiindex to single index

df_state_treemap = products_df[['order_id','purchase_year','price','product_category_name_english','customer_state']]
df_state_treemap_3_states = df_state_treemap.groupby(['customer_state','product_category_name_english'],as_index=False).agg({'price':['sum','count']})
df_state_treemap_3_states.columns =list(map(''.join, df_state_treemap_3_states.columns.values))
df_state_treemap_3_states = df_state_treemap_3_states.sort_values(by='pricesum', ascending=False)
df_state_treemap_3_states = df_state_treemap_3_states.groupby('customer_state').head(5).reset_index(drop=True)

top_3 = df_state_treemap_3_states.groupby('customer_state').sum().sort_values('pricesum', ascending=False).head(3).index
df_state_treemap_3_states = df_state_treemap_3_states[df_state_treemap_3_states.customer_state.isin(top_3)]

#Graph 4

best_selling_date = ord_Nov_17['revenue($R1000)'].idxmax()
best_selling_revenue = ord_Nov_17['revenue($R1000)'].max()

#Graph 6
# Reviews
reviews_df = pd.read_csv('Data_analysis/datasets/order_reviews.csv')
orders_df = pd.read_csv('Data_analysis/datasets/orders.csv')
customers_df = pd.read_csv('Data_analysis/datasets/customers.csv')
geolocation_df = pd.read_csv('Data_analysis/datasets/geolocation.csv')
df_comments = reviews_df.loc[:, ['review_score', 'review_comment_message']]
df_comments = df_comments.dropna(subset=['review_comment_message'])
df_comments = df_comments.reset_index(drop=True)
print(f'Dataset shape: {df_comments.shape}')
df_comments.columns = ['score', 'comment']
df_comments.head()

geolocation_df = geolocation_df.rename(columns={'geolocation_zip_code_prefix':'customer_zip_code_prefix'})

# create a new dataset with the customers geolocation_lat and geolocation_long
customers_geolocation = customers_df.merge(geolocation_df, how='left', on='customer_zip_code_prefix')

# clean the customer_df duplicates customer_unique_id
customers_geolocation = customers_geolocation.drop_duplicates(subset=['customer_unique_id'], keep='first')

# merge customers_geolocation and orders_df
customers_geolocation_orders = orders_df.merge(customers_geolocation, how='left', on='customer_id')

# mergue customers_geolocation_orders with reviews_df
customers__orders_reviews = customers_geolocation_orders.merge(reviews_df, how='left', on='order_id')

# map the review_score to a numerical value
score_map = {
    1: 'negative',
    2: 'negative',
    3: 'positive',
    4: 'positive',
    5: 'positive'
}
customers__orders_reviews['sentiment_label'] = customers__orders_reviews['review_score'].map(score_map)

# drop customers__orders_reviews with zip_code_prefix = 0 or nan
customers__orders_reviews = customers__orders_reviews[customers__orders_reviews['customer_zip_code_prefix'] != 0]
customers__orders_reviews = customers__orders_reviews[customers__orders_reviews['customer_zip_code_prefix'].notnull()]

#drop nan reviews_score
customers__orders_reviews = customers__orders_reviews[customers__orders_reviews['review_score'].notnull()]

#keeping only the columns required for the graph
customers__orders_reviews = customers__orders_reviews[['order_status', 'customer_unique_id', 'customer_city',
       'customer_state', 'geolocation_lat', 'geolocation_lng', 'geolocation_city', 'geolocation_state', 
       'review_id', 'review_score', 'sentiment_label']]

#Graph 6.1 
# group customers__orders_reviews by state aggregation of the count of reviews and review_score mean
customers__orders_reviews_state = customers__orders_reviews.groupby(['customer_state'])['review_score'].agg(['count', 'mean'])

#rename column to merge the coordinates
geolocation_df.rename(columns={'geolocation_state':'customer_state'}, inplace=True)

# add the cordinates of the state to the customers__orders_reviews_state
customers__orders_reviews_state = customers__orders_reviews_state.merge(geolocation_df, how='left', on='customer_state')

#drop duplicates
customers__orders_reviews_state = customers__orders_reviews_state.drop_duplicates(subset=['customer_state'], keep='first')

#keeping only the columns required for the graph
customers__orders_reviews_state = customers__orders_reviews_state[['customer_state','count','mean','geolocation_lat','geolocation_lng']]

#reset index
customers__orders_reviews_state = customers__orders_reviews_state.reset_index(drop=True)

#Some of the default values overlaps on the graph so we need to replace them with more propel values

#replace cordinates of RN with lat= -5.83 and lng= -35.21
customers__orders_reviews_state.loc[customers__orders_reviews_state['customer_state'] == 'RN', 'geolocation_lat'] = -5.833
customers__orders_reviews_state.loc[customers__orders_reviews_state['customer_state'] == 'RN', 'geolocation_lng'] = -35.21

#replace cordinates of AC with lat= -8.63 and lng= -69.78
customers__orders_reviews_state.loc[customers__orders_reviews_state['customer_state'] == 'AC', 'geolocation_lat'] = -8.63
customers__orders_reviews_state.loc[customers__orders_reviews_state['customer_state'] == 'AC', 'geolocation_lng'] = -69.78

#replace cordinates of GO with lat= -15.78 and lng= -50.69
customers__orders_reviews_state.loc[customers__orders_reviews_state['customer_state'] == 'GO', 'geolocation_lat'] = -15.78
customers__orders_reviews_state.loc[customers__orders_reviews_state['customer_state'] == 'GO', 'geolocation_lng'] = -50.69

#replaces cordinates of PA with lat= -5.93 and lng= -52.27
customers__orders_reviews_state.loc[customers__orders_reviews_state['customer_state'] == 'PA', 'geolocation_lat'] = -5.93
customers__orders_reviews_state.loc[customers__orders_reviews_state['customer_state'] == 'PA', 'geolocation_lng'] = -52.27

#replace cordinates of BA with lat= -12.29 and lng= -42.34
customers__orders_reviews_state.loc[customers__orders_reviews_state['customer_state'] == 'BA', 'geolocation_lat'] = -12.29
customers__orders_reviews_state.loc[customers__orders_reviews_state['customer_state'] == 'BA', 'geolocation_lng'] = -42.34

#replace cordinates of PE with lat= -8.59 and lng= -38.34
customers__orders_reviews_state.loc[customers__orders_reviews_state['customer_state'] == 'PE', 'geolocation_lat'] = -8.59
customers__orders_reviews_state.loc[customers__orders_reviews_state['customer_state'] == 'PE', 'geolocation_lng'] = -38.34


#Graph 7
churn = customer_counter.head(5).copy(deep=True)
churn.rename(columns={'order_count': 'Orders', 'num_customer' : 'Customers', 'percentage_customer' : 'Percentage'}, inplace=True)
churn['Percentage'] = pd.Series([round(val, 2) for val in churn['Percentage']])
churn['Percentage'] = churn['Percentage'].astype(str)
churn['Percentage'] = churn['Percentage'] + '%'
churn['Customers'] = churn['Customers'].astype(str)
churn['Customers'] = churn['Customers'].replace(['90557', '2573'], ['90,557', '2,573'])

#----------------------------------------------------------------------------------------------------------------------------------------------------------------




# DASHBOARD
# Row A
a1, a2 = st.columns(2)
a1.image(Image.open('Data_analysis/figures/masterlogo.png'))
a1.image(Image.open('Data_analysis/figures/logo-olist.png'))
a1.metric("2017-2018 Growth",format(products_Maximun["%Growth"].iloc[0],'.2f')+'%')
with a2:
    st.markdown('# Brazilian E-Commerce Dashboard')
    st.markdown('''## c10-data-modeling team
    - Gabriela Gutierrez    - Felipe Saldarriaga 
    - Daniel Reyes          - Miguel Rodriguez
    - Leandro Peñaloza      - Martin Cruz
    ''')


# Row B
b1, b2 = st.columns(2)
with b1:
    #st.write(heatmap1)
    b1.markdown('### Revenue per year of the Top 3 states (G1)')
    bar1 = px.bar(srb, x="revenue($R1000)", y="date", color="customer_city")
    b1.write(bar1)
   
with b2:
    #st.write(heatmap2)
    b2.markdown('### Graph 2')
    bar2 = px.bar(top5_cat_year.head(10),x='purchase_year',y='price',color='product_category_name_english')
    b2.write(bar2)


#h
b3, b4 =st.columns(2)
with b3:
    b3.markdown('### Graph 3')
    bar3 = px.treemap(df_state_treemap_3_states,path=['customer_state','product_category_name_english'],values='pricesum',color='product_category_name_english')
    b3.write(bar3)


with b4:
    b4.markdown("### Graph 4")
    print(best_selling_date,'Total Revenue($): ',format(best_selling_revenue*1000,'.2f'))
    bar4 = px.imshow(ord_daytime.iloc[:,:7], text_auto=True, labels=dict(x='Day of the Week', y='Purchase Time'), x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday','Sunday'],color_continuous_scale='deep')
    b4.write(bar4)
    

# Row C
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('### Graph 5')
    pie_actual_late_delivery1 = px.pie(deliver, values='order_id', names = 'status', hover_name='status')
    pie_actual_late_delivery1.update_layout(showlegend=False,width=300,height=300,margin=dict(l=1,r=1,b=1,t=1),font=dict(color='#383635', size=15))
    pie_actual_late_delivery1.update_traces(textposition='inside', textinfo='percent+label', textfont_size=20) # this function adds labels to the pie chart
    st.write(pie_actual_late_delivery1)

with c2:
    st.markdown('### Graph 6')
    # mapbox density heatmap of the reviews by customers with the geolocation_lat and geolocation_long and the review_score as the color
    bar6 = px.density_mapbox(customers__orders_reviews, lat="geolocation_lat", lon="geolocation_lng",
                            z="review_score", radius=3, mapbox_style="open-street-map",
                            range_color=(0, 5),
                            color_continuous_scale = px.colors.diverging.RdYlGn
                            ,zoom=2, center={"lat": -10.0, "lon": -51.0})
    bar6.update_layout(title_text="Reviews by customers")
    c2.write(bar6)      
with c3:
    st.markdown('### Graph 6.1')
    #plot bubble chart of customers__orders_reviews_state
    bar61 = px.scatter_geo(customers__orders_reviews_state, lat="geolocation_lat", lon="geolocation_lng",
                            color="mean", size="count",size_max=50,
                            range_color=(2, 5),
                            color_continuous_scale = px.colors.diverging.RdYlGn,
                            hover_name="customer_state",
                            hover_data=["count", "mean"],
                            text = "count",
                            center={"lat": -10.0, "lon": -51.0},
                            scope="south america",
                            template="plotly_dark")

    bar61.update_layout(title_text="Reviews by customers")
    c3.write(bar61)

#Row D
d1, d2 = st.columns(2)

with d1:
    st.markdown('### Graph 7')
    bar7 = go.Figure(data=[go.Table(
        header=dict(values=list(churn.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[churn.Orders, churn.Customers, churn.Percentage],
                fill_color='lavender',
                align='left'))
    ])

    bar7.update_layout(
            title_text = 'Churn Analysis'
        )

    d1.write(bar7)

with d2:
    st.markdown('### Graph 8')
    pie_payment1 = px.pie(payments_df, values='count', names = 'payment_type', hover_name='payment_type')
    pie_payment1.update_traces(textposition='inside', textinfo='percent+label', textfont_size=20) # this function adds labels to the pie chart
    d2.write(pie_payment1)