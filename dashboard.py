#data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
import json

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

#Churn analysis
# Dataframe to count how many times a customer shop 
df_order = products_df.groupby(['order_id','purchase_MMYYYY','purchase_year','customer_unique_id'], as_index=False).sum().loc[:, ['order_id','customer_unique_id','purchase_MMYYYY','purchase_year', 'price']].sort_values(by='purchase_MMYYYY', ascending=True)
df_order['time_to_shop'] = 1
df_order['time_to_shop']=df_order.groupby(['customer_unique_id']).cumcount() + 1 #cumcount() starts at 0, add 1 so that it starts at 1

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

payments_dict= {'credit_card':'Credit Card',
'boleto':'Ticket',
'voucher':'Voucher',
'debit_card':'Debit Card',
'not_defined':'Not Defined'
}

payments_df['payment_type'] = payments_df['payment_type'].apply(lambda x: payments_dict[x])

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
product_dict={'health_beauty':'Health & Beauty',
              'watches_gifts':'Watches & Gifts',
                'bed_bath_table':'Bed & Baths',
                'sports_leisure':'Sports & Leisure',
                'computers_accessories':'Computers & Accessories',
                'furniture_decor':'Furniture & Decoration',
                'perfumery':'Perfumery',
                'toys':'Toys',
                'consoles_games':'Consoles & Games'}

top5_cat_year['product_category_name_english'] = top5_cat_year['product_category_name_english'].apply(lambda x: product_dict[x])

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
df_state_treemap_3_states['product_category_name_english'] = df_state_treemap_3_states['product_category_name_english'].apply(lambda x: product_dict[x])

#Graph 4
best_selling_date = ord_Nov_17['revenue($R1000)'].idxmax()
best_selling_revenue = ord_Nov_17['revenue($R1000)'].max()

#Graph 6
customers_orders_reviews = pd.read_csv('Data_analysis/datasets/processed/customers_orders_reviews.csv')
geolocation_df = pd.read_csv('Data_analysis/datasets/geolocation.csv')

#Graph 6.1 
# group customers__orders_reviews by state aggregation of the count of reviews and review_score mean
customers__orders_reviews_state = customers_orders_reviews.groupby(['customer_state'])['review_score'].agg(['count', 'mean'])

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


# open brazil_states.json
with open('Data_analysis/datasets/brazil-states.geojson') as json_file:
    brazil_geo = json.load(json_file)

# creates a dict with the correspinding id with the sigla
state_id_map = {}
for state in brazil_geo['features']:
    state_id_map[state['properties']['sigla']] = state['properties']['id']

# add regiao_id to the customers__orders_reviews_state
customers__orders_reviews_state['regiao_id'] = customers__orders_reviews_state['customer_state'].map(state_id_map)

#Graph 7
churn = customer_counter.head(5).copy(deep=True)
churn.rename(columns={'order_count': 'Orders', 'num_customer' : 'Customers', 'percentage_customer' : 'Percentage'}, inplace=True)
churn['Percentage'] = pd.Series([round(val, 2) for val in churn['Percentage']])
churn['Percentage'] = churn['Percentage'].astype(str)
churn['Percentage'] = churn['Percentage'] + '%'
churn['Customers'] = churn['Customers'].astype(str)
churn['Customers'] = churn['Customers'].replace(['90557', '2573'], ['90,557', '2,573'])


#Graph 9
olist_revs = pd.read_csv('Data_analysis/datasets/processed/nlp_to_olist.csv') 
external_revs = pd.read_csv('Data_analysis/datasets/processed/nlp_to_external_data.csv')

filt = olist_revs.groupby(['category'])['sentiment'].count() >= 400
mean_olist = olist_revs.groupby(['category'])[['sentiment']].mean()
filt_olist = mean_olist[filt]
filt_olist.reset_index(inplace=True)

filt_olist.drop(5, inplace=True)
filt_olist.reset_index(inplace=True)

gped_external = external_revs.groupby(['category'])[['sentiment']].mean()
gped_external.reset_index(inplace=True)

translated_cats = pd.read_csv('Data_analysis/datasets/product_category_name_translation.csv')
translated_cats.rename({'product_category_name':'category'}, axis=1, inplace=True)

translated_revs = pd.merge(gped_external, translated_cats, on='category', how='left')

translated_revs.fillna('sports_leisure', inplace=True)
translated_revs.sort_values(by='product_category_name_english', inplace=True)

category_dict = {'auto': 'Auto',
 'baby': 'Baby',
 'bed_bath_table': 'Bed & Baths',
 'computers_accessories': 'Computers & Accessories',
 'consoles_games': 'Consoles & Games',
 'electronics': 'Electronics',
 'fashion_bags_accessories': 'Fashion & Bags',
 'furniture_decor': 'Furniture & Decoration',
 'garden_tools': 'Garden Tools',
 'health_beauty': 'Health & Beauty',
 'housewares': 'Housewares',
 'luggage_accessories': 'Luggages',
 'office_furniture': 'Office Furniture',
 'perfumery': 'Perfumery',
 'pet_shop': 'Pet Shop',
 'sports_leisure': 'Sports & Leisure',
 'stationery': 'Stationery',
 'telephony': 'Telephony',
 'toys': 'Toys',
 'watches_gifts': 'Watches & Gifts'}

filt_olist['category'] = filt_olist['category'].apply(lambda x: category_dict[x])

Total_rev_f = float(products_Maximun[2018])*1000

Total_rev_f = format(Total_rev_f,'.2f')

best_sell_f = best_selling_revenue*1000

best_sell_f = format(best_sell_f,'.2f')


#----------------------------------------------------------------------------------------------------------------------------------------------------------------


# DASHBOARD

# Row A
a1, a2 = st.columns(2)
a1.image(Image.open('Data_analysis/figures/logo-olist.png'))
a2.title('Brazilian E-Commerce Dashboard')

st.markdown('***')

a3,a4 =st.columns(2)
a3.metric(label='Total Revenue ', value='$'+f'{float(Total_rev_f):,}', delta=format(products_Maximun["%Growth"].iloc[0],'.2f')+'%')
a4.metric(label='Record date of Revenue',value=best_selling_date,delta='$'+f'{float(best_sell_f):,}')


# Row B
b1, b2 = st.columns(2)
with b1:
    b1.markdown('### 🥇 Revenue from the Top 3 States (G1)')
    bar1 = px.bar(srb, x="revenue($R1000)", y="date", color="customer_city",
                  labels=dict(date='Date',customer_city='Customer City'),width=600,height=350)
    bar1.update_layout(legend=dict(yanchor="top",y=0.40,xanchor="left",x=0.75),margin=dict(l=1,r=2,b=1,t=1))
    bar1.update_xaxes(title='Revenue ($R1000)',visible=True, showticklabels=True)
    bar1.update_yaxes(title=None,visible=True, showticklabels=True)
    b1.write(bar1)
   
with b2:
    b2.markdown('### 💎 Revenue from the Top 5 Categories (G2)')
    bar2 = px.bar(top5_cat_year.head(10),x='purchase_year',y='price',color='product_category_name_english', 
                  labels=dict(purchase_year='Year', price='Revenue',product_category_name_english='Product Category'),width=600,height=350)
    bar2.update_xaxes(title='Years (2017-2018)',visible=True, showticklabels=False)
    bar2.update_layout(margin=dict(l=1,r=1,b=1,t=1))
    b2.write(bar2)


#b part 2
b3, b4 =st.columns(2)
with b3:
    b3.markdown('### 🔝 Share of revenue at Top 3 States (G3)')
    bar3 = px.treemap(df_state_treemap_3_states,path=['customer_state','product_category_name_english'],values='pricesum',color='product_category_name_english',width=600,height=350)
    bar3.update_layout(margin=dict(l=1,r=1,b=1,t=1))
    b3.write(bar3)


with b4:
    b4.markdown("### ⏰ Orders distribuited by Time and Day (G4)")
    bar4 = px.imshow(ord_daytime.iloc[:,:7], text_auto=True, labels=dict(x='Day of the Week', y='Purchase Time'), x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday','Sunday'],color_continuous_scale='deep',width=600,height=450)
    bar4.update_yaxes(title=None,visible=True, showticklabels=True)
    bar4.update_xaxes(title=None,visible=True, showticklabels=True)
    bar4.update_layout_images(margin=dict(l=1,r=1,b=1,t=1))
    b4.write(bar4)
    
c1,c2 = st.columns(2)
with c1:
    st.markdown('### 🚛 On Time vs Late Deliveries (G5)')
    pie_actual_late_delivery1 = px.pie(deliver, values='order_id', names = 'status', hover_name='status',color_discrete_sequence=['DarkBlue','DarkRed'],width=600,height=450)
    pie_actual_late_delivery1.update_layout(showlegend=False,width=300,height=300,margin=dict(l=1,r=1,b=1,t=1),font=dict(color='#383635', size=15))
    pie_actual_late_delivery1.update_traces(textposition='inside', textinfo='percent+label', textfont_size=20) # this function adds labels to the pie chart
    st.write(pie_actual_late_delivery1)
with c2:
    st.markdown('### 💱 Payment Preferences (G8)')
    pie_payment1 = px.pie(payments_df, values='count', names = 'payment_type', hover_name='payment_type',color_discrete_sequence=px.colors.sequential.Agsunset,width=600,height=450)
    pie_payment1.update_layout(showlegend=True,width=600,height=300,margin=dict(l=1,r=1,b=1,t=1),font=dict(color='#383635', size=15))
    pie_payment1.update_traces(textposition='inside', textinfo='percent+label', textfont_size=20) # this function adds labels to the pie chart
    c2.write(pie_payment1)



# Row C
d1, d2 = st.columns(2)

with d1:
    st.markdown('### 🤔 Volume of Reviews per State (G6.1)')
    # creates a choropleth map using the lat_lng column and state_id_map
    bar62 = px.choropleth(customers__orders_reviews_state, geojson=brazil_geo, locations="customer_state",
                    color="mean",
                    color_continuous_scale=px.colors.diverging.RdYlGn,
                    hover_name="customer_state",
                    hover_data=["count", "mean"], featureidkey="properties.sigla",
                    scope="south america",
                    template="plotly_dark",width=600,height=450)
    bar62.update_geos(fitbounds="locations")
    bar62.update_coloraxes(showscale=False)
    bar62.update_layout(margin=dict(l=1,r=1,b=1,t=1))
    d1.write(bar62)

with d2:
    st.markdown('### 💙 Review Score per User (G6.2)')
    # mapbox density heatmap of the reviews by customers with the geolocation_lat and geolocation_long and the review_score as the color
    bar6 = px.density_mapbox(customers_orders_reviews, lat="geolocation_lat", lon="geolocation_lng",
                            z="review_score", radius=3, mapbox_style="open-street-map",
                            range_color=(0, 5),
                            color_continuous_scale = px.colors.diverging.RdYlGn
                            ,zoom=3, center={"lat": -10.0, "lon": -51.0},width=600,height=450,
                            labels=dict(review_score='Review Score'))
    bar6.update_layout()
    d2.write(bar6)
    
#Row E
e1, e2 = st.columns(2)

with e1:
    st.markdown('### 🏃🏽‍♀️💨 Churn Analysis (G7)')
    bar7 = go.Figure(data=[go.Table(
        header=dict(values=list(churn.columns),
                    fill_color='darkblue',
                    align='center',
                    font_size=16),
        cells=dict(values=[churn.Orders, churn.Customers, churn.Percentage],
                fill_color='DarkSlateBlue',
                align='center',
                font_size=14))])
    bar7.update_layout(margin=dict(l=1,r=1,b=1,t=1),width=600)
    e1.markdown('\n')
    e1.markdown('\n')
    e1.markdown('\n')
    e1.markdown('\n')
    e1.markdown('\n')
    e1.markdown('\n')
    e1.write(bar7)


with e2:
    #graph 9
    categories = filt_olist['category']

    radar_graph = go.Figure()

    radar_graph.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0,1],
                
            )),
            showlegend=False,
            template="plotly_dark")

    radar_graph.add_trace(go.Scatterpolar(
        r=filt_olist['sentiment'],
        theta=categories,
        name='Olist',
    ))


    radar_graph.add_trace(go.Scatterpolar(
        r=translated_revs['sentiment'],
        theta=categories,
        name='Competitor',
    ))
    radar_graph.update_layout(margin=dict(l=1,r=1,b=20,t=20))
    radar_graph.update_layout(width=600,height=400)
    e2.markdown('### 🐱‍🚀 Sentiment Benchmarking (G9)')
    
    e2.write(radar_graph)

st.markdown('***')
a4,a5,a6,a7 = st.columns(4)
a4.image(Image.open('Data_analysis/figures/masterlogo.png'))
a5.markdown(''' ### Cohort 10 [CX] 
[Data Modeling Team](https://github.com/C10-Brazilian-e-commerce-modeling-team/brazilian-e-commerce)''')
a6.markdown('''- Gabriela Gutierrez [@GabyGO2108](https://github.com/GabyGO2108)
- Daniel Reyes [@danieldhats7](https://github.com/danieldhats7)
- Leandro Peñaloza [@leopensaa](https://github.com/leopensaa)''')
a7.markdown('''- Felipe Saldarriaga [@felipesaldata](https://github.com/felipesaldata)
- Alejandro Rodriguez [@alexrods](https://github.com/alexrods)
- Martin Cruz [@martin-crdev](https://github.com/martin-crdev)''')

    

