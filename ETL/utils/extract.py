import gdown
import pandas as pd
from constants import DATA_DIR


class Extract_csv:
    def __init__(self, name, url):      
        self.name = name
        self.url = url
        self.path = DATA_DIR / f'{name}.csv'

    def extract(self):
        """Extract data from url, stores it on self.path and return it

        Returns:
            str: csv path location
        """
        
        self.path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(self.url, self.path.as_posix(), quiet=False)
        print(f'Downloaded {self.name} to {self.path}')
        return self.path

    def tranform(self, df):
        
        return self.path

class Extract_enrich_order(Extract_csv):

    def time_periods(self, x):
        if x>=5 and x<12:
            return "Morning"
        elif x>=12 and x<17:
            return "Afternoon"
        elif x>=17 and x<21:
            return "Evening"
        else:
            return "Night"


    def tranform(self, df):
        """Transform raw data and returns it on a pd.Dataframe

        Args:
            df (pd.DataFrame): Dataframe to transform

        Returns:
            pd.Dataframe: Transformed df
        """
        orders_df = df[['order_id','customer_id','order_status','order_purchase_timestamp','order_estimated_delivery_date',
                            'order_delivered_customer_date','order_delivered_carrier_date']]

        # add the date columns
        orders_df = orders_df.assign(
            purchase_date = pd.to_datetime(orders_df['order_purchase_timestamp']).dt.date,
            purchase_year = pd.to_datetime(orders_df['order_purchase_timestamp']).dt.year,
            purchase_month = pd.to_datetime(orders_df['order_purchase_timestamp']).dt.month,
            purchase_mmyyyy= pd.to_datetime(orders_df['order_purchase_timestamp']).dt.strftime('%b-%y'),
            purchase_day = pd.to_datetime(orders_df['order_purchase_timestamp']).dt.day_name(),
            purchase_hour = pd.to_datetime(orders_df['order_purchase_timestamp']).dt.hour)

        #add the time_periods
        orders_df["purchase_time"]= orders_df["purchase_hour"].apply(self.time_periods)

        orders_df.to_csv(DATA_DIR / 'orders_T.csv', index=False)
        path = str(self.path).replace('.csv', '_T.csv')

        return path
