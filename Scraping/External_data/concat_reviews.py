import glob
import pandas as pd

"""
Scrip for cancatenate all categories reviews in one file
It'll be created in 

/Data_analysis/datasets/external_data.csv
"""

extension = 'csv'
files = [i for i in glob.glob('*.{}'.format(extension))]

external_data_reviews = pd.concat([pd.read_csv(f) for f in files ])
external_data_reviews.to_csv('../../Data_analysis/datasets/external_data.csv', index=False, encoding='utf-8')

print("Process Finished...")