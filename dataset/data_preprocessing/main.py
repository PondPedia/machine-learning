#!/usr/bin/python

from utils import data_preprocessing as dp
import pandas as pd

if __name__ == '__main__':
    pond = 2

    # Uncomment this if you want to clean the data
    # df = pd.read_csv(f"../raw/IoTPond{pond}.csv", index_col=0, parse_dates=True)
    # dp.wrap_it_up(pond, df)


    # Uncomment this if you want to change the interval of time into an hour
    df_list = []
    for i in range(1, 5):
        df = pd.read_csv(f'../processed/filled_IoTPond{pond}_part{i}.csv',index_col=0, parse_dates=[0])
        df_list.append(df)
    df = pd.concat(df_list)

    df = df.resample('1H').mean() # Specify The Interval
    df.to_csv(f'../processed/one_hour_IoTPond{pond}.csv', index=True)

"""
    - (Done) Change the name of the column
    - (Done) Delete column entry_id and population automatically
    - (Done) Divide The Nitrate value with 100
"""



""" # For Water Quality Parameters
    - (Done) Clean The Dataset First
    - (Done) Predict The zero value
    - 1 hour interval AVG || 1 day Interval AVG

"""


""" # Catfish size Parameters
    - 1 day Interval Avg
    - Add Delta Column
"""
