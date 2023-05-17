#!/usr/bin/python

from utils import data_preprocessing as dp
import pandas as pd
import numpy as np

if __name__ == '__main__':
    pond = 2
    df = pd.read_csv(f"../raw/IoTPond{pond}.csv", index_col=0, parse_dates=True)
    df = dp.drop_column(dp.search_columns(df), df)
    df = dp.column_preprocessing(dp.search_columns(df), df)
    df = dp.filter_column(dp.search_columns(df), df)
    df = dp.rename_column(df)

    df.to_csv(f'../processed/cleaned_IoTpond{pond}.csv', index=True)
    print('DONE')


"""
    - Change the name of the column (Done)
    - Delete column entry_id and population automatically (Done)
    - Divide The Nitrate value with 100 (Done)
"""



""" # For Water Quality Parameters
    - Clean The Dataset First
    - Predict The zero value
    - 1 hour interval AVG || 1 day Interval AVG

"""


""" # Catfish size Parameters
    - 1 day Interval Avg
    - Add Delta Column
"""
