#!/usr/bin/python

from utils import data_preprocessing as dp
import pandas as pd

if __name__ == "__main__":
    pond = 2
    split_amount = 1

    # Uncomment this if you want to clean the data
    # df = pd.read_csv(f"../raw/IoTPond{pond}.csv", index_col=0, parse_dates=True)
    # dp.wrap_it_up(pond, df, split_amount=split_amount)

    # Uncomment this if you want to change the interval of time into an hour
    df_list = []
    for i in range(1, split_amount + 1):
        df = pd.read_csv(
            f"../processed/IoTPond{pond}/cleaned_IoTPond{pond}_part{i}.csv",
            index_col=0,
            parse_dates=[0],
        )
        df_list.append(df)
    df = pd.concat(df_list)

    # Specify The Interval
    df_resampled = df.resample("6H")
    df_mean = df_resampled.mean().iloc[:, 0:-2]
    df_max = df_resampled.max().iloc[:, -2:]  
    df_mean.join(df_max).to_csv(f"../processed/IoTPond{pond}/6_hours_IoTPond{pond}.csv", index=True)

"""
    - (Done) Change the name of the column
    - (Done) Delete column entry_id and population automatically
    - (Done) Divide The Nitrate value with 100
"""


""" # For Water Quality Parameters
    - (Done) Clean The Dataset First
    - (Done) Predict The zero value
    - (Done) 1 hour interval AVG || 1 day Interval AVG
"""


""" # Catfish size Parameters
    - 1 day Interval Avg
    - Add Delta Column
    - water quality parameters (avg)
    - fish size parameters (max)
"""

"""
    1. Friza
    - Email 
    - Create a model with combined version of IoTPond 1 and 2 (6 Hours Intervals)
    

    2. Irvan
    - Keep Playing with 1 Day Interval
    - Python script for filling missing values automatically

    3. Astrid
    - Preprocessing Data 3 & 4 (8 Hours Intervals)
    - 
"""
