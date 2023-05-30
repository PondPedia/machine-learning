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