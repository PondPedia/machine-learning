#!/usr/bin/python
import pandas as pd
from utils import data_cleaning, data_sampling, get_user_inputs

if __name__ == "__main__":
    options, choice, pond, split_amount, interval = get_user_inputs()
    df = pd.read_csv(f"../raw/IoTPond{pond}.csv", index_col=0, parse_dates=True)

    if (choice == 1):
        data_cleaning.wrap_it_up(pond, df, split_amount)
    elif (choice == 2):
        data_sampling.sampling_data(split_amount, pond, interval)
    else:
        print('Invalid Choice')
