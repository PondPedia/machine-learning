import pandas as pd

def sampling_data(split_amount: int, pond: int, interval: str) -> None:
    """
    Read multiple CSV files, concatenate them and resample data based on the specified interval.
    The resampled data is then stored in a new CSV file.

    Args:
        split_amount (int): The number of CSV files to read.
        pond (int): The pond number used in the file path.
        interval (str): The time interval for resampling.

    Returns:
        None
    """

    # Create an empty list to store the dataframes
    df_list = []

    # Loop through the range of split_amount and read each CSV file, then append it to the list
    for i in range(1, split_amount + 1):
        df = pd.read_csv(
            f"../processed/IoTPond{pond}/cleaned_IoTPond{pond}_part{i}.csv",
            index_col=0,
            parse_dates=[0],
        )
        df_list.append(df)

    # Concatenate all dataframes in the list into one dataframe
    df = pd.concat(df_list)

    # Resample the concatenated dataframe based on the specified interval
    df_resampled = df.resample(interval)

    # Calculate the mean of all columns except the last two and store it in a new dataframe
    df_mean = df_resampled.mean().iloc[:, 0:-2]

    # Calculate the maximum value of the last two columns and store it in a new dataframe
    df_max = df_resampled.max().iloc[:, -2:]

    # Join the mean and max dataframes and store the result in a new CSV file
    df_mean.join(df_max).to_csv(f"../processed/IoTPond{pond}/hours_IoTPond{pond}.csv", index=True)
