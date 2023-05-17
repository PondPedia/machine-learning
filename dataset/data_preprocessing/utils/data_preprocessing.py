import pandas as pd
import numpy as np

def search_columns(df) -> list:
    """
    Returns a list of columns from the given dataframe that contain any of the keywords bellow. The function takes a pandas dataframe as its only parameter and returns a list of strings.
    """
    keywords = ['temp', 'oxygen', 'ph', 'ammonia', 'nitrat', 'entry_id', 'population']
    column_index = [col for col in df.columns if any(word in col.lower() for word in keywords)]

    return column_index

def drop_column(column_name: str, df):
    """
    This function drops a column from a pandas dataframe given its name and returns the updated dataframe.

    Args:
        column_name (str): The name of the column to be dropped.
        df (pandas.DataFrame): The dataframe from which the column is to be dropped.

    Returns:
        pandas.DataFrame: The updated dataframe with the specified column dropped.
    """
    df.drop(columns=[column_name[0], column_name[-1]], inplace=True)
    return df

def column_preprocessing(column_name: str, df):
    """
    Apply a preprocessing step to a given column of a pandas DataFrame. 

    Args:
        column_name (str): Name of the column to be preprocessed. 
        df (pandas.DataFrame): DataFrame containing the target column.

    Returns:
        pandas.DataFrame: The updated DataFrame with the preprocessed column. 
    """
    df[column_name[4]] = df[column_name[4]].apply(lambda x: x / 100)
    return df

def filter_column(column_name: str, df) :
    df.loc[:, column_name[0]] = df.loc[:, column_name[0]].apply(lambda x: x if 20 <= x <= 35 else None) # Temperature
    df.loc[:, column_name[1]] = df.loc[:, column_name[1]].apply(lambda x: x if 0 <= x <= 10 else None) # Dissolved Oxygen
    df.loc[:, column_name[2]] = df.loc[:, column_name[2]].apply(lambda x: x if 1 <= x <= 14 else None) # pH
    df.loc[:, column_name[3]] = df.loc[:, column_name[3]].apply(lambda x: x if 0 <= x <= 1 else None) # Ammonia
    df.loc[:, column_name[4]] = df.loc[:, column_name[4]].apply(lambda x: x if 0 <= x <= 20 else None) # Nitrate

    return df

def rename_column(df):
    """
    Renames the columns of a given pandas DataFrame object. The function modifies the DataFrame object in place.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame object to rename its columns.

    Returns:
    --------
    pandas.DataFrame
        The modified DataFrame object with the renamed columns.
    """
    df.rename(columns={df.columns[0]: 'temperature_c',
                       df.columns[1]: 'turbidity_ntu',
                       df.columns[2]: 'do',
                       df.columns[3]: 'ph',
                       df.columns[4]: 'ammonia',
                       df.columns[5]: 'nitrate',
                       df.columns[6]: 'fish_length_cm',
                       df.columns[7]: 'fish_weight_g'}, inplace=True)

    return df

import numpy as np
import pandas as pd

def wrap_it_up(pond: int, df: pd.DataFrame) -> None:
    """
    Preprocesses and splits a DataFrame into 4 parts, saving each part as a CSV file.

    Args:
        pond (int): The pond number associated with the DataFrame.
        df (pd.DataFrame): The DataFrame to preprocess and split.
    Returns:
        None
    """
    # Drop unnecessary columns
    df = drop_column(search_columns(df), df)

    # Preprocess columns
    df = column_preprocessing(search_columns(df), df)

    # Filter rows
    df = filter_column(search_columns(df), df)

    # Rename columns
    df = rename_column(df)

    # Split DataFrame into 4 parts and save each part as a CSV file
    df_split = np.array_split(df, 4)
    for i, split in enumerate(df_split):
        split.to_csv(f"../processed/cleaned_IoTPond{pond}_part{i+1}.csv", index=True)

    print('DONE')
