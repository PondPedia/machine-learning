import numpy as np

def search_columns(df) -> dict:
    """
    Returns a dictionary with keywords as keys and the corresponding indices of columns from the given dataframe as values. The function takes a pandas dataframe as its only parameter and returns a dictionary.
    """
    keywords = {'temp', 'turbidity', 'oxygen', 'ph', 'ammonia', 'nitrat', 'entry_id', 'population', 'date', 'length', 'weight'}
    lower_cols = [col.lower() for col in df.columns]
    column_dict = {word: next((idx for idx, lower_col in enumerate(lower_cols) if word in lower_col), None) for word in keywords}

    return column_dict

def drop_column(column_dict: dict, df):
    """
    This function drops a column from a pandas dataframe given its name and returns the updated dataframe.
``
    Args:
        column_dict (dict): A dictionary of column names and their corresponding indices.
        df (pandas.DataFrame): The dataframe from which the columns are to be dropped.

    Returns:
        pandas.DataFrame: The updated dataframe with the specified columns dropped.
    """
    indices_to_drop = [column_dict.get('date'), column_dict.get('entry_id'), column_dict.get('population')]
    df.drop(columns=[col for col_idx, col in enumerate(df.columns) if col_idx in indices_to_drop], inplace=True)

    return df


def column_preprocessing(column_dict: dict, df):
    """
    Apply a preprocessing step to a given column of a pandas DataFrame. 

    Args:
        column_name (str): Name of the column to be preprocessed. 
        df (pandas.DataFrame): DataFrame containing the target column.

    Returns:
        pandas.DataFrame: The updated DataFrame with the preprocessed column. 
    """
    df.iloc[:, column_dict['nitrat']] = df.iloc[:, column_dict['nitrat']].apply(lambda x: x / 100)

    return df

def filter_column(column_dict: dict, df) :
    df.iloc[:, column_dict['temp']] = df.iloc[:, column_dict['temp']].apply(lambda x: x if 20 <= x <= 35 else None) # Temperature
    df.iloc[:, column_dict['oxygen']] = df.iloc[:, column_dict['oxygen']].apply(lambda x: x if 0 <= x <= 10 else None) # Dissolved Oxygen
    df.iloc[:, column_dict['ph']] = df.iloc[:, column_dict['ph']].apply(lambda x: x if 1 <= x <= 14 else None) # pH
    df.iloc[:, column_dict['ammonia']] = df.iloc[:, column_dict['ammonia']].apply(lambda x: x if 0 <= x <= 1 else None) # Ammonia
    df.iloc[:, column_dict['nitrat']] = df.iloc[:, column_dict['nitrat']].apply(lambda x: x if 0 <= x <= 20 else None) # Nitrate

    return df


def rename_column(column_dict: dict, df):
    df.rename(columns={df.columns[v]: k for k, v in column_dict.items() if v is not None}, inplace=True)

    return df

import os

def wrap_it_up(pond: int, df) -> None:
    """
    Preprocesses and splits a DataFrame into 4 parts, saving each part as a CSV file.

    Args:
        pond (int): The pond that you want to choose.
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
    df = rename_column(search_columns(df), df)

    # Check if directory exists, if not create it
    directory = f"../processed/IoTPond{pond}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Split DataFrame into 4 parts and save each part as a CSV file
    df_split = np.array_split(df, 4)
    for i, split in enumerate(df_split):
        split.to_csv(f"{directory}/cleaned_IoTPond{pond}_part{i+1}.csv", index=True)

    print('DONE')
