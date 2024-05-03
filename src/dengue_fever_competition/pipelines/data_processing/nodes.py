import numpy as np
import pandas as pd
import category_encoders as ce
import logging


def _merge_dataframes(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """Merge train and test into a single dataframe and annotates each row with ``train`` or ``test``.

    Args:
        df_train: Training dataset.
        df_test: Test dataset.
    
    Returns:
        df: Concatenated dataframe.
    """
    df_train['source'] = 'train'
    df_test['source'] = 'test'
    df = pd.concat([df_train, df_test])
    return df

def _sort_dataframe(df: pd.DataFrame, sort_column: list) -> pd.DataFrame:
    """Sorts the dataframe by columns in ``sort_column``

     Args:
         df: Dataframe
         sort_column: List of column names to sort by
     Returns:
         DataFrame sorted by columns in ``sort_column``.
     """
    return df.sort_values(by=sort_column)

def create_rolling_averages(df: pd.DataFrame, column_name: list):
    """Something smart
    
    """
    # Calculate rolling average for temperature grouped by 'city'
    for n in range(2,5):
        for col in column_name:
            rollingcolumnname = 'rolling_avg_' + str(n) + "_" + col
            df[rollingcolumnname] = df.groupby('city')[col].transform(
                lambda x: x.rolling(window=n, min_periods=1).mean())
    return df

def add_season_column(df: pd.DataFrame) -> pd.DataFrame:
    # Define a dictionary mapping week numbers to seasons
    
    # df.loc[df['weekofyear'] == 53, 'weekofyear'] = 52
    season_mapping = {
        1: 'Winter', 2: 'Winter', 3: 'Winter', 4: 'Winter', 5: 'Winter', 6: 'Winter',
        7: 'Winter', 8: 'Winter', 9: 'Winter', 10: 'Winter', 11: 'Winter', 12: 'Spring',
        13: 'Spring', 14: 'Spring', 15: 'Spring', 16: 'Spring', 17: 'Spring', 18: 'Spring',
        19: 'Spring', 20: 'Spring', 21: 'Spring', 22: 'Spring', 23: 'Spring', 24: 'Summer',
        25: 'Summer', 26: 'Summer', 27: 'Summer', 28: 'Summer', 29: 'Summer', 30: 'Summer',
        31: 'Summer', 32: 'Summer', 33: 'Summer', 34: 'Summer', 35: 'Summer', 36: 'Summer',
        37: 'Summer', 38: 'Summer', 39: 'Summer', 40: 'Autumn', 41: 'Autumn', 42: 'Autumn',
        43: 'Autumn', 44: 'Autumn', 45: 'Autumn', 46: 'Autumn', 47: 'Autumn', 48: 'Autumn',
        49: 'Autumn', 50: 'Autumn', 51: 'Autumn', 52: 'Winter', 53: 'Winter'
    }
    
    df['season'] = df['weekofyear'].map(season_mapping)
    df['season'] = df['season'].astype('category')
    return df

def encode_cosine_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    df['weekofyear_cyclic'] = np.cos((df['weekofyear'] - 26.5)/25.5)
    return df
    
def _drop_unused_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """Drops unused columns from ``columns_to_drop``

    Args:
        df: Dataframe
        columns_to_drop: List of column names to drop.
    """
    df = df.drop(columns_to_drop, axis=1)
    return df

def _remove_nulls(df: pd.DataFrame, columns_with_nulls: list) -> pd.DataFrame:
    """Handle null values by using forward fill on ndvi coluumns and taking the mean for all numerical columns.
    
    Args:
        df: Dataframe
        columns_with_nulls: List of column names that contain null values.
    """
    # replace missing vegetation pixels with previous week
    float_columns = list(df.select_dtypes('float64').columns)
    all_columns = columns_with_nulls + float_columns
    
    for col in all_columns:
        df[col] = df[col].fillna(method='ffill')
    
    return df

def _encode_features(df: pd.DataFrame, columns_to_encode: list) -> pd.DataFrame:
    """Onehot encode features according to columns defined in ``columns_to_encode``

    Args:
        df: DataFrame
        columns_to_encode: List of column names that need to be encoded.
    
    """
    ce_ohe = ce.OneHotEncoder(cols=columns_to_encode)
    df = ce_ohe.fit_transform(df)
    
    return df