import numpy as np
import pandas as pd
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

def _create_rolling_averages(df: pd.DataFrame, rolling_cols: list, window: int = 2):
    """Taking the features most correlated with ``total_cases`` and creating rolling averages with a window of two.
    
    Args:  
        df: DataFrame
        rolling_cols: List of column names for which rolling averages are to be created.
        window: Setting backwards-looking window size.
    
    Returns:
        df: DataFrame with rolling averages.
    """
    for col in rolling_cols:
        df[col + '_rolling_' + str(window)] = df[col].rolling(window=window).mean()

    return df

def _cyclical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Creates sin and cos encodings for ``weekofyear``.

    Args:
        df: DataFrame
    
    Returns:
        df: DataFrame with ``week_sin`` and ``week_cos``.
    
    """

    df['week_sin'] = np.sin(2 * np.pi * df['weekofyear']/53)
    df['week_cos'] = np.cos(2 * np.pi * df['weekofyear']/53)
    
    return df
    
def _drop_unused_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """Drops unused columns from ``columns_to_drop``

    Args:
        df: DataFrame
        columns_to_drop: List of column names to drop.

    Returns:
        df: DataFrame without columns specified in ``columns_to_drop``.
    """
    df = df.drop(columns_to_drop, axis=1)
    return df

def _remove_nulls(df: pd.DataFrame, columns_with_nulls: list) -> pd.DataFrame:
    """Handle null values by using forward fill on ndvi coluumns and taking the mean for all numerical columns.
    
    Args:
        df: Dataframe
        columns_with_nulls: List of column names that contain null values.
    
    Returns:
        df: DataFrame without Null values (forward-filled).
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
    
    Returns:
        df: DataFrame with onehot-encoded columns.    
    
    """
    for col in columns_to_encode:
        unique_values = df[col].unique()
        for unique_value in unique_values:
            df[unique_value] = df[col].apply(lambda x: 1 if x == unique_value else 0)
        df = df.drop([col],axis=1)
    
    return df