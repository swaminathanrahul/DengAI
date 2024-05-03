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
    for col in columns_with_nulls:
        df[col] = df[col].fillna(method='ffill')
    
    # replace remaining numerical missing values with mean
    float_columns = df.select_dtypes('float64').columns
    for col in float_columns:
        df[col] = df[col].fillna(df[col].mean())
    
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