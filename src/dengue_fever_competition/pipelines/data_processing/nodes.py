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
    """
    
    """
    ce_ohe = ce.OneHotEncoder(cols=columns_to_encode)
    df = ce_ohe.fit_transform(df)
    
    return df

# def _is_true(x: pd.Series) -> pd.Series:
#     return x == "t"

# def _parse_percentage(x: pd.Series) -> pd.Series:
#     x = x.str.replace("%", "")
#     x = x.astype(float) / 100
#     return x

# def _parse_money(x: pd.Series) -> pd.Series:
#     x = x.str.replace("$", "").str.replace(",", "")
#     x = x.astype(float)
#     return x

# def preprocess_companies(companies: pd.DataFrame) -> pd.DataFrame:
#     """Preprocesses the data for companies.

#     Args:
#         companies: Raw data.
#     Returns:
#         Preprocessed data, with `company_rating` converted to a float and
#         `iata_approved` converted to boolean.
#     """
#     companies["iata_approved"] = _is_true(companies["iata_approved"])
#     companies["company_rating"] = _parse_percentage(companies["company_rating"])
#     return companies


# def preprocess_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
#     """Preprocesses the data for shuttles.

#     Args:
#         shuttles: Raw data.
#     Returns:
#         Preprocessed data, with `price` converted to a float and `d_check_complete`,
#         `moon_clearance_complete` converted to boolean.
#     """
#     shuttles["d_check_complete"] = _is_true(shuttles["d_check_complete"])
#     shuttles["moon_clearance_complete"] = _is_true(shuttles["moon_clearance_complete"])
#     shuttles["price"] = _parse_money(shuttles["price"])
#     return shuttles


# def create_model_input_table(
#     shuttles: pd.DataFrame, companies: pd.DataFrame, reviews: pd.DataFrame
# ) -> pd.DataFrame:
#     """Combines all data to create a model input table.

#     Args:
#         shuttles: Preprocessed data for shuttles.
#         companies: Preprocessed data for companies.
#         reviews: Raw data for reviews.
#     Returns:
#         Model input table.

#     """
#     rated_shuttles = shuttles.merge(reviews, left_on="id", right_on="shuttle_id")
#     rated_shuttles = rated_shuttles.drop("id", axis=1)
#     model_input_table = rated_shuttles.merge(
#         companies, left_on="company_id", right_on="id"
#     )
#     model_input_table = model_input_table.dropna()
#     return model_input_table
