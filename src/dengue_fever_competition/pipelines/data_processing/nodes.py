import pandas as pd
import category_encoders as ce
import logging

def _sort_dataframe(df_train: pd.DataFrame, sort_column: list) -> pd.DataFrame:
    """
    Something smart
    """
    return df_train.sort_values(by=sort_column)

def _remove_nulls(df_train: pd.DataFrame, ndvis: list) -> pd.DataFrame:
    """
    Something smart
    """
    # replace missing vegetation pixels with previous week
    for ndvi in ndvis:
        df_train[ndvi] = df_train[ndvi].fillna(method='ffill')
    
    # replace remaining numerical missing values with mean
    float_columns = df_train.select_dtypes('float64').columns
    for col in float_columns:
        df_train[col] = df_train[col].fillna(df_train[col].mean())
    
    return df_train

def _encode_features(df_train: pd.DataFrame, columns_to_encode: list) -> pd.DataFrame:
    """
    Something smart
    """
    ce_ohe = ce.OneHotEncoder(cols=[columns_to_encode])
    df_train = ce_ohe.fit_transform(df_train)
    
    return df_train

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
