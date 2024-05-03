import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def split_data(df: pd.DataFrame, labels_train: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    df_train = df.query("source == 'train'").drop(['source'],axis=1)
    X = df_train.copy()
    y = labels_train.loc[:, parameters["target"]]

    return X, y

def train_model(X: pd.DataFrame, y: pd.Series, hyperparameters: Dict) -> RandomForestRegressor:
    """Trains the Random Forest Regressor on training data.

    Args:
        X: Training data of independent features.
        y: Training target.

    Returns:
        Trained regressor.
    """
    regressor = RandomForestRegressor(
        max_depth=hyperparameters["max_depth"],
        n_estimators=hyperparameters["n_estimators"],
        random_state=hyperparameters["random_state"],
        n_jobs=-1
    )
    regressor.fit(X, y.values.ravel())
    return regressor

def create_submission(regressor: RandomForestRegressor, submissions_format: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Creates a prediction with the fitted regressor.

    Args:
        regressor: Trained model.
        submissions_format: CSV file

    """
    df_test = df.query("source == 'test'").drop(['source'],axis=1)
    df_test['predictions'] = regressor.predict(df_test)
    df_test['total_cases'] = df_test['predictions'].apply(lambda x: int(round(x)))
    df_test = df_test.reset_index(drop=True)
    submissions = pd.merge(submissions_format.drop(['total_cases'],axis=1),
                           df_test['total_cases'],
                           how='left',
                           left_index=True,
                           right_index=True)

    return submissions
