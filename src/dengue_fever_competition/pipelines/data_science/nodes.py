import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

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

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"], shuffle=parameters["shuffle"]
    )
    return X_train, X_val, y_train, y_val


def train_model(X_train: pd.DataFrame, y_train: pd.Series, hyperparameters: Dict) -> RandomForestRegressor:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = RandomForestRegressor(
        max_depth=hyperparameters["max_depth"],
        n_estimators=hyperparameters["n_estimators"],
        random_state=hyperparameters["random_state"],
        n_jobs=-1
    )
    regressor.fit(X_train, y_train.values.ravel())
    return regressor


def evaluate_model(
    regressor: RandomForestRegressor, X_val: pd.DataFrame, y_val: pd.Series
) -> None:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    # rf_preds = rf.predict(X_val)
    # rf_preds_int = [round(x) for x in rf_preds]
    # mean_absolute_error(y_val, rf_preds_int)

    y_pred = regressor.predict(X_val)
    y_pred = [round(x) for x in y_pred]
    score = mean_absolute_error(y_val, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a mean absolute error of %.3f on val data.", score)

def create_submission(
        regressor: RandomForestRegressor, submissions_format: pd.DataFrame, df: pd.DataFrame, submission_params: Dict
) -> None:
    """Creates a prediction with the fitted regressor.

    Args:
        regressor: Trained model.
        submissions_format: CSV file 
    
    """

    df_test = df.query("source == 'test'").drop(['source'],axis=1)

    submission_predictions = regressor.predict(df_test)
    submission_predictions = [round(x) for x in submission_predictions]

    submissions_format['total_cases'] = submission_predictions
    save_to_path = submission_params["path"] + '/' + submission_params["name"]
    submissions_format.to_csv(save_to_path, sep=',', index=None)
    logger = logging.getLogger(__name__)
    logger.info(f"Saved submissions to {save_to_path}.")
