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
    df_test = df.query("source == 'test'").drop(['source'],axis=1)
    X = df_train.copy()
    y = labels_train.loc[:, parameters["target"]]

    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, y, test_size=parameters["test_size"], random_state=parameters["random_state"], shuffle=parameters["shuffle"]
    # )
    # return X_train, X_val, y_train, y_val, df_test
    return X, y

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

def perform_grid_search(X,y):
    #X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, train_size=0.7)
    
    param_grid = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [None, 5, 7, 10],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestRegressor()
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=3)
    grid_search.fit(X, y.values.ravel())

    logger = logging.getLogger(__name__)
    logger.info("Best hyperparameters:", grid_search.best_params_)
    logger.info("Best score (MSE):", -grid_search.best_score_)

def create_submission(
        regressor: RandomForestRegressor, submissions_format: pd.DataFrame, df: pd.DataFrame
) -> pd.DataFrame:
    """Creates a prediction with the fitted regressor.

    Args:
        regressor: Trained model.
        submissions_format: CSV file

    """

    df_test = df.query("source == 'test'").drop(['source'],axis=1)
    df_test_merge = df_test[['city_1', 'year', 'weekofyear']]
    df_test_merge['city'] = df_test_merge['city_1'].map({1: 'iq', 0: 'sj'})
    
    submission_predictions = regressor.predict(df_test)
    submission_predictions = [round(x) for x in submission_predictions]

    df_test_merge['total_cases'] = submission_predictions

    merged_submission = pd.merge(df_test_merge.drop(['city_1'], axis=1), submissions_format.drop(['total_cases'], axis=1), on=['city', 'year', 'weekofyear'], how='right')
    merged_submission = merged_submission[['city','year','weekofyear','total_cases']]

    return merged_submission
    # submissions_format['total_cases'] = submission_predictions
    # return submissions_format
    #save_to_path = submission_params["path"] + '/' + submission_params["name"]
    #submissions_format.to_csv(save_to_path, sep=',', index=None)
    #logger = logging.getLogger(__name__)
    #logger.info(f"Saved submissions to {save_to_path}.")
