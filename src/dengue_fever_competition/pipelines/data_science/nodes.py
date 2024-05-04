import pandas as pd
import logging
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV

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

def find_best_hyperparameters(X, y) -> Dict:
    """Performs CV grid search to find best hyperparameters.
    
    Args:
        X: Training data of independent features.
        y: Training target.

    Returns:
        Dictionary with best hyperparameters found in grid search.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.2, 
                                                        random_state=42)

    rf = RandomForestRegressor()

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(estimator=rf, 
                               param_grid=param_grid, 
                               cv=5, 
                               scoring='neg_mean_absolute_error',
                               verbose=3,
                               n_jobs=-1)
    
    grid_search.fit(X_train, y_train.values.ravel())

    print("Best parameters found by GridSearchCV:")
    print(grid_search.best_params_)

    best_rf_model = grid_search.best_estimator_
    predictions = best_rf_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    print("Mean Absolute Error on test set:", mae)
    return grid_search.best_params_


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
