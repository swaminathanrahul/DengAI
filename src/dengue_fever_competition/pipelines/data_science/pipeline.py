from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data, train_model, create_submission, find_best_hyperparameters

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs= ["df_features_processed","dengue_labels_train", "params:split_options"],
                outputs=["X","y"],
                name="split_data_into_X_and_y"
             ),
            node(
                  func=train_model,
                  inputs=["X", "y", "params:hyperparameters"],
                  outputs="regressor",
                  name="train_model"
              ),
             node(
                 func=create_submission,
                 inputs=["regressor", "submission_format", "df_features_processed"],
                 outputs="submissions",
                 name="create_submission_file"
             ),
        ]
    )
