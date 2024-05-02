from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data, train_model, evaluate_model, create_submission


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs= ["dengue_features_train_encoded","params:model_options"],
                outputs=["X_train", "X_val", "y_train", "y_val"],
                name="split_data"
             ),
            node(
                  func=train_model,
                  inputs=["X_train", "y_train", "params:hyperparameters"],
                  outputs="regressor",
                  name="train_model_node"
              ),
            node(
                  func=evaluate_model,
                  inputs=["regressor","X_val", "y_val"],
                  outputs=None,
                  name="evaluate_model_node",
             ),
             node(
                 func=create_submission,
                 inputs=["regressor", "submission_format", "dengue_features_test", "params:submission"],
                 outputs=None,
                 name="create_submission"
             )
             
            # node(
            #     func=split_data,
            #     inputs=["model_input_table", "params:model_options"],
            #     outputs=["X_train", "X_test", "y_train", "y_test"],
            #     name="split_data_node",
            # ),
            # node(
            #     func=train_model,
            #     inputs=["X_train", "y_train"],
            #     outputs="regressor",
            #     name="train_model_node",
            # ),
            # node(
            #     func=evaluate_model,
            #     inputs=["regressor", "X_test", "y_test"],
            #     outputs=None,
            #     name="evaluate_model_node",
            # ),
        ]
    )
