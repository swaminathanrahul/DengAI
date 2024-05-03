from kedro.pipeline import Pipeline, node, pipeline
from .nodes import _remove_nulls, _encode_features, _sort_dataframe, _drop_unused_columns


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=_sort_dataframe,
                inputs=["dengue_features_train", "params:sort_columns"],
                outputs="dengue_features_train_sorted",
                name="sort_dataframe"
            ),
            node(
                func=_drop_unused_columns,
                inputs=["dengue_features_train_sorted", "params:unused_columns"],
                outputs="dengue_features_train_sorted_drop",
                name="drop_unused_columns"
            ),
            node(
                func=_remove_nulls,
                inputs=["dengue_features_train_sorted_drop", "params:forwardfills"],
                outputs="dengue_features_train_without_nulls",
                name="remove_null_values"
            ),
            node(
                func=_encode_features,
                inputs=["dengue_features_train_without_nulls", "params:encoding"],
                outputs="dengue_features_train_encoded",
                name="encode_features"
            )
        ]
    )
