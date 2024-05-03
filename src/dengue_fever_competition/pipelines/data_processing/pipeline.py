from kedro.pipeline import Pipeline, node, pipeline
from .nodes import _remove_nulls, _encode_features, _sort_dataframe, _drop_unused_columns, _merge_dataframes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=_merge_dataframes,
                inputs=["dengue_features_train", "dengue_features_test"],
                outputs="dengue_features",
                name="merge_dataframes"
            ),
            node(
                func=_sort_dataframe,
                inputs=["dengue_features", "params:sort_columns"],
                outputs="dengue_features_sorted",
                name="sort_dataframe"
            ),
            node(
                func=_drop_unused_columns,
                inputs=["dengue_features_sorted", "params:unused_columns"],
                outputs="dengue_features_sorted_drop",
                name="drop_unused_columns"
            ),
            node(
                func=_remove_nulls,
                inputs=["dengue_features_sorted_drop", "params:forwardfills"],
                outputs="dengue_features_without_nulls",
                name="remove_null_values"
            ),
            node(
                func=_encode_features,
                inputs=["dengue_features_without_nulls", "params:encoding"],
                outputs="dengue_features_encoded",
                name="encode_features"
            )
        ]
    )
