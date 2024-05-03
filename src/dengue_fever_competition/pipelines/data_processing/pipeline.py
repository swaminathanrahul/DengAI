from kedro.pipeline import Pipeline, node, pipeline
from .nodes import _remove_nulls, _encode_features, _drop_unused_columns, _merge_dataframes, _create_rolling_averages, _cyclical_encoding


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=_merge_dataframes,
                inputs=["dengue_features_train", "dengue_features_test"],
                outputs="df_features",
                name="merge_dataframes"
            ),
            node(
                func=_remove_nulls,
                inputs=["df_features", "params:forwardfills"],
                outputs="df_features_no_nulls",
                name="remove_null_values"
            ),
            node(
                func=_create_rolling_averages,
                inputs=["df_features_no_nulls","params:rolling_cols"],
                outputs="df_features_rolling",
                name="create_moving_averages"
            ),
            node(
                func=_cyclical_encoding,
                inputs="df_features_rolling",
                outputs="df_features_cyclical",
                name="cyclical_encoding"
            ),
            node(
                func=_encode_features,
                inputs=["df_features_cyclical", "params:encoding"],
                outputs="df_features_onehot",
                name="encode_features"
            ),
            node(
                func=_drop_unused_columns,
                inputs=["df_features_onehot", "params:unused_columns"],
                outputs="df_features_processed",
                name="drop_unused_columns"
            ),
 
        ]
    )
