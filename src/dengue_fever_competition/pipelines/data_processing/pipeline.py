from kedro.pipeline import Pipeline, node, pipeline
from .nodes import _remove_nulls, _encode_features, _sort_dataframe, _drop_unused_columns, _merge_dataframes, create_rolling_averages, add_season_column, encode_cosine_seasonality


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
                func=create_rolling_averages,
                inputs=["dengue_features_sorted","params:rolling_columns"],
                outputs="dengue_features_rolling",
                name="create_moving_averages"
            ),
            # node(
            #     func=add_season_column,
            #     inputs="dengue_features_rolling",
            #     outputs="dengue_features_season",
            #     name="add_seasonality"
            # ),
            node(
                func=encode_cosine_seasonality,
                inputs="dengue_features_rolling",
                outputs="dengue_features_cosine",
                name="cosine_seasonality"
            ),
            node(
                func=_drop_unused_columns,
                inputs=["dengue_features_cosine", "params:unused_columns"],
                outputs="dengue_features_drop",
                name="drop_unused_columns"
            ),
            node(
                func=_remove_nulls,
                inputs=["dengue_features_drop", "params:forwardfills"],
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
