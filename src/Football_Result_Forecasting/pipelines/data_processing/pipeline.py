from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preporcessing, numerical_prep

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preporcessing,
            inputs="combined_seasons",
            outputs="preprocessed_categorical",
            name="preprocess_categorical_node",
        ),
        node(
            func=numerical_prep,
            inputs="preprocessed_categorical",
            outputs="preprocessed_data",
            name="preprocess_numerical_node",
        ),
    ])
