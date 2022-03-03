"""Project pipelines."""

from typing import Dict
from kedro.pipeline import Pipeline

from Football_Result_Forecasting.pipelines import data_processing as dp
from Football_Result_Forecasting.pipelines import data_science as ds



def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipeline.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    data_processing_pipeline = dp.create_pipeline()
    data_science_pipeline = ds.create_pipeline()

    return {
        "__default__": data_processing_pipeline + data_science_pipeline,
        "dp": data_processing_pipeline,
        "ds": data_science_pipeline,
    }