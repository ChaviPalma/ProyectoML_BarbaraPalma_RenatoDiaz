from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from proyecto_ml.pipelines.data_processing import pipeline as dp
from proyecto_ml.pipelines.data_science import pipeline as ds


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = dp.create_pipeline()
    data_science_pipeline = ds.create_pipeline()

    return {
        "__default__": data_processing_pipeline + data_science_pipeline,
        "data_processing": data_processing_pipeline,
        "data_science": data_science_pipeline,
    }