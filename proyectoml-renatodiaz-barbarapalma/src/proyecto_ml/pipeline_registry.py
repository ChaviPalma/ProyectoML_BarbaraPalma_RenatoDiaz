# src/proyecto_ml/pipeline_registry.py

from __future__ import annotations
from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from proyecto_ml.pipelines import data_processing as dp
# Importamos el módulo que AHORA contiene el pipeline RLS completo
from proyecto_ml.pipelines.data_science import pipeline_regresion as ds_regresion_module
from proyecto_ml.pipelines.data_science import pipeline_clasificacion as ds_clasificacion_module

def register_pipelines() -> dict[str, Pipeline]:
    """Registra los pipelines del proyecto."""

    data_processing_pipeline = dp.create_pipeline()
    # create_pipeline() ahora es el pipeline RLS completo
    regresion_pipeline = ds_regresion_module.create_pipeline()
    clasificacion_pipeline = ds_clasificacion_module.create_pipeline()

    return {

        "__default__": data_processing_pipeline + regresion_pipeline,
        "data_processing": data_processing_pipeline,
        "regresion": regresion_pipeline, # Registra el pipeline RLS completo
        "clasificacion": clasificacion_pipeline, # Registra el pipeline de clasificación
    }