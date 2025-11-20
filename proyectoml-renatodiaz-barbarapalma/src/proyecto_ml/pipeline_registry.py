# src/proyecto_ml/pipeline_registry.py

from __future__ import annotations
from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from proyecto_ml.pipelines import data_processing as dp
# Importamos el módulo que AHORA contiene el pipeline RLS completo
from proyecto_ml.pipelines.data_science import pipeline_regresion as ds_regresion_module
from proyecto_ml.pipelines.data_science import pipeline_clasificacion as ds_clasificacion_module
from proyecto_ml.pipelines.reporting import pipeline_reportingClasificacion as rpC
from proyecto_ml.pipelines.reporting import pipeline_reportingRegresion as rpR
from proyecto_ml.pipelines.unsupervised_learning.clustering import pipeline_clustering as ds_clustering_module
from proyecto_ml.pipelines.unsupervised_learning.dimensionality_reduction import pipeline_reduction as ds_reduction_module
from proyecto_ml.pipelines.unsupervised_learning.anomaly_detection import pipeline_anomalydetection as rpA

def register_pipelines() -> dict[str, Pipeline]:
    """Registra los pipelines del proyecto."""

    data_processing_pipeline = dp.create_pipeline()
    # --- Modelos Pipelines --- #
    regresion_pipeline = ds_regresion_module.create_pipeline()
    clasificacion_pipeline = ds_clasificacion_module.create_pipeline()
    clustering_pipeline = ds_clustering_module.create_pipeline()
    dimensionality_reduction_pipeline = ds_reduction_module.create_pipeline()
    #--- Reporting Pipelines --- #
    reporting_pipeline_clasificacion = rpC.create_pipeline()
    reporting_pipeline_regresion = rpR.create_pipeline()
    reporting_pipeline_anomalydetection = rpA.create_pipeline()


    return {

        "__default__": (
            data_processing_pipeline
            + regresion_pipeline
            + clasificacion_pipeline
            + reporting_pipeline_clasificacion
            + reporting_pipeline_regresion
            + clustering_pipeline
            + dimensionality_reduction_pipeline
            + reporting_pipeline_anomalydetection
        ),
        "data_processing": data_processing_pipeline,
        "regresion": regresion_pipeline, # Registra el pipeline RLS completo
        "clasificacion": clasificacion_pipeline, # Registra el pipeline de clasificación
        "reporting_clasificacion": reporting_pipeline_clasificacion,
        "reporting_regresion": reporting_pipeline_regresion,
        "clustering": clustering_pipeline,
        "reduccion": dimensionality_reduction_pipeline,
        "anomalydetection": reporting_pipeline_anomalydetection,
    }