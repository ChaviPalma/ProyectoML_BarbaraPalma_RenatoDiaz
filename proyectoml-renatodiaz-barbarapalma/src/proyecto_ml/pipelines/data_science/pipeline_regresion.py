# En: src/proyecto_ml/pipelines/data_science/pipeline_regresion.py

from kedro.pipeline import Pipeline, node, pipeline
from .node_regresion import (
    Entrenar_modelo_regresion,
    preprocesar_anime_dataset
)

def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline COMPLETO para la Regresión Lineal Simple."""
    return pipeline(
        [
        node(
            func=preprocesar_anime_dataset,
            inputs="final_anime_dataset_raw",
            outputs="final_anime_dataset",
            name="node_preprocesar_anime_dataset"
        ),
        node(
            func=Entrenar_modelo_regresion,
            inputs=["final_anime_dataset_regresion", "params:parametros_regresion"],
            outputs=["modelos_entrenados_regresion", "X_test_regresion", "y_test_regresion"],
            name="node_entrenar_modelo_regresion"
          )
        ]

    )