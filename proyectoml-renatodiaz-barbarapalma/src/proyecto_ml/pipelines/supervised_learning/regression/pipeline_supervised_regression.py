from kedro.pipeline import Pipeline, node
from proyecto_ml.pipelines.supervised_learning.regression.node_supervised_regression import (
    modelo_regresion_supervisada)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=modelo_regresion_supervisada,
                inputs=["final_anime_dataset_reduction", "params:parametros_regresion"],
                outputs="metricas_regresion_supervisada",
                name="node_modelo_regresion_supervisada",
            ),
        ]
    )