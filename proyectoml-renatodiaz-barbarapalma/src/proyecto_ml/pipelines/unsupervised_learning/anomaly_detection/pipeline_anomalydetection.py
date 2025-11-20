from kedro.pipeline import Pipeline, node
from .node_anomalydetection import (
    detectar_anomalias)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=detectar_anomalias,
                inputs=["final_anime_dataset_reduction", "X_scaled_clustering", "params:parametros_clustering"],
                outputs=["final_anime_dataset_anomalydetection", "fig_anomalias", "anomalias_detectadas"],
                name="node_detectar_anomalias",
            ),
        ]
    )