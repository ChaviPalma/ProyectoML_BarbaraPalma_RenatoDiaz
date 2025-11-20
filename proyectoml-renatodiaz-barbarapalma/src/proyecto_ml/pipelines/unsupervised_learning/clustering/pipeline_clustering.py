from kedro.pipeline import Pipeline, node
from .node_clustering import (
    entrenamiento_clustering)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=entrenamiento_clustering,
                inputs=["final_anime_dataset_regresion", "params:parametros_clustering"],
                outputs=["final_anime_dataset_clustering", "modelos_entrenados_clustering", "X_scaled_clustering"],
                name="node_entrenamiento_clustering",
            ),
        ]
    )