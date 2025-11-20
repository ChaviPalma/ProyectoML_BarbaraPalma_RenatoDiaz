from kedro.pipeline import Pipeline, node
from .node_reduction import (
    reducir_dimensionalidad)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [ 
            node(
                func=reducir_dimensionalidad,
                inputs=["final_anime_dataset_clustering", "params:parametros_clustering", "X_scaled_clustering"],
                outputs="final_anime_dataset_reduction",
                name="node_reducir_dimensionalidad",
            ),
        ]
    )
    