from kedro.pipeline import Pipeline, node
from .nodes import (
    procesar_anime_filtered, procesar_users_score, union_datasets  # Importamos SOLO la nueva función consolidada
    # Ya no necesitamos importar las funciones antiguas
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=procesar_anime_filtered,
                inputs="anime_filtered",
                outputs="anime_filtered_processed",
                name="procesar_anime_filtered_node",
            ),
            node(
                func=procesar_users_score,
                inputs="users_score",
                outputs="users_score_processed",
                name="procesar_users_score_node",
            ),
            node(
                func=union_datasets,
                inputs=["anime_filtered_processed", "users_score_processed"],
                outputs="final_anime_dataset_raw",
                name="union_datasets_node",
            ),
        ]
    )