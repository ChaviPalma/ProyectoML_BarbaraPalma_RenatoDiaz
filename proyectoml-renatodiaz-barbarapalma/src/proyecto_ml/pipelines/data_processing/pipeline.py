from kedro.pipeline import Pipeline, Node
from .nodes import limpiar_anime_dataset, limpiar_convertir_episodes   # o la función que uses


def create_pipeline(**kwargs):
    return Pipeline(
        [
            Node(
                func=limpiar_anime_dataset,
                inputs="anime_dataset",
                outputs="anime_dataset_limpio",
                name="limpiar_anime_dataset_node"
            ),
            Node(
                func=limpiar_convertir_episodes,
                inputs="anime_dataset_limpio",
                outputs="anime_dataset_final",
                name="limpiar_convertir_episodes_node"
            ),
        ]
    )
    