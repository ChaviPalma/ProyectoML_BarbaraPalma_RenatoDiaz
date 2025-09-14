from kedro.pipeline import Pipeline, Node
from .nodes import limpiar_anime_dataset, limpiar_convertir_episodes, convertir_premiered_booleano   # o la función que uses


def create_pipeline():
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
                outputs="anime_dataset_paso_episodes",
                name="limpiar_convertir_episodes_node"
            ),
            Node(
                func=convertir_premiered_booleano,
                inputs="anime_dataset_paso_episodes",
                outputs="anime_dataset_paso_premiered",
                name="convertir_premiered_booleano_node"
            )
        ]
    )
    