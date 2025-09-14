from kedro.pipeline import Pipeline, Node
from .nodes import limpiar_anime_dataset   # o la función que uses

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=limpiar_anime_dataset,
                inputs="anime_dataset",
                outputs="anime_dataset_limpio",
                name="limpiar_anime_dataset_node"
            )
        ]
    )
