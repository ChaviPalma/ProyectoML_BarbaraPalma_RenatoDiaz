from kedro.pipeline import Pipeline, node
from .node_clasificacion import (
    preprocesar_anime_dataset_clasificacion,
    Entrenar_modelo_clasificacion
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preprocesar_anime_dataset_clasificacion,
                inputs="final_anime_dataset_regresion",
                outputs="final_anime_dataset_clasificacion",
                name="node_preprocesar_anime_dataset_clasificacion",
            ),
            node(
                func=Entrenar_modelo_clasificacion,
                inputs=["final_anime_dataset_clasificacion", "params:parametros_clasificacion"],
                outputs=["modelos_entrenados_clasificacion", "X_test_clasificacion", "y_test_clasificacion"],
                name="node_entrenar_modelo_clasificacion",
            ),
        ]
    )
