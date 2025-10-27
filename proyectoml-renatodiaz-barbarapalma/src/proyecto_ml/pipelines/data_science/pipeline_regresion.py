
from kedro.pipeline import Pipeline, node

from .nodes import (
    preprocesar_datos,
    dividir_datos
)



def create_pipeline() -> Pipeline:
    """Crea el pipeline de procesamiento de datos."""
    return Pipeline(
        [
            # Nodo 1: Toma datos crudos y los procesa
            node(
                func=preprocesar_datos,
                inputs="anime_raw_data",  # Input: El dataset crudo (de catalog.yml)
                outputs=["X_processed", "y_target"], # Outputs: Features y target
                name="preprocesar_datos_node" # Un nombre único para este nodo
            ),
            
            # Nodo 2: Toma features y target, y los divide
            node(
                func=dividir_datos,
                inputs=["X_processed", "y_target"], # Inputs: Los outputs del nodo anterior
                outputs=["X_train", "X_test", "y_train", "y_test"], # Outputs: Los 4 sets de datos
                name="dividir_datos_node"
            )
        ]
    )
