# En: src/proyecto_ml/pipelines/data_science/pipeline_regresion.py

from kedro.pipeline import Pipeline, node, pipeline
from .node_regresion import (
    preprocesar_datos,
    dividir_datos,
    seleccionar_feature_slr,
    entrenar_modelo_slr,
    evaluar_modelo_slr,
    crear_grafico_qq_residuos
)

def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline COMPLETO para la Regresión Lineal Simple."""
    return pipeline(
        [
            # --- Pasos Comunes ---
            node(
                func=preprocesar_datos,
                # --- CAMBIO AQUÍ ---
                inputs="final_anime_dataset", # <-- Nombre correcto del dataset
                # -----------------
                outputs=["X_processed", "y_target"],
                name="preprocesar_datos_node"
            ),
            # ... (El resto del pipeline se queda IGUAL) ...
            node(
                func=dividir_datos,
                inputs=["X_processed", "y_target"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="dividir_datos_node"
            ),
            node(
                func=seleccionar_feature_slr,
                inputs={
                    "X": "X_train",
                    "feature_name": "params:slr_feature"
                },
                outputs="X_train_slr",
                name="seleccionar_feature_train_slr_node"
            ),
            node(
                func=seleccionar_feature_slr,
                inputs={
                    "X": "X_test",
                    "feature_name": "params:slr_feature"
                },
                outputs="X_test_slr",
                name="seleccionar_feature_test_slr_node"
            ),
            node(
                func=entrenar_modelo_slr,
                inputs=["X_train_slr", "y_train"],
                outputs="modelo_slr",
                name="entrenar_modelo_slr_node"
            ),
            node(
                func=evaluar_modelo_slr,
                inputs=["modelo_slr", "X_test_slr", "y_test"],
                outputs=["metricas_slr", "predicciones_slr"],
                name="evaluar_modelo_slr_node"
            ),
            node(
                func=crear_grafico_qq_residuos,
                inputs=["y_test", "predicciones_slr"], 
                outputs="grafico_qq_residuos",
                name="crear_grafico_qq_node"
            )
        ]
    )