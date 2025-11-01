from kedro.pipeline import Pipeline, node
from .nodes import (
    calcular_metricas_modelos,
    plot_metricas_comparativas,
    plot_actual_vs_predicted,
    plot_residuos,
    get_model_from_dict
)

def create_pipeline(**kwargs) -> Pipeline:
    
    nombres_modelos = [
        "LinearRegression", 
        "Ridge", 
        "Lasso", 
        "RandomForest", 
        "GradientBoosting"
    ]

    # --- NODO 1: Calcular todas las métricas primero ---
    nodo_calculo_metricas = node(
        func=calcular_metricas_modelos,
        inputs=["modelos_entrenados", "X_test_regresion", "y_test_regresion"],
        outputs="metricas_modelos",
        name="calcular_metricas_regresion"
    )

    # --- NODO 2: Gráfico de comparación  ---
    nodo_grafico_comparativo = node(
        func=plot_metricas_comparativas,
        inputs="metricas_modelos",
        # Salida: El gráfico de barras comparativo
        outputs="grafico_comparativo_metricas",
        name="plot_comparacion_metricas"
    )

    # Inicializamos la lista de nodos del pipeline
    pipeline_nodos = [nodo_calculo_metricas, nodo_grafico_comparativo]

    # --- BUCLE: Crear los gráficos para CADA modelo ---
    for nombre in nombres_modelos:
        
        # Nodo 'A': Extrae el modelo del diccionario
        nodo_get_model = node(
            func=get_model_from_dict,
            inputs=["modelos_entrenados", f"params:{nombre}_name"],
            outputs=f"modelo_{nombre}", 
            name=f"get_modelo_{nombre}"
        )
        
        # Nodo 'B': Gráfico Reales vs. Predichos
        nodo_actual_vs_pred = node(
            func=plot_actual_vs_predicted,
            inputs=[
                f"modelo_{nombre}", 
                "X_test_regresion", 
                "y_test_regresion", 
                f"params:{nombre}_name"
            ],
            outputs=f"grafico_actual_vs_pred_{nombre}", 
            name=f"plot_actual_vs_pred_{nombre}"
        )
        
        # Nodo 'C': Gráfico de Residuos
        nodo_residuos = node(
            func=plot_residuos,
            inputs=[
                f"modelo_{nombre}", 
                "X_test_regresion", 
                "y_test_regresion", 
                f"params:{nombre}_name"
            ],
            outputs=f"grafico_residuos_{nombre}", 
            name=f"plot_residuos_{nombre}"
        )
        
        
        pipeline_nodos.extend([nodo_get_model, nodo_actual_vs_pred, nodo_residuos])

    return Pipeline(pipeline_nodos)