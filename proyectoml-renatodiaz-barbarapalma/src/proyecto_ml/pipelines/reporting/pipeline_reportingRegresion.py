from kedro.pipeline import Pipeline, node
from .nodes_reportingRegresion import (
    calcular_metricas_modelos,
    plot_metricas_comparativas,
    plot_actual_vs_predicted,
    plot_residuos,
    get_model_from_dict
)

def create_pipeline(**kwargs) -> Pipeline:
    # Nombres de tus modelos 
    nombres_modelos = [
        "LinearRegression",
        "Ridge",
        "Lasso",
        "RandomForest",
        "GradientBoosting"
    ]
    
    pipeline_nodos = [] 

    # 1. Nodo de cálculo de métricas 
    nodo_calculo_metricas = node(
        func=calcular_metricas_modelos,
        inputs=[
            "modelos_entrenados_regresion",
            "X_test_regresion",
            "y_test_regresion"
        ],
        outputs="metricas_modelos_regresion",
        name="calcular_metricas_regresion_node"
    )
    pipeline_nodos.append(nodo_calculo_metricas)

    # 2. Nodo de Gráfico de comparación (usa las métricas del nodo 1)
    nodo_grafico_comparativo = node(
        func=plot_metricas_comparativas,
        inputs="metricas_modelos_regresion",
        outputs="grafico_comparativo_metricas",
        name="plot_comparacion_metricas"
    )
    pipeline_nodos.append(nodo_grafico_comparativo)

    # 3. Bucle para generar Gráficos Individuales (Matriz X Modelos)
    for nombre in nombres_modelos:
        

        nodo_get_model = node(
            func=get_model_from_dict,
            inputs=[
                "modelos_entrenados_regresion", 
                f"params:model_names.{nombre}" 
            ], 
            outputs=f"modelo_individual_{nombre}_reg",
            name=f"get_modelo_individual_{nombre}_reg"
        )
        
        # 3.2 NODO para el gráfico de REALES VS. PREDICHOS
        nodo_actual_vs_pred = node(
            func=plot_actual_vs_predicted, 
            inputs=[
                f"modelo_individual_{nombre}_reg",
                "X_test_regresion",
                "y_test_regresion",
                f"params:model_names.{nombre}" 
            ],
            outputs=f"grafico_actual_vs_pred_{nombre}", 
            name=f"plot_actual_vs_pred_{nombre}"
        )

        # 3.3 NODO para el gráfico de RESIDUOS
        nodo_residuos = node(
            func=plot_residuos, 
            inputs=[
                f"modelo_individual_{nombre}_reg",
                "X_test_regresion",
                "y_test_regresion",
                f"params:model_names.{nombre}" 
            ],
            outputs=f"grafico_residuos_{nombre}", 
            name=f"plot_residuos_{nombre}"
        )
        
        pipeline_nodos.extend([nodo_get_model, nodo_actual_vs_pred, nodo_residuos])

    return Pipeline(pipeline_nodos)
