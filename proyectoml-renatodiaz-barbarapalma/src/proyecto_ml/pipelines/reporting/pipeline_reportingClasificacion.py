from kedro.pipeline import Pipeline, node
from .nodes_reportingClasificacion import (
    calcular_metricas_clasificacion,
    plot_metricas_comparativas_clasificacion,
    plot_confusion_matrix,
    plot_roc_curve,  
    get_model_from_dict
)

def create_pipeline(**kwargs) -> Pipeline:
    
    # Nombres de tus modelos
    nombres_modelos = [
        "LogisticRegression",
        "SGDClassifier",
        "RandomForest",
        "DecisionTree",
        "KNeighborsClassifier"
    ]

    pipeline_nodos = []

    # --- NODO 1: Calcular métricas y reportes ---
    nodo_calculo_metricas = node(
        func=calcular_metricas_clasificacion,
        inputs=["modelos_entrenados_clasificacion", "X_test_clasificacion", "y_test_clasificacion"],
        outputs=[
            "metricas_modelos_clasificacion", 
            "reportes_clasificacion"
        ],
        name="calcular_metricas_clasificacion_node"
    )
    pipeline_nodos.append(nodo_calculo_metricas)

    # --- NODO 2: Gráfico de comparación ---
    nodo_grafico_comparativo = node(
        func=plot_metricas_comparativas_clasificacion,
        inputs="metricas_modelos_clasificacion",
        outputs="grafico_comparativo_metricas_clasificacion",
        name="plot_comparacion_metricas_clasificacion_node"
    )
    pipeline_nodos.append(nodo_grafico_comparativo)

    # --- BUCLE: Crear matrices de confusión y curvas ROC ---
    for nombre in nombres_modelos:
        nodo_get_model = node(
            func=get_model_from_dict,
            inputs=[
                "modelos_entrenados_clasificacion",
                f"params:parametros_modelos_clasificacion.{nombre}"
            ],
            outputs=f"modelo_individual_{nombre}_class",
            name=f"get_modelo_individual_{nombre}_class"
        )

        nodo_confusion_matrix = node(
            func=plot_confusion_matrix,
            inputs=[
                f"modelo_individual_{nombre}_class",
                "X_test_clasificacion",
                "y_test_clasificacion",
                f"params:parametros_modelos_clasificacion.{nombre}"
            ],
            outputs=f"grafico_matriz_confusion_{nombre}",
            name=f"plot_matriz_confusion_{nombre}_class"
        )

        # --- 2. CREAR EL NODO PARA LA CURVA ROC ---
        nodo_roc_curve = node(
            func=plot_roc_curve,
            inputs=[
                f"modelo_individual_{nombre}_class",
                "X_test_clasificacion",
                "y_test_clasificacion",
                f"params:parametros_modelos_clasificacion.{nombre}"
            ],
            outputs=f"grafico_roc_curve_{nombre}",
            name=f"plot_roc_curve_{nombre}_class"
        )

        # --- 3. AÑADIR LOS TRES NODOS (get, cm, roc) ---
        pipeline_nodos.extend([
            nodo_get_model, 
            nodo_confusion_matrix,
            nodo_roc_curve  # <--- AÑADIRLO AQUÍ
        ])

    return Pipeline(pipeline_nodos)