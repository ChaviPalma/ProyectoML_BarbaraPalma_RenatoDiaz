from kedro.pipeline import Pipeline, node
from .nodes import (
    calcular_metricas_clasificacion,
    plot_metricas_comparativas_clasificacion,
    plot_confusion_matrix,
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

    # Nodo 1: Calcular métricas y reportes
    nodo_calculo_metricas = node(
        func=calcular_metricas_clasificacion,
        inputs=["modelos_entrenados", "X_test_clasificacion", "y_test_clasificacion"],
        outputs=[
            "metricas_modelos_clasificacion", 
            "reportes_clasificacion"           
        ],
        name="calcular_metricas_clasificacion"
    )

    # Nodo 2: Gráfico de comparación
    nodo_grafico_comparativo = node(
        func=plot_metricas_comparativas_clasificacion,
        inputs="metricas_modelos_clasificacion",
        outputs="grafico_comparativo_metricas_clasificacion",
        name="plot_comparacion_metricas_clasificacion"
    )

    pipeline_nodos = [nodo_calculo_metricas, nodo_grafico_comparativo]

    # Bucle para crear matrices de confusión
    for nombre in nombres_modelos:
        
        nodo_get_model = node(
            func=get_model_from_dict,
            inputs=["modelos_entrenados", f"params:{nombre}_name_class"],
            outputs=f"modelo_{nombre}_class",
            name=f"get_modelo_{nombre}_class"
        )
        
        nodo_confusion_matrix = node(
            func=plot_confusion_matrix,
            inputs=[
                f"modelo_{nombre}_class",
                "X_test_clasificacion",
                "y_test_clasificacion",
                f"params:{nombre}_name_class"
            ],
            outputs=f"grafico_matriz_confusion_{nombre}",
            name=f"plot_matriz_confusion_{nombre}"
        )
        
        pipeline_nodos.extend([nodo_get_model, nodo_confusion_matrix])

    return Pipeline(pipeline_nodos)