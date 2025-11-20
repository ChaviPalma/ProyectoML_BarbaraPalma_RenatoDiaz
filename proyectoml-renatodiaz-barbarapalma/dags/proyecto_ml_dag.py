from datetime import datetime, timedelta
from airflow import DAG
# Importamos BashOperator, que es ideal para comandos de línea (como Kedro CLI)
from airflow.operators.bash import BashOperator

# Eliminamos la función run_kedro_node y sus imports asociados (KedroSession, gc, sys, pathlib)
# ya que ahora usaremos el Kedro CLI directamente.

# Definición de la plantilla del comando Kedro. Los nodos se ejecutan usando '--nodes'.
KEDRO_RUN_COMMAND = "cd /app && kedro run --nodes {node_tag}"

# Definición del DAGgg
with DAG(
    dag_id="proyecto_ml_pipeline",
    schedule_interval=None,
    start_date=datetime(2025, 10, 31),
    catchup=False,
    default_args={
        "owner": "renato",
        "depends_on_past": False,
        "retries": 0,
        "execution_timeout": timedelta(minutes=240),
    },
    description="Pipeline de Kedro dividido por nodes, ejecutado con BashOperator",
) as dag:

    # Utilizamos BashOperator para llamar al CLI de Kedro para cada node
    procesar_anime = BashOperator(
        task_id="procesar_anime_filtered",
        bash_command=KEDRO_RUN_COMMAND.format(node_tag="procesar_anime_filtered_node"),
    )

    procesar_users = BashOperator(
        task_id="procesar_users_score",
        bash_command=KEDRO_RUN_COMMAND.format(node_tag="procesar_users_score_node"),
    )

    unir_datasets = BashOperator(
        task_id="union_datasets",
        bash_command=KEDRO_RUN_COMMAND.format(node_tag="union_datasets_node"),
    )

    preprocesar_anime_dataset_regresion = BashOperator(
        task_id="preprocesar_anime_dataset_regresion",
        bash_command=KEDRO_RUN_COMMAND.format(node_tag="node_preprocesar_anime_dataset"),
    )


    entrenar_regresion = BashOperator(
        task_id="Entrenar_modelo_regresion",
        bash_command=KEDRO_RUN_COMMAND.format(node_tag="node_entrenar_modelo_regresion"),
    )


    preprocesar_anime_dataset_clasificacion = BashOperator(
        task_id="preprocesar_anime_dataset_clasificacion",
        bash_command=KEDRO_RUN_COMMAND.format(node_tag="node_preprocesar_anime_dataset_clasificacion"),
    )

    Entrenar_modelo_clasificacion = BashOperator(
        task_id="Entrenar_modelo_clasificacion",
        bash_command=KEDRO_RUN_COMMAND.format(node_tag="node_entrenar_modelo_clasificacion"),
    )

    calcular_metricas = BashOperator(
        task_id="calcular_metricas_regresion_node",
        bash_command=KEDRO_RUN_COMMAND.format(node_tag="calcular_metricas_regresion_node"),
    )

    calcular_metricas_clasificacion = BashOperator(
        task_id="calcular_metricas_clasificacion_node",
        bash_command=KEDRO_RUN_COMMAND.format(node_tag="calcular_metricas_clasificacion_node"),
    )

    plot_comparacion_metricas_regresion = BashOperator(
        task_id="plot_comparacion_metricas",
        bash_command=KEDRO_RUN_COMMAND.format(node_tag="plot_comparacion_metricas"),
    )

    plot_comparacion_metricas_clasificacion = BashOperator(
        task_id="plot_comparacion_metricas_clasificacion_node",
        bash_command=KEDRO_RUN_COMMAND.format(node_tag="plot_comparacion_metricas_clasificacion_node"),
    )

    clustering = BashOperator(
        task_id="unsupervised_clustering",
        bash_command=KEDRO_RUN_COMMAND.format(node_tag="node_entrenamiento_clustering"),
    )

    reduction = BashOperator(
        task_id="dimensionality_reduction",
        bash_command=KEDRO_RUN_COMMAND.format(node_tag="node_reducir_dimensionalidad"),
    )

    detectar_anomalias = BashOperator(
        task_id="detectar_anomalias",
        bash_command=KEDRO_RUN_COMMAND.format(node_tag="node_detectar_anomalias"),
    )



    # Procesamiento inicial
    procesar_anime >> unir_datasets
    procesar_users >> unir_datasets

    # Regresión
    unir_datasets >> preprocesar_anime_dataset_regresion >> entrenar_regresion

    # Clasificación
    unir_datasets >> preprocesar_anime_dataset_clasificacion >> Entrenar_modelo_clasificacion

    # Evaluación
    entrenar_regresion >> calcular_metricas
    Entrenar_modelo_clasificacion >> calcular_metricas_clasificacion

    # Unsupervised Learning
    entrenar_regresion >> clustering >> reduction >> detectar_anomalias

    # Visualización
    calcular_metricas >> plot_comparacion_metricas_regresion
    calcular_metricas_clasificacion >> plot_comparacion_metricas_clasificacion