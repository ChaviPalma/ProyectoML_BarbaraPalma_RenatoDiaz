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

    # Definición del flujo
    [procesar_anime, procesar_users] >> unir_datasets >> preprocesar_anime_dataset_regresion >> entrenar_regresion >> preprocesar_anime_dataset_clasificacion >> Entrenar_modelo_clasificacion  
