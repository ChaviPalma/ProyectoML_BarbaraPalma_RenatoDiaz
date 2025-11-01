# PROYECTO DE MACHINE LEARNING PARA UN SISTEMA DE RECOMENDACIÓN DE ANIME

## Integrantes 

1. Bárbara Palma
2. Renato Díaz

## Descripción 

Este proyecto busca el crear un sistema de recomendación de animes para usuarios. Para ello se utilizarón dataset de kaggle sobre anime. El trabajo incluye kedro para la automatización de los datos.

Se trabajó en Cuatro fases principales:
1. **Business Understanding**: Análisis del problema y definición de objetivos.
2. **Data Understanding**: Exploración y análisis del dataset.
3. **Data Preparation**: Limpieza, transformación y preparación de los datos para el sistema de recomendación, automatizado con Kedro.
4. **Data Modeling**: Creación de modelos y métricas, con sus visualizaciones.


## Link kaggle 
https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset/code?datasetId=3384322&sortBy=voteCount

## Link video Evaluación 1

https://drive.google.com/drive/folders/1skjSbRVAH0YZRHZBv82O1jU3wuTuoH6V?usp=sharing

## Link video Evaluación 2


## Requisitos
- **Python** 3.11
- **Kedro** >= 0.18.0
- **Kedro-Viz** >= 6.0.0
- **Kedro-Datasets** >= 1.0.0
- **Pandas** >= 1.3.0
- **NumPy** >= 1.21.0
- **PyArrow**, **fastparquet**, **gzip** (para manipulación de archivos Parquet)
- **ipykernel**, **ipython**, **jupyter**, **jupyterlab** (Para los notebooks)
- **Matplotlib** >= 3.4.0
- **Seaborn** ~= 0.12.1
- **Plotly** >=  5.0.0
- **scikit-learn** ~= 1.5.1
- **DVC** <= 2.0.0 (Versionamiento de datos)
- **Docker y Docker Compose**
- **Apache-Airflow** >= 2.8.0
- **Kedro-Airflow** >= 0.4.0
- **psycopg2-binary** (Base de datos de Airflow)
- **statsmodels** >= 0.14.5
- **pasty** >= 1.0.2

## Instalación

1. Clonar el repositorio:
```bash
https://github.com/ChaviPalma/ProyectoML_BarbaraPalma_RenatoDiaz
cd proyectoml-renatodiaz-barbarapalma
```
2. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```
3. Instalar dependencias
```bash
pip install -r requirements.txt
```

## Estructura del proyecto
```bash
├── .config
├── .dvc  #Metadatos de DVC
|  ├── config 
├── conf/ #Configuración de Kedro
|  ├── base/ # Configuración principal 
│ | ├── catalog.yml Configuración de datasets
│ | ├── parameters.yml Parámetros del proyecto
│ | └── airflow.yml
│ | └── spark.yml
│  ├── local/  #Configuración local
│  ├── production/
│ ├── logging.yml Configuración de logs
├── dags/  #Dags de Airflow
| ├── proyecto_ml_dag.py
├── data/
│ ├── 01_raw/ Datos originales
│ ├── 02_intermediate/ Datos procesados parcialmente
│ ├── 03_primary/ Datos limpios
│ ├── 04_feature/ Features para modelado
│ └── 05_model_input/ Datos listos para entrenar
│ └── 06_models/ #Modelos entrenados
│ └── 07_model_output/ #Predicciones
│ └── 08_reporting/  #Gráficps y Métricas
│ └── 01_raw.dvc/ 
├── docker/ #Definiciones de Docker
│ └── Dockerfile.airflow
│ └── Dockerfile.jupyter
│ └── Dockerfile.kedro
├── docs/
│ └── source/ Documentación del proyecto
│ └── conf.py
├── notebooks/
│ ├── 01_business_understanding.ipynb
│ ├── 02_data_understanding.ipynb
│ └── 03_data_preparation.ipynb
├── src/
│ └── proyecto_ml/
│ ├── pipelines/
│ │ ├── data_processing/
│ │ │ ├── nodes.py Funciones de procesamiento de la fase 3
│ │ │ └── pipeline.py
│ │ ├── data_science/
│ │ │ ├── node_clasificacion.py
│ │ │ └── pipeline_clasificacion.py
| | | └── node_regresion.py
| | | └── pipeline_regresion.py
│ │ └── reporting/
| | | └── nodes_reportingClasificacion.py
| | | └── pipeline_reportingClasificacion.py
| | | └── nodes_reportingRegresion.py
| | | └── pipeline_reportingRegresion.py
│ └── pipeline_registry.py
├── .dive-ci
├── .dockerignore
├── .dvcignore
├── docker-compose.airflow.yml  #Docker compose para deplegar Airflow
├── docker-compose.yml #Docker compose para ejecutar el proyecto
├── info.log  #Log de Kedro
├── pyproject.toml  #Definición de proyecto Python
├── README.md 
├── requirements.txt #Dependencias de Python
└── .gitignore
```
## Ejecución para la fase 3

```bash
kedro run --pipeline=data_processing
```
## Ejecución para la fase 4 (Regresion y clasificación)

```bash
kedro run --pipeline=regresion
kedro run --pipeline=clasificacion
```
## Ejecución para la fase 5

```bash
kedro run --pipeline reporting_clasificacion
kedro run --pipeline reporting_regresion
```
## Ejecución para todo el proyecto

```bash
kedro run 
```

## Visualización de los modelos

```bash
kedro viz
```

## Construcción de manual de imágenes

```bash
docker build -f docker/Dockerfile.kedro -t custom_kedro #Construir la imagen de Kedro
docker build -f docker/Dockerfile.airflow -t custom_airflow #Construir la imagen de Airflow
docker build -f docker/Dockerfile.jupyter -t custom_jupyter #Construir la imagen de Jupyter
```
## Construcción con Docker Compose

```bash
docker-compose build
```
