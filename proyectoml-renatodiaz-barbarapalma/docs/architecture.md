# Arquitectura del Sistema de Recomendaciones KitsuneList (Plataforma de Streaming)

## Objetivo principal y alcance

### 1.1 Propósito de Proyecto

El objetivo principal es desarrollar un **Sistema de Recomendaciones Personalizadas** para la plataforma de streaming KitsuneList, generando sugerencias de animes basadas en el historial de visualizaciones, géneros, temáticas y año de estreno del anime.

### 1.2 Métricas y Predicciones de Éxito

Este proyecto abordará dos objetivos principales:
1. **Predicción de Regresión(Afinidad):** Estimar una **Puntuación de afinidad** cuantitativa para reflejar qué tan relevante es un anime para un usuario, permitiendo poder ordenar las recomendaciones.
2. **Predicción de Clasificación(Interés):** Determinar si un usuario **mostrará interés (Sí/No)** en un anime recomendado, ayudando a filtrar las sugerencias más relevantes.

### 1.3 Impacto en el Negocio

Se espera un aumento en la retención y el engagament de los usuarios de la plataforma de streaming. Adicinalmente, se busca la optimización de costos de licencias al predecir la popularidad potencial de un anime.

## Visión General de los Pipeline

### Flujo de Ejecución por Etapas:

1. data_processing:
* **Propósito:** Limpieza, estandarización y transformación inicial de los datos de usuarios y anime.
* **Salida:** Datos limpios **02_intermediate** y **03_primary**.

2. unsupervised_learning:
* **Pipelines:** clustering, dimensionality_reduction, anomaly_detection.
* **Propósito:** Ingeniería de Features avanzada y control de calidad de datos.
* **Salida:** Features enriquecidad en **04_feature** y datasets listos en **05_model_input**.

3. Modelado supervisado(data_science y supervised_learning):
* **Pipelines:** pipeline_regresion, pipeline_clasificacion, pipeline_supervised_regression(Modelo con features de clustering).
* **Propósito:** Entrenamiento y validación de los modelos de afinidad y clasificación.
* **Salida:** Modelos serializados(**.pkl**) en **06_models**.

4. reporting:
* **Pipelines:** pipeline_reportingClasificacion, pipeline_reportingRegresion, pipeline_reportingClustering.
* **Propósito:** Evaluación exhaustiva de resultados y generación de reportes específicos por cada tarea.
* **Salida:** Métricas en JSON y gráficos en PNG en **08_reporting**.

## Componentes de Implementación y Tecnología 

| Componente | Uso Específico | Configuración Clave |
| :--- | :--- | :--- |
| **Motor de Computación** | Apache Spark: Utilizado para el procesamiento de grandes volúmenes de datos (`data_processing`). | `conf/base/spark.yml` |
| **Versión de Datos (DVC)** | Data Version Control (DVC): Se utiliza para versionar los datasets crudos (`01_raw`) y los modelos. | `dvc.yaml` |
| **Orquestación y Despliegue** | Docker y Airflow: La arquitectura está preparada para el despliegue en producción utilizando contenedores (Docker) y un orquestador (Airflow). | `docker-compose.airflow.yml`, `conf/base/airflow.yml` |
| **Modelos ML** | Se evalúa una suite de modelos (ej. Random Forest, Logistic Regression) para ambas tareas. | `data/06_models` |

## Estructura de Datos y Configuración

### 4.1 Tareas de ML separadas y gestión de Features
Se mantiene una clara división de activos para poder gestionar la complejidad de este proyecto:

* **Supervisado:** Las tareas de Regresión y Clasificación se implementan con datasets y pipelines completamente separados.

* **Aprendizaje no supervisado (Clustering):** El clustering es considerado una fuente de features. La saluda enriquece los datasets de la capa **05_model_input**, que luego alimenta a los modelos supervisados.

### 4.2 Catálogo y Persistencia 
Catalog.yml gestiona todos los activos de la siguiente manera:

* **Datasets:** Se prioriza **pandas.ParquetDataset** para los datasets de datos.
* **Modelos y Features:** Los modelos serializados que estan en **06_models** se guardan usando **pickle.PickleDataset**. Esto incluye el modelo de clustering entrenado (**modelo_clustering.pkl**), que debe persistir para ser reutilizado en la inferencia.
* **Reporting:** Se utiliza **json.JSONDataset** y **matplotlib.MatplotlibWriter** para guardar métricas y gráficos, incluyendo los resultados de la validación del clustering

### 4.3 Entornos de configuración

La carpeta **conf/** define los tres entornos para poder gestionar las configuraciones específicas: 

* **Base:** Contiene las configuraciones por defecto.
* **local:** Sobreescribe configuraciones para el desarrollo local.
* **production:** Contiene los parámetros finales optimizados y los settings para la ejecución productiva. Esto incluye también la configuración de los hiperparámetros para el proceso de clustering. 