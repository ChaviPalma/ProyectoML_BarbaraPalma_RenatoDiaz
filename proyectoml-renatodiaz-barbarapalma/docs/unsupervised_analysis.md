# Análisis No Supervisado y Enriquecimiento de Features para KitsuneList 

Este documento detalla la arquitectura técnica y la justificación de las pipelines Apredizaje No Supervisado, cuyo objetivo es la ingeniería de Características que precede al modelado supervisado de KinetsuList.

## 1. Rol Estratégico y Flujo de Datos

El análisis no supervisado opera sobre los datos primarios (**03_primary**) para poder generar variables latentes en la capa de features(**04_feature**), que son fundamentales para el entrenamiento de los modelos de regresión y clasificación.

## 2. Detalle por Pipeline de unsupervised_learning

### 2.1 Clustering

Este pipeline es la fuente principal de las features enriquecidas utilizadas por los modelos de Regresión y Clasificación

| Aspecto | Detalle de Implementación | Activos Kedro Clave |
| :--- | :--- | :--- |
| **Objetivo** | Segmentar usuarios o animes para identificar grupos con perfiles de consumo o propiedades homogéneas. | `final_anime_dataset_clustering.parquet` |
| **Preprocesamiento** | Se utiliza un objeto de escalado (`X_scaled_clustering.pkl`) para **normalizar las features** antes del entrenamiento. | `X_scaled_clustering.pkl` |
| **Modelo** | El modelo de clustering entrenado (ej. K-Means) se persiste en la capa de modelos (`06_models`) para ser utilizado en el *scoring* de nuevos datos. | `modelos_entrenados_clustering.pkl` |
| **Validación y Reporte** | Se reportan métricas intrínsecas del clustering (ej. **Silhouette Score**) y visualizaciones para interpretar los segmentos. | `metricas_clustering.json`, `figura_mapa_calor_clustering.png` |

### 2.2 Reducción de Dimensionalidad 

Este pipeline apoya la interpretabilidad y la eficiencia en el feature space

| Aspecto | Detalle de Implementación | Activos Kedro Clave |
| :--- | :--- | :--- |
| **Objetivo** | Visualización y simplificación de las **features** (características) del dataset. | `final_anime_dataset_reduction.parquet` |
| **Técnicas** | Se utilizan técnicas de proyección (ej. **UMAP** o **PCA**) para generar proyecciones 2D y 3D. | N/A (el modelo de reducción se ejecuta dentro del nodo) |
| **Reporte de Visualización** | Generación de gráficos de dispersión 2D y 3D para **mapear los clusters** y facilitar la interpretación de los segmentos al equipo de negocio. | `figura_scatter2d_html.png`, `figura_scatter3d_html.png` |

### 2.3 Detección de Anomalías

Este pipeline actúa como un filtro de calidad de datos avanzados previo al modelado supervisado

| Aspecto | Detalle de Implementación | Activos Kedro Clave |
| :--- | :--- | :--- |
| **Objetivo** | Identificar y marcar *outliers* o registros anómalos que puedan sesgar los modelos supervisados de Regresión/Clasificación. | `final_anime_dataset_anomalydetection.parquet` |
| **Resultado** | La salida es el *dataset* con las filas **marcadas** para su revisión y posible exclusión del entrenamiento de modelos. | `anomalias_detectadas.parquet` |
| **Reporte** | Generación de una figura de diagnóstico para visualizar la distribución y el **marcado de las anomalías**. | `fig_anomalias.png` |

## 3. Impacto y Trazabilidad en Modelos Supervisados

El éxito de este análisis no supervisado se mide por el impacto directo que tiene sobre el rendimiento de los modelos de recomendación

* **Integración de Features:** La feature de Cluster ID se inyecto en el dataset de entrenamiento creando la versión enriquecida utilizada por el módulo de **supervised_learning/regression**.
* **Validación de Valor:** El pipeline tiene una etapa de reporting específica (**regresion_supervisada**) que valida la contribución del clustering.
* **Trazabilidad:** El archivo de **metricas_regresion_supervisada.json** registra el rendimiento del modelo supervisado que incluye el clustering, permitiendo cuantificar la mejora aportada por esta feature avanzada en comparación con un modelo base.