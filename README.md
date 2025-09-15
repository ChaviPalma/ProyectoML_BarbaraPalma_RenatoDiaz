# PROYECTO DE MACHINE LEARNING PARA UN SISTEMA DE RECOMENDACIÓN DE ANIME

## Descripción 

Este proyecto busca el crear un sistema de recomendación de animes para usuarios. Para ello se utilizarón dataset de kaggle sobre anime. El trabajo incluye kedro para la automatización de los datos.

Se trabajó en tres fases principales:
1. **Business Understanding**: Análisis del problema y definición de objetivos.
2. **Data Understanding**: Exploración y análisis del dataset.
3. **Data Preparation**: Limpieza, transformación y preparación de los datos para el sistema de recomendación, automatizado con Kedro.


## Requisitos
- **Python** 3.11
- **Kedro** >= 0.18.0
- **Kedro-Viz** >= 6.0.0
- **Kedro-Datasets** >= 1.0.0
- **Pandas** >= 1.3.0
- **NumPy** >= 1.21.0
- **PyArrow**, **fastparquet**, **gzip** (para manipulación de archivos Parquet)


## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/usuario/proyecto_recomendacion.git
cd proyecto_recomendacion
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
├── conf/
│ ├── base/
│ │ ├── catalog.yml Configuración de datasets
│ │ ├── parameters.yml Parámetros del proyecto
│ │ └── logging.yml Configuración de logs
├── data/
│ ├── 01_raw/ Datos originales
│ ├── 02_intermediate/ Datos procesados parcialmente
│ ├── 03_primary/ Datos limpios
│ ├── 04_feature/ Features para modelado
│ └── 05_model_input/ Datos listos para entrenar
├── docs/
│ └── source/ Documentación del proyecto
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
│ │ │ ├── nodes.py
│ │ │ └── pipeline.py
│ │ └── reporting/
│ │ ├── nodes.py
│ │ └── pipeline.py
│ └── pipeline_registry.py
├── README.md
├── requirements.txt
└── .gitignore
```
## Ejecución para la fase 3

```bash
kedro run --pipeline=data_processing
```





