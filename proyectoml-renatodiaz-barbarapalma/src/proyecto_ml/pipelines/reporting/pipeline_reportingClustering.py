from kedro.pipeline import Pipeline, node
from proyecto_ml.pipelines.reporting.nodes_reportingClustering import (
    graficos_clustering, evaluacion_modelos_clustering, mapa_calor_clustering)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=graficos_clustering,
                inputs="final_anime_dataset_reduction",
                outputs=["figura_scatter2d_html", "figura_scatter3d_html"],
                name="graficos_clustering_node",
            ),
            node(
                func=evaluacion_modelos_clustering,
                inputs=["X_scaled_clustering", "final_anime_dataset_reduction"],
                outputs="metricas_clustering",
                name="evaluacion_modelos_clustering_node",
            ),
            node(
                func=mapa_calor_clustering,
                inputs=["final_anime_dataset_reduction", "params:parametros_clustering"],
                outputs="figura_mapa_calor_clustering",
                name="mapa_calor_clustering_node",
            ),
        ]
    )