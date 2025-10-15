from kedro.pipeline import Pipeline, Node
from .nodes import (
    limpiar_anime_dataset,
    limpiar_convertir_episodes,
    limpiar_convertir_rank,
    inspeccionar_users_detail,
    limpiar_users_detail,
    inspeccionar_users_score,
    union_dataset_score_detail,
    union_dataset_anime_users,
    create_basic_anime_features,   # <-- agregamos esta importación
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            Node(
                func=limpiar_anime_dataset,
                inputs="anime_dataset",
                outputs="anime_dataset_limpio",
                name="limpiar_anime_dataset_node"
            ),
            Node(
                func=limpiar_convertir_episodes,
                inputs="anime_dataset_limpio",
                outputs="anime_dataset_episodes",   
                name="limpiar_convertir_episodes_node"
            ),
            Node(
                func=limpiar_convertir_rank,
                inputs="anime_dataset_episodes",
                outputs="anime_dataset_rank",   
                name="limpiar_convertir_rank_node"
            ),
            # 🔹 Nuevo nodo: crear features básicas del anime
            Node(
                func=create_basic_anime_features,
                inputs="anime_dataset_rank",
                outputs="anime_dataset_features",   # dataset con features nuevas
                name="create_basic_anime_features_node"
            ),
            Node(
                func=inspeccionar_users_detail,
                inputs="users_detail",
                outputs="users_detail_inspeccionado",
                name="inspeccionar_users_detail_node_users"
            ),
            Node(
                func=limpiar_users_detail,
                inputs="users_detail_inspeccionado",
                outputs="users_detail_limpio",
                name="limpiar_users_detail_node_users"
            ),
            Node(
                func=inspeccionar_users_score,
                inputs="users_score",
                outputs="users_score_inspeccionado",
                name="inspeccionar_users_score_node_v1"
            ),
            Node(
                func=union_dataset_score_detail,
                inputs=["users_score_inspeccionado", "users_detail_limpio"],
                outputs="final_users",
                name="union_dataset_score_detail_node_v1"
            ),
            # 🔹 Y aquí hacemos la unión final usando el dataset con features
            Node(
                func=union_dataset_anime_users,
                inputs=["anime_dataset_features", "final_users"],  # <-- usamos las features nuevas
                outputs="final_anime_dataset",
                name="union_dataset_anime_users_node_v1"
            ),
        ]
    )
