from kedro.pipeline import Pipeline, Node
from .nodes import limpiar_anime_dataset, limpiar_convertir_episodes, limpiar_convertir_rank, inspeccionar_users_detail, limpiar_users_detail, inspeccionar_users_score # o la función que uses
from .nodes import union_dataset_score_detail, union_dataset_anime_users

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
            Node(
                func=inspeccionar_users_detail,
                inputs="users_detail",
                outputs="users_detail_inspeccionado",  # intermedio temporal
                name="inspeccionar_users_detail_node_users"
            ),
            Node(
                func=limpiar_users_detail,
                inputs="users_detail_inspeccionado",
                outputs="users_detail_limpio",  # dataset final intermedio
                name="limpiar_users_detail_node_users"
            ),
            Node(
                func=inspeccionar_users_score,
                inputs="users_score",
                outputs="users_score_inspeccionado",  # dataset intermedio
                name="inspeccionar_users_score_node_v1"
            ),
            Node(
                func=union_dataset_score_detail,
                inputs=["users_score_inspeccionado", "users_detail_limpio"],
                outputs="final_users",  # dataset final intermedio
                name="union_dataset_score_detail_node_v1"
            ),
            Node(
                func=union_dataset_anime_users,
                inputs=["anime_dataset_rank", "final_users"],
                outputs="final_anime_dataset",  # dataset final
                name="union_dataset_anime_users_node_v1"
            ),
        ]
    )
    