from sklearn.decomposition import PCA
import pandas as pd
import umap

def reducir_dimensionalidad(df_model: pd.DataFrame, parametros_clustering: dict, X_scaled,) -> pd.DataFrame:
    print("📉 Ejecutando Reducción de Dimensionalidad...")

    # 1. PCA (Análisis de Componentes Principales)
    # Calculamos 3 componentes para poder hacer gráficos 3D si queremos
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X_scaled)

    df_model['pca_1'] = pca_result[:, 0]
    df_model['pca_2'] = pca_result[:, 1]
    df_model['pca_3'] = pca_result[:, 2]

    # Varianza Explicada (Requisito del PDF: varianza explicada)
    varianza = pca.explained_variance_ratio_
    print(f"📊 Varianza explicada por PCA: {sum(varianza):.2%} (PC1: {varianza[0]:.2%}, PC2: {varianza[1]:.2%})")

    # 2. UMAP (Proyección No Lineal - Mejor para separar visualmente)
    # Ajusta n_neighbors: valores bajos (5-10) localizan mejor detalles, altos (20-50) ven la estructura global
    reducer = umap.UMAP(n_neighbors= parametros_clustering["n_neighbors"], min_dist= parametros_clustering["min_dist"], random_state= parametros_clustering["random_state"])
    umap_result = reducer.fit_transform(X_scaled)

    df_model['umap_1'] = umap_result[:, 0]
    df_model['umap_2'] = umap_result[:, 1]

    print("✅ Reducción dimensional completada.")
    return df_model