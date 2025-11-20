import pandas as pd
import plotly.express as px
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D 

def graficos_clustering(df_model: pd.DataFrame) -> tuple[Figure, Figure]:
    clusters = df_model['cluster_kmeans'].unique()
    colores = plt.cm.get_cmap('tab10', len(clusters))

    # --- Figura 2D ---
    fig2d, ax2d = plt.subplots(figsize=(10, 6))
    for i, cluster in enumerate(sorted(clusters)):
        subset = df_model[df_model['cluster_kmeans'] == cluster]
        ax2d.scatter(
            subset['umap_1'],
            subset['umap_2'],
            label=f'Cluster {cluster}',
            alpha=0.7,
            color=colores(i)
        )
    ax2d.set_title('Mapa de Clusters de Anime (UMAP)')
    ax2d.set_xlabel('UMAP 1')
    ax2d.set_ylabel('UMAP 2')
    ax2d.legend(title='Cluster')
    ax2d.grid(True)

    # --- Figura 3D ---
    fig3d = plt.figure(figsize=(10, 6))
    ax3d = fig3d.add_subplot(111, projection='3d')
    for i, cluster in enumerate(sorted(clusters)):
        subset = df_model[df_model['cluster_kmeans'] == cluster]
        ax3d.scatter(
            subset['pca_1'],
            subset['pca_2'],
            subset['pca_3'],
            label=f'Cluster {cluster}',
            alpha=0.7,
            color=colores(i)
        )
    ax3d.set_title('Espacio PCA 3D')
    ax3d.set_xlabel('PCA 1')
    ax3d.set_ylabel('PCA 2')
    ax3d.set_zlabel('PCA 3')
    ax3d.legend(title='Cluster')

    return fig2d, fig3d



def evaluacion_modelos_clustering(X_scaled: pd.DataFrame, df_model: pd.DataFrame) -> dict:
  
    def evaluar_modelo(nombre_algo, labels, data):
        # Filtracion de ruido (-1) si aplica
        mask = labels != -1

        # Si el modelo clasificó todo como ruido o solo encontró 1 cluster, no se puede calcular
        if sum(mask) < 2 or len(set(labels[mask])) < 2:
            return {'Algoritmo': nombre_algo, 'Status': 'Fallido (Puro ruido o 1 cluster)'}

        X_limpio = data[mask]
        labels_limpio = labels[mask]

        if len(X_limpio) > 10000:
            idx = np.random.choice(len(X_limpio), 5000, replace=False)
            sil = silhouette_score(X_limpio[idx], labels_limpio[idx])
        else:
            sil = silhouette_score(X_limpio, labels_limpio)

        return {
            'Algoritmo': nombre_algo,
            'Clusters Encontrados': len(set(labels_limpio)),
            'Silhouette (Alto es mejor)': sil,
            'Calinski-Harabasz (Alto es mejor)': calinski_harabasz_score(X_limpio, labels_limpio),
            'Davies-Bouldin (Bajo es mejor)': davies_bouldin_score(X_limpio, labels_limpio)
        }

    # Recopilar resultados
    resultados = []
    columnas_clusters = [col for col in df_model.columns if 'cluster_' in col]

    print("📊 Calculando métricas para todos los modelos...")
    for col in columnas_clusters:
        nombre_bonito = col.replace('cluster_', '').upper()
        res = evaluar_modelo(nombre_bonito, df_model[col], X_scaled)
        resultados.append(res)

    return {res['Algoritmo']: res for res in resultados}


def mapa_calor_clustering(df_model: pd.DataFrame, parametros_clustering: dict) -> Figure:
    perfil = df_model.groupby('cluster_kmeans')[parametros_clustering["features_cols"]].mean()
    perfil['Cantidad_Animes'] = df_model['cluster_kmeans'].value_counts()
    perfil = perfil.sort_index()

    perfil_norm = (perfil - perfil.min()) / (perfil.max() - perfil.min())

    # Crear figura y eje explícitamente
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        perfil_norm.drop(columns=['Cantidad_Animes']),
        annot=True,
        cmap='YlGnBu',
        fmt=".2f",
        ax=ax
    )
    ax.set_title('Mapa de Calor: Características distintivas de cada Cluster')

    return fig
