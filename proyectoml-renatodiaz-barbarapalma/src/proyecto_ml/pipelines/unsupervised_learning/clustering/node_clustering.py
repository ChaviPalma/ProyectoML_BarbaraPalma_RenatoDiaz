import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt



def entrenamiento_clustering(final_anime_dataset: pd.DataFrame, parametros_clustering: dict) -> tuple:
    
    try:
        df_animes = final_anime_dataset.drop_duplicates(subset=['id_anime']).copy()
    except NameError:
        # df_animes = pd.read_parquet("final_animedataset (1).parquet").drop_duplicates(subset=['IDAnime'])
        print("Asegúrate de haber cargado el dataset en la variable 'df' o 'anime'.")

    print(f" Animes únicos iniciales: {len(df_animes)}")

    features_cols = parametros_clustering["features_cols"]

    # --- CORRECCIÓN IMPORTANTE: FORZAR NUMÉRICO ---
    print(" Limpiando valores 'UNKNOWN' y textos extraños...")
    for col in features_cols:
        if col in df_animes.columns:
            df_animes[col] = pd.to_numeric(df_animes[col], errors='coerce')
        else:
            print(f"Advertencia: La columna '{col}' no se encontró en df_animes. Se omitirá.")

    df_model = df_animes.dropna(subset=features_cols).copy()
    print(f" Datos limpios finales: {len(df_model)} animes listos para usar.")

    if len(df_model) > 0:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_model[features_cols])
        print(f" Escalado exitoso. Shape final: {X_scaled.shape}")
    else:
        print(" ¡Error! Te has quedado sin datos después de limpiar. Revisa tus columnas.")
    
    K_FINAL = parametros_clustering["K_FINAL"] 

    print(f" Iniciando entrenamiento de 5 modelos con K={K_FINAL} (donde aplique)...")

    modelos_entrenados = []  # <-- lista donde guardaremos (nombre, modelo)

    # 1. K-MEANS (Algoritmo Principal - Particional)
    print("1️ Entrenando K-Means...")
    kmeans = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
    df_model['cluster_kmeans'] = kmeans.fit_predict(X_scaled)
    modelos_entrenados.append(("kmeans", kmeans))

    # 2. CLUSTERING JERÁRQUICO (Aglomerativo)
    print("2️ Entrenando Jerárquico...")
    plt.figure(figsize=(10, 4))
    dendrogram(linkage(X_scaled[:1000], method='ward'))

    try:
        hierarchical = AgglomerativeClustering(n_clusters=K_FINAL)
        df_model['cluster_hierarchical'] = hierarchical.fit_predict(X_scaled)
        modelos_entrenados.append(("hierarchical", hierarchical))
    except MemoryError:
        print(" Memoria insuficiente para Jerárquico completo. Se omite.")

    # 3. DBSCAN (Basado en Densidad)
    print("3️ Entrenando DBSCAN...")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df_model['cluster_dbscan'] = dbscan.fit_predict(X_scaled)
    modelos_entrenados.append(("dbscan", dbscan))

    # 4. GAUSSIAN MIXTURE MODELS (Probabilístico)
    print("4 Entrenando GMM...")
    gmm = GaussianMixture(n_components=K_FINAL, random_state=42)
    gmm.fit(X_scaled)
    df_model['cluster_gmm'] = gmm.predict(X_scaled)
    modelos_entrenados.append(("gmm", gmm))

    # 5. OPTICS (Densidad variable - Alternativa avanzada a DBSCAN)
    print("5️ Entrenando OPTICS...")
    optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)
    df_model['cluster_optics'] = optics.fit_predict(X_scaled)
    modelos_entrenados.append(("optics", optics))

    print("\n ¡Los 5 modelos han sido entrenados y guardados en el DataFrame!")
    print("Columnas generadas:", [col for col in df_model.columns if 'cluster_' in col])

    # Retornar el DataFrame y la lista de modelos entrenados
    return df_model, modelos_entrenados, X_scaled