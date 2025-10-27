import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocesar_datos(anime):
    # Copia para evitar modificar el original
    anime_processed = anime.copy()

    # 1. Definir el Target (y)
    target_variable = 'Puntuacion'
    y = anime_processed[target_variable]

    # 2. Ingeniería de Características (X)
    
    # Manejo de Fechas
    anime_processed['FechaEstreno'] = pd.to_datetime(anime_processed['FechaEstreno'], errors='coerce', format='%Y-%m-%d')
    anime_processed['FechaEstreno_year'] = anime_processed['FechaEstreno'].dt.year.fillna(0).astype(int)

    # One-Hot Encoding para Género
    gender_encoded = pd.get_dummies(anime_processed['GeneroUsuario'], prefix='GeneroUsuario', dummy_na=False)
    anime_processed = pd.concat([anime_processed, gender_encoded], axis=1)

    # 3. Definir Columnas de Features (X)
    feature_columns = [
        'PuntuacionAnime_numeric', 'Episodios', 'Ranking', 'Popularidad', 
        'Favoritos', 'CantidadDeMiembros', 'FechaEstreno_year', 
        'episodes_numeric', 'is_movie', 'is_long_series', 'is_popular', 
        'is_highly_rated'
    ]
    
    # Añadir columnas de género de forma segura
    gender_cols = [col for col in gender_encoded.columns if col in anime_processed.columns]
    feature_columns.extend(gender_cols)

    # Filtrar columnas que realmente existen en el DataFrame final
    final_feature_columns = [col for col in feature_columns if col in anime_processed.columns]
    X = anime_processed[final_feature_columns]
    
    print(f"Preprocesamiento completado. X tiene {X.shape[1]} características.")
    
    return X, y

def dividir_datos(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print("Datos divididos en entrenamiento y prueba.")
    return X_train, X_test, y_train, y_test

