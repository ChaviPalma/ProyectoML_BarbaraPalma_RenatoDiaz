# src/proyecto_ml/pipelines/data_science/node_regresion.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Imports para el gráfico ---
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.figure import Figure # Para el type hint

# --- Función de preprocesamiento (la que ya tienes) ---
def preprocesar_datos(final_anime_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesa datos usando los nombres de columna correctos (rating_x, genres, etc.)
    y crea las features faltantes (is_movie, FechaEstreno_year, etc.)
    """
    
    anime_processed = final_anime_dataset.copy()
    
    # 1. Extraer Target (usando el nombre correcto)
    target_variable = 'rating_x' # <-- CAMBIO: Nombre de columna correcto
    if target_variable not in anime_processed.columns:
        raise KeyError(f"La columna target '{target_variable}' no se encuentra en el DataFrame.")
    y = anime_processed[target_variable]
    y_df = y.to_frame()

    # 2. Binarizar Géneros (usando 'genres')
    genre_col = 'genres' # <-- CAMBIO: Nombre de columna correcto
    if genre_col not in anime_processed.columns:
         raise KeyError(f"La columna '{genre_col}' no se encuentra.")

    listas_de_generos = anime_processed[genre_col].fillna('').str.split(',')
    mlb = MultiLabelBinarizer()
    transformed_data = mlb.fit_transform(listas_de_generos)
    
    column_names = [f"Genero_{g.strip()}" for g in mlb.classes_ if g.strip()]
    
    gender_encoded_df = pd.DataFrame(
         transformed_data,
         columns=column_names,
         index=anime_processed.index
    )
    anime_processed = pd.concat([anime_processed.drop(columns=[genre_col], errors='ignore'), gender_encoded_df], axis=1)

    # 3. Crear Features Faltantes
    
    # Fecha desde 'premiered'
    if 'premiered' in anime_processed.columns:
        anime_processed['FechaEstreno_year'] = pd.to_numeric(
            anime_processed['premiered'].astype(str).str.extract(r'(\d{4})', expand=False),
            errors='coerce'
        ).fillna(0)
    else:
        anime_processed['FechaEstreno_year'] = 0

    # Features desde 'episodes'
    if 'episodes' in anime_processed.columns:
        eps = pd.to_numeric(anime_processed['episodes'], errors='coerce').fillna(0)
        anime_processed['is_movie'] = (eps == 1)
        anime_processed['is_long_series'] = (eps > 26)
    else:
        anime_processed['is_movie'] = False
        anime_processed['is_long_series'] = False

    # Features desde 'members'
    if 'members' in anime_processed.columns:
        members_num = pd.to_numeric(anime_processed['members'], errors='coerce').fillna(0)
        thr_members = members_num.quantile(0.75)
        anime_processed['is_popular'] = (members_num >= thr_members)
    else:
        anime_processed['is_popular'] = False

    # Features desde 'score'
    if 'score' in anime_processed.columns:
        score_num = pd.to_numeric(anime_processed['score'], errors='coerce').fillna(0)
        thr_score = score_num.quantile(0.75)
        anime_processed['is_highly_rated'] = (score_num >= thr_score)
    else:
        anime_processed['is_highly_rated'] = False

    # 4. Definir lista final de features (con nombres correctos)
    feature_columns = [
        'score',        # <-- CAMBIO: PuntuacionAnime
        'episodes',     # <-- CAMBIO: Episodios
        'ranked',       # <-- CAMBIO: Ranking
        'popularity',   # <-- CAMBIO: Popularidad
        'favorites',    # <-- CAMBIO: Favoritos
        'members',      # <-- CAMBIO: CantidadDeMiembros
        'FechaEstreno_year',
        'is_movie', 
        'is_long_series', 
        'is_popular', 
        'is_highly_rated',
    ]
    gender_cols = [col for col in column_names if col in gender_encoded_df.columns]
    feature_columns.extend(gender_cols)
    
    final_feature_columns = [col for col in feature_columns if col in anime_processed.columns]

    if not final_feature_columns:
        existing_cols = anime_processed.columns.tolist()
        raise ValueError(f"La lista final de features está vacía. Columnas disponibles: {existing_cols}")

    X = anime_processed[final_feature_columns]
    X = X.fillna(0)
    for col in X.columns:
         X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)

    print(f"Preprocesamiento completado. X tiene {X.shape[1]} características.")
    return X, y_df

# --- Función de división (la que ya tienes) ---
def dividir_datos(X: pd.DataFrame, y: pd.DataFrame, test_size=0.2, random_state=42) -> tuple:
    # Convertir y a Serie antes de dividir si aún es DataFrame
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y_series = y.iloc[:, 0]
    else:
        y_series = y # Asume que ya es Serie o el usuario sabe lo que hace

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_series, test_size=test_size, random_state=random_state # Pasar y_series
    )
    print("Datos divididos en entrenamiento y prueba.")
    # Devolver y_train/y_test como DataFrames de nuevo para consistencia con catalog
    return X_train, X_test, y_train.to_frame(), y_test.to_frame()

# --- Funciones específicas para RLS ---
def seleccionar_feature_slr(X: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """Selecciona una única columna feature para la Regresión Lineal Simple."""
    print(f"Seleccionando feature para RLS: {feature_name}")
    if feature_name not in X.columns:
        raise KeyError(f"La feature '{feature_name}' no se encuentra en X. Columnas disponibles: {X.columns.tolist()}")
    return X[[feature_name]]

def entrenar_modelo_slr(X_slr: pd.DataFrame, y_train: pd.DataFrame) -> LinearRegression:
    """Entrena un modelo de Regresión Lineal Simple."""
    print("Entrenando modelo RLS...")
    if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
        y_train_series = y_train.iloc[:, 0]
    else: y_train_series = y_train
    modelo_slr = LinearRegression()
    modelo_slr.fit(X_slr, y_train_series)
    print(f"RLS - Intercepto: {modelo_slr.intercept_}")
    print(f"RLS - Coeficiente: {modelo_slr.coef_[0]}")
    print("Modelo RLS entrenado.")
    return modelo_slr

# --- FUNCIÓN MODIFICADA ---
def evaluar_modelo_slr(
    modelo_slr: LinearRegression,
    X_test_slr: pd.DataFrame,
    y_test: pd.DataFrame
) -> tuple[dict, pd.DataFrame]: # <-- CAMBIO: Devuelve DataFrame
    """Evalúa el modelo RLS y devuelve métricas y predicciones COMO DATAFRAME."""
    print("Evaluando modelo RLS...")
    if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1:
        y_test_series = y_test.iloc[:, 0]
    else: y_test_series = y_test

    # Obtener predicciones (sigue siendo un array numpy aquí)
    y_pred_slr_np = modelo_slr.predict(X_test_slr) 

    # Calcular métricas
    mae_slr = mean_absolute_error(y_test_series, y_pred_slr_np)
    mse_slr = mean_squared_error(y_test_series, y_pred_slr_np)
    rmse_slr = np.sqrt(mse_slr)
    r2_slr = r2_score(y_test_series, y_pred_slr_np)
    metrics = {"slr_mae": mae_slr, "slr_mse": mse_slr, "slr_rmse": rmse_slr, "slr_r2": r2_slr}

    print("Métricas de evaluación RLS:")
    print(f"  MAE: {mae_slr:.4f}")
    print(f"  MSE: {mse_slr:.4f}")
    print(f"  RMSE: {rmse_slr:.4f}")
    print(f"  R²: {r2_slr:.4f}")

    # --- NUEVO: Convertir predicciones a DataFrame ---
    # Usar el índice de y_test para alinear las predicciones
    predicciones_df = pd.DataFrame({'predicciones': y_pred_slr_np.flatten()}, index=y_test.index)
    # ------------------------------------------------

    return metrics, predicciones_df # <-- CAMBIO: Devuelve el DataFrame

# --- NUEVA FUNCIÓN PARA EL GRÁFICO ---
def crear_grafico_qq_residuos(y_test: pd.DataFrame, y_pred_slr: pd.DataFrame) -> Figure: # <-- Input y_pred_slr es DataFrame
    """Crea un gráfico Q-Q de los residuos del modelo."""
    print("Creando gráfico Q-Q de residuos...")
    
    # Asegurar que y_test sea una Serie
    if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1:
        y_test_series = y_test.iloc[:, 0]
    else: y_test_series = y_test # Asume que ya es Serie

    # --- CAMBIO AQUÍ ---
    # Extraer los valores de la columna 'predicciones' del DataFrame y_pred_slr
    if 'predicciones' in y_pred_slr.columns:
         y_pred_values = y_pred_slr['predicciones'].values
    else:
        # Fallback si la columna tiene otro nombre (poco probable)
        y_pred_values = y_pred_slr.iloc[:, 0].values 
        
    # Calcular residuos: array numpy - array numpy
    residuals_slr = y_test_series.values - y_pred_values
    # Ya no necesitamos el try-except
    # ------------------

    fig, ax = plt.subplots(figsize=(7, 5))
    stats.probplot(residuals_slr, dist="norm", plot=ax)
    ax.set_title("Gráfico Q-Q de Residuos (Regresión Lineal Simple)")
    ax.set_xlabel("Cuantiles Teóricos")
    ax.set_ylabel("Cuantiles de los Residuos")
    ax.grid(True)
    print("Gráfico Q-Q creado.")
    return fig