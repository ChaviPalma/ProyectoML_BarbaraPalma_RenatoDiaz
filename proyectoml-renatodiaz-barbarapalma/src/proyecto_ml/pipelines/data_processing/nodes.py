import pandas as pd
import numpy as np
import gc
import re

def procesar_anime_filtered(anime_filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa el DataFrame 'anime_filtered_filtered':
      - Renombra columnas.
      - Elimina géneros específicos.
      - Convierte la duración a minutos.
      - Elimina columnas innecesarias.
    Retorna el DataFrame procesado.
    """

    # --- Mapeo de columnas ---
    anime_filtered_column_mapping = {
        'anime_id': 'id_anime',
        'Name': 'nombre_anime',
        'Score': 'puntuacion',
        'Genres': 'generos_anime',
        'English name': 'Nombre_Ingles',
        'Japanese name': 'Nombre_Japones',
        'sypnopsis': 'sinopsis',
        'Type': 'tipo_anime_filtered',
        'Episodes': 'total_episodios',
        'Aired': 'emitido',
        'Premiered': 'fecha_estreno',
        'Producers': 'Productores',
        'Licensors': 'licenciantes',
        'Studios': 'estudios',
        'Source': 'fuente',
        'Duration': 'duracion',
        'Rating': 'clasificacion',
        'Ranked': 'posicion_anime',
        'Popularity': 'popularidad',
        'Members': 'miembros',
        'Favorites': 'favoritos',
        'Watching': 'viendo',
        'Completed': 'completado',
        'On-Hold': 'en_espera',
        'Dropped': 'abandonado'
    }

    anime_filtered= anime_filtered.rename(columns=anime_filtered_column_mapping)
    print("\n Columnas del dataframe 'anime_filtered' después de renombrar:")
    print(anime_filtered.columns)

    # --- Filtro de géneros no deseados ---
    genres_to_remove = ['Yuri', 'Yaoi', 'Harem', 'Hentai', 'Ecchi', 'Unknown']
    mask_to_remove = anime_filtered['generos_anime'].apply(
        lambda x: isinstance(x, str) and any(genre in x for genre in genres_to_remove)
    )

    anime_filtered = anime_filtered[~mask_to_remove].copy()

    print(f"\n Dimensiones después de eliminar géneros específicos: {anime_filtered.shape}")

    # --- Función interna: convertir duración a minutos ---
    def convert_duration_to_minutes(duration_str):
        """
        Convierte una cadena de duración a minutos.
        Maneja formatos como 'X min. per ep.', 'X hr. Y min.', 'X min.', 'X hr.', 'X sec.'.
        Retorna None para formatos no reconocidos o nulos.
        """
        if not isinstance(duration_str, str) or duration_str.lower() == 'unknown':
            return None

        duration_str = duration_str.lower()

        # "X min. per ep."
        match_min_per_ep = re.search(r'(\d+)\s*min\.\s*per\s*ep\.', duration_str)
        if match_min_per_ep:
            try:
                return int(match_min_per_ep.group(1))
            except ValueError:
                return None

        # "X hr. Y min." / "X hr." / "Y min." / "X sec."
        match_hr = re.search(r'(\d+)\s*hr\.', duration_str)
        match_min = re.search(r'(\d+)\s*min\.', duration_str)
        match_sec = re.search(r'(\d+)\s*sec\.', duration_str)

        hours = int(match_hr.group(1)) if match_hr else 0
        minutes = int(match_min.group(1)) if match_min else 0
        seconds = int(match_sec.group(1)) if match_sec else 0

        if hours > 0 or minutes > 0 or seconds > 0:
            return hours * 60 + minutes + round(seconds / 60, 2)

        return None

    # --- Aplicar conversión de duración ---
    anime_filtered['duracion_minutos'] = anime_filtered['duracion'].apply(convert_duration_to_minutes)
    print(" Columna 'duracion_minutos' creada y convertida a formato numérico.")

    # --- Eliminar columnas no necesarias ---
    columns_to_drop = ['Nombre_Ingles', 'Nombre_Japones', 'Productores', 'clasificacion', 'licenciantes', ]
    anime_filtered = anime_filtered.drop(columns=columns_to_drop, errors='ignore')

    # --- Mostrar resumen ---
    print("\n Resumen de 'duracion_minutos':")


    gc.collect()

    anime_filtered['total_episodios'] = pd.to_numeric(anime_filtered['total_episodios'], errors='coerce')
    # Eliminar filas donde 'duracion_minutos' es nulo
    anime_before_drop = anime_filtered.shape[0]
    anime_filtered = anime_filtered.dropna(subset=['duracion_minutos'])

    print(f"Dimensiones del dataframe 'anime' antes de eliminar nulos en 'duracion_minutos': ({anime_before_drop}, {anime_filtered.shape[1]})")
    print(f"Dimensiones del dataframe 'anime' después de eliminar nulos en 'duracion_minutos': {anime_filtered.shape}")
    return anime_filtered


def procesar_users_score(users_score: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa el DataFrame 'users_score':
      - Renombra columnas.
      - Elimina columnas innecesarias.  
    Retorna el DataFrame procesado.
    """
    users_score_column_mapping = {
    'user_id': 'id_usuario',
    'Username': 'nombre_usuario',
    'anime_id': 'id_anime',
    'Anime Title': 'titulo_anime',
    'rating': 'puntuacion_usuario'
    }   

    
    users_score = users_score.rename(columns=users_score_column_mapping)
    return users_score

def union_datasets(anime_filtered: pd.DataFrame, users_score: pd.DataFrame) -> pd.DataFrame:
    """
    Une los DataFrames 'anime_filtered' y 'users_score' en base a la columna 'id_anime'.
    Retorna el DataFrame unido.
    """
    final_anime_dataset = pd.merge(users_score, anime_filtered, on='id_anime', how='inner')


    print(f"\n Dimensiones del DataFrame final después de la unión: {final_anime_dataset.shape}")
    return final_anime_dataset

