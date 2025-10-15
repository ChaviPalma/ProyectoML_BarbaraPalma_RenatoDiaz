import pandas as pd
import numpy as np

##Anime_dataset

##Limpia las tablas innecesarias

def limpiar_anime_dataset(anime_dataset: pd.DataFrame) -> pd.DataFrame:
    # columnas que no necesitas
    columns_to_drop = [
        'English name', 'Other name',
        'Producers', 'Licensors',
        'Studios', 'Image URL'
    ]
    
    # eliminamos columnas
    anime_dataset = anime_dataset.drop(columns=columns_to_drop, errors="ignore")
    
    return anime_dataset

#Cambiamos los datos UNKNOWN dentro de episodios a 0 y cambiamos el tipo a numerico
def limpiar_convertir_episodes(anime_dataset: pd.DataFrame) -> pd.DataFrame:
    # Reemplazar 'UNKNOWN' por 0
    anime_dataset['Episodes'] = anime_dataset['Episodes'].replace('UNKNOWN', 0)
    # Convertir a numérico, coerción de errores, rellenar NaN con 0 antes de convertir a int
    anime_dataset['Episodes'] = pd.to_numeric(anime_dataset['Episodes'], errors='coerce').fillna(0).astype(int)
    
    return anime_dataset

#Modificación de la columna rank, se cambiaron los datos UNKNOWN por 0 y se cambio el tipo de category a int para una posible regresión
def limpiar_convertir_rank(anime_dataset: pd.DataFrame) -> pd.DataFrame:
    # Reemplazar 'UNKNOWN' por 0
    anime_dataset['Rank'] = anime_dataset['Rank'].replace('UNKNOWN', 0)
    # Convertir a numérico, coerción de errores, rellenar NaN con 0 antes de convertir a int
    anime_dataset['Rank'] = pd.to_numeric(anime_dataset['Rank'], errors='coerce').fillna(0).astype(int)
    
    return anime_dataset

##users_detail

def inspeccionar_users_detail(users_detail: pd.DataFrame) -> pd.DataFrame:
    """Imprime el head y la info del dataset."""
    print("=== Head del dataset ===")
    print(users_detail.head())
    print("\n=== Información del dataset ===")
    print(users_detail.info())
    return users_detail


def limpiar_users_detail(users_detail: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas innecesarias de users_detail."""
    columns_a_eliminar = [
        'Location', 'Joined', 'Birthday', 'Days Watched', 'Total Entries',
        'On Hold', 'Dropped', 'Watching', 'Completed', 'Plan to Watch',
        'Rewatched', 'Episodes Watched', 'Mean Score'
    ]
    users_detail = users_detail.drop(columns=[col for col in columns_a_eliminar if col in users_detail.columns])
    
    print("=== Columnas restantes ===")
    print(users_detail.columns)
    
    return users_detail


##users_score

def inspeccionar_users_score(users_score: pd.DataFrame) -> pd.DataFrame:
   
    print("=== Información del dataset ===")
    print(users_score.info())
    
    print("\n=== Conteo de ratings ===")
    rating_counts = users_score['rating'].value_counts()
    print(rating_counts)
    
    print("\n=== Valores nulos por columna ===")
    print(users_score.isnull().sum())
    
    return users_score

##Union de datasets


#Union de datastets users_score y users_detail
def union_dataset_score_detail(users_score: pd.DataFrame, users_detail: pd.DataFrame) -> pd.DataFrame:
    final_users = pd.merge(users_detail, users_score, left_on='Mal ID', right_on='user_id', how='inner')
    #Eliminacion de columnas innecesarias
    columns_a_eliminar = ['Mal ID', 'Username_y']
    final_users = final_users.drop(columns=columns_a_eliminar)
    return final_users

# === CREA LAS FEATURES BÁSICAS PRIMERO ===
def create_basic_anime_features(df: pd.DataFrame) -> pd.DataFrame:
    anime_features_df = df.copy()

    # --- Feature 1: Tipo ---
    if 'Type' in anime_features_df.columns:
        if not pd.api.types.is_object_dtype(anime_features_df['Type']):
            anime_features_df['Type'] = anime_features_df['Type'].astype(str)
        anime_features_df = pd.get_dummies(anime_features_df, columns=['Type'], prefix='type', dummy_na=False)

    # --- Feature 2: Duración (episodios) ---
    if 'Episodes' in anime_features_df.columns:
        anime_features_df['Episodes'] = pd.to_numeric(anime_features_df['Episodes'], errors='coerce').fillna(0)
        anime_features_df['is_movie'] = (anime_features_df['Episodes'] == 1).astype(int)
        anime_features_df['is_long_series'] = (anime_features_df['Episodes'] > 26).astype(int)

    # --- Feature 3: Popularidad ---
    if 'Members' in anime_features_df.columns:
        anime_features_df['Members'] = pd.to_numeric(anime_features_df['Members'], errors='coerce').fillna(0)
        threshold = anime_features_df['Members'].quantile(0.75)
        anime_features_df['is_popular'] = (anime_features_df['Members'] >= threshold).astype(int)

    # --- Feature 4: Puntuación ---
    if 'Score' in anime_features_df.columns:
        anime_features_df['Score'] = pd.to_numeric(anime_features_df['Score'], errors='coerce').fillna(0)
        threshold = anime_features_df['Score'].quantile(0.75)
        anime_features_df['is_highly_rated'] = (anime_features_df['Score'] >= threshold).astype(int)

    return anime_features_df


# === AHORA SE MODIFICA LA UNIÓN ===
def union_dataset_anime_users(anime_dataset: pd.DataFrame, final_users: pd.DataFrame) -> pd.DataFrame:
    # 🟢 APLICAR LAS FEATURES AQUÍ ANTES DE HACER EL MERGE
    anime_dataset = create_basic_anime_features(anime_dataset)

    final_anime_dataset = pd.merge(anime_dataset, final_users, left_on='anime_id', right_on='anime_id', how='inner')

    # Eliminar la columna 'Name' ya que 'Anime Title' es similar
    if 'Name' in final_anime_dataset.columns:
        final_anime_dataset = final_anime_dataset.drop(columns=['Name'])

    # Reordenar columnas: primero las de usuario
    user_columns = [col for col in final_users.columns if col != 'anime_id']
    anime_columns = [col for col in anime_dataset.columns if col not in ['anime_id', 'Name']]

    new_column_order = user_columns + ['anime_id'] + anime_columns
    final_anime_dataset = final_anime_dataset[new_column_order]

    nuevos_nombres_columnas = {
        'Username_x': 'NombreUsuario',
        'Gender': 'GeneroUsuario',
        'user_id': 'IDUsuario',
        'Anime Title': 'TituloAnime',
        'rating': 'Puntuacion',
        'anime_id': 'IDAnime',
        'Score': 'PuntuacionAnime',
        'Genres': 'GenerosAnime',
        'Synopsis': 'Sinopsis',
        'Type': 'Tipo',
        'Episodes': 'Episodios',
        'Aired': 'FechaEmision',
        'Premiered': 'FechaEstreno',
        'Status': 'Estado',
        'Source': 'Fuente',
        'Duration': 'Duración',
        'Rating': 'Clasificación',
        'Rank': 'Ranking',
        'Popularity': 'Popularidad',
        'Favorites': 'Favoritos',
        'Scored By': 'PuntuadoPor',
        'Members': 'CantidadDeMiembros'
    }

    final_anime_dataset = final_anime_dataset.rename(columns=nuevos_nombres_columnas)
    return final_anime_dataset