import pandas as pd

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


def union_dataset_anime_users(anime_dataset: pd.DataFrame, final_users: pd.DataFrame) -> pd.DataFrame:
    final_anime_dataset = pd.merge(anime_dataset, final_users, left_on='anime_id', right_on='anime_id', how='inner')

    # Eliminar la columna 'Name' ya que 'Anime Title' es similar
    final_anime_dataset = final_anime_dataset.drop(columns=['Name'])

    # Reordenar las columnas para poner las de usuario primero
    user_columns = [col for col in final_users.columns if col != 'anime_id']
    # Identificar las columnas de anime (excluyendo 'anime_id' y 'Name' que eliminamos)
    anime_columns = [col for col in anime_dataset.columns if col != 'anime_id' and col != 'Name']

    # Crear la nueva lista de orden de columnas: columnas de usuario + 'anime_id' + columnas de anime
    new_column_order = user_columns + ['anime_id'] + anime_columns

    # Aplicar el nuevo orden de columnas al DataFrame
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

    # Renombrar las columnas
    final_anime_dataset = final_anime_dataset.rename(columns=nuevos_nombres_columnas)

    return final_anime_dataset