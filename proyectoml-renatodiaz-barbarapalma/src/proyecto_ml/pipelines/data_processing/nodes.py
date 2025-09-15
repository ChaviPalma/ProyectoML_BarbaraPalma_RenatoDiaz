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

