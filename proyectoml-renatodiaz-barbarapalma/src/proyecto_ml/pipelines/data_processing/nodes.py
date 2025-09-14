import pandas as pd

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

import pandas as pd

def limpiar_convertir_episodes(anime_dataset: pd.DataFrame) -> pd.DataFrame:
    # Reemplazar 'UNKNOWN' por 0
    anime_dataset['Episodes'] = anime_dataset['Episodes'].replace('UNKNOWN', 0)
    # Convertir a numérico, coerción de errores, rellenar NaN con 0 antes de convertir a int
    anime_dataset['Episodes'] = pd.to_numeric(anime_dataset['Episodes'], errors='coerce').fillna(0).astype(int)
    return anime_dataset

def convertir_premiered_booleano(anime_dataset: pd.DataFrame) -> pd.DataFrame:
    # Reemplazar 'UNKNOWN' por False, otros por True
    anime_dataset.loc[:, 'Premiered'] = anime_dataset['Premiered'] != 'UNKNOWN'
    return anime_dataset