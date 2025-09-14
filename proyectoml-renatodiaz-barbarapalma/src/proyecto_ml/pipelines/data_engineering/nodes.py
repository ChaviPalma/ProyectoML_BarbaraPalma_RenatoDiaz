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
