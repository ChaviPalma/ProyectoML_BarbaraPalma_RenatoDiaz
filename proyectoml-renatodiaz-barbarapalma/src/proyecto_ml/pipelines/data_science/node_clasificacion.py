import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint
import warnings
import gc

def preprocesar_anime_dataset_clasificacion(final_anime_dataset: pd.DataFrame) -> pd.DataFrame:


    # obtener las columnas de géneros
    genre_cols = [col for col in final_anime_dataset.columns if col in ['Action', 'Adventure', 'Cars', 'Comedy', 'Dementia', 'Demons', 'Drama', 'Fantasy', 'Game', 'Historical', 'Horror', 'Josei', 'Kids', 'Magic', 'Martial Arts', 'Mecha', 'Military', 'Music', 'Mystery', 'Parody', 'Police', 'Psychological', 'Romance', 'Samurai', 'School', 'Sci-Fi', 'Seinen', 'Shoujo', 'Shoujo Ai', 'Shounen', 'Shounen Ai', 'Slice of Life', 'Space', 'Sports', 'Super Power', 'Supernatural', 'Thriller', 'Vampire']]

    # crear una columna temporal que contenga la lista de géneros para cada anime
    final_anime_dataset['GenerosAnime_list'] = final_anime_dataset[genre_cols].apply(
        lambda row: [col for col in genre_cols if row[col] == 1], axis=1
    )

    # Expandir el DataFrame para que cada fila represente un anime y un género asociado
    anime_exploded_genres = final_anime_dataset.explode('GenerosAnime_list')

    # group by user y género para contar la cantidad de veces que un usuario ha interactuado con cada género
    user_genre_counts = anime_exploded_genres.groupby(['id_usuario', 'GenerosAnime_list']).size().reset_index(name='genre_count')
    # para cada usuario, obtener el género con la mayor cantidad de interacciones
    user_preferred_genres = user_genre_counts.loc[user_genre_counts.groupby('id_usuario')['genre_count'].idxmax()]

    # renombrar columnas para mayor claridad
    user_preferred_genres.rename(columns={'GenerosAnime_list': 'genero_preferido', 'genre_count': 'cantidad_genero_preferido'}, inplace=True)

    final_anime_dataset = pd.merge(final_anime_dataset, user_preferred_genres[['id_usuario', 'genero_preferido']], on='id_usuario', how='left')

        # Calcular la puntuación promedio de cada usuario
    user_average_rating = final_anime_dataset.groupby('id_usuario')['puntuacion_usuario'].mean().reset_index(name='puntuacion_promedio_usuario')

    # Eliminar columnas duplicadas de fusiones anteriores si existen
    cols_to_drop = ['User_Average_Rating_x', 'User_Average_Rating_y', 'User_Average_Rating']
    existing_cols_to_drop = [col for col in cols_to_drop if col in final_anime_dataset.columns]
    if existing_cols_to_drop:
        final_anime_dataset = final_anime_dataset.drop(columns=existing_cols_to_drop)

    # Fusionar el DataFrame 'final_anime_dataset' con las puntuaciones promedio de los usuarios
    final_anime_dataset = pd.merge(final_anime_dataset, user_average_rating, on='id_usuario', how='left')

    final_anime_dataset['Matches_Preferred_Genre'] = final_anime_dataset.apply(
        lambda row: 1 if row['genero_preferido'] in row['GenerosAnime_list'] else 0,
        axis=1
    )

    # Aplicar One-Hot Encoding a la columna 'genero_preferido'
    anime_encoded = pd.get_dummies(final_anime_dataset, columns=['genero_preferido'], dummy_na=False)

    # Actualizar el dataframe original con las nuevas columnas
    final_anime_dataset = anime_encoded


    return final_anime_dataset

def Entrenar_modelo_clasificacion( 
    final_anime_dataset: pd.DataFrame, 
    parametros_clasificacion: dict
) -> tuple:
    """
    Entrena varios modelos de clasificación.
    Implementa Standard Scaling y retorna los modelos, X_test (escalado) y y_test.
    """

    umbral_de_interes = 7
    final_anime_dataset['Interesado'] = (final_anime_dataset['puntuacion_usuario'] > umbral_de_interes).astype(int)
    y = final_anime_dataset['Interesado']
    
    features_to_drop = [
        'nombre_usuario', 'id_anime', 'titulo_anime', 'puntuacion_usuario', 'nombre_anime',
        'puntuacion', 'sinopsis', 'tipo_anime_filtered', 'total_episodios', 'emitido',
        'fecha_estreno', 'estudios', 'fuente', 'duracion', 'posicion_anime',
        'popularidad', 'miembros', 'favoritos', 'viendo', 'completado',
        'en_espera', 'abandonado', 'duracion_minutos', 'GenerosAnime_list',
        'genero_preferido', 'User_Average_Rating', 'Interesado'
    ]
    df_temp = final_anime_dataset.copy()
    existing_features_to_drop = [col for col in features_to_drop if col in df_temp.columns]
    X = df_temp.drop(columns=existing_features_to_drop, errors='ignore')
    X = X.select_dtypes(include=['number', 'bool'])
    
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)
        
    combined = pd.concat([X, y], axis=1).dropna()
    X = combined.drop(columns=y.name)
    y = combined[y.name]
    
    if X.empty or X.shape[1] == 0:
        raise ValueError("La matriz de características (X) está vacía.")
    
    print(f" X final contiene {X.shape[1]} características: {X.columns.tolist()[:5]}...")

    # --- 1. DIVISIÓN DE DATOS ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=parametros_clasificacion["test_size"],
        random_state=parametros_clasificacion["random_state"],
        stratify=y
    )

    # --- 2. ESCALADO Y RECONSTRUCCIÓN ---
    scaler = StandardScaler()
    X_train_scaled_array = scaler.fit_transform(X_train)
    X_test_scaled_array = scaler.transform(X_test)
    
    # Se sobrescribe X_test con la versión escalada para el retorno
    X_train = pd.DataFrame(X_train_scaled_array, columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled_array, columns=X_test.columns, index=X_test.index) 
    # --- 3. DEFINICIÓN Y ENTRENAMIENTO DE MODELOS ---
    modelos = {
        "LogisticRegression": (LogisticRegression(max_iter=10000, solver='liblinear', random_state=parametros_clasificacion["random_state"]), {
            "C": uniform(0.01, 10), "penalty": ['l1', 'l2']
        }),
        "SGDClassifier": (SGDClassifier(random_state=parametros_clasificacion["random_state"], max_iter=10000), {
            "loss": ['hinge', 'log_loss'], "alpha": uniform(0.0001, 0.01),
            "penalty": ['l2', 'l1', 'elasticnet']
        }),
        "RandomForest": (RandomForestClassifier(random_state=parametros_clasificacion["random_state"]), {
            "n_estimators": randint(50, 100), "max_depth": randint(3, 10)
        }),
        "DecisionTree": (DecisionTreeClassifier(random_state=parametros_clasificacion["random_state"]), {
            "max_depth": randint(2, 10), "min_samples_split": randint(2, 10),
            "criterion": ['gini', 'entropy']
        }),
        "KNeighborsClassifier": (KNeighborsClassifier(), {
            "n_neighbors": randint(3, 20), "weights": ['uniform', 'distance']
        })
    }

    modelos_entrenados = {}
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        
        for nombre, (modelo, distribucion) in modelos.items():
            
            X_train_data = X_train 
            
            if distribucion:
                print(f" Buscando mejores hiperparámetros para {nombre}...")
                search = RandomizedSearchCV(
                    modelo, distribucion,
                    n_iter=parametros_clasificacion.get("n_iter", 5),
                    cv=parametros_clasificacion.get("cv", 5),
                    random_state=parametros_clasificacion["random_state"],
                    n_jobs=1,
                    scoring='accuracy'
                )
                search.fit(X_train_data, y_train)
                mejor_modelo = search.best_estimator_
                
            else:
                mejor_modelo = modelo.fit(X_train_data, y_train)
                
            modelos_entrenados[nombre] = mejor_modelo
            print(f" {nombre} entrenado.")

    print(" Entrenamiento de todos los modelos de clasificación completado.")
    gc.collect()
    y_test = y_test.to_frame()
    # Se retorna X_test que ahora contiene los datos ESCALADOS
    return modelos_entrenados, X_test, y_test