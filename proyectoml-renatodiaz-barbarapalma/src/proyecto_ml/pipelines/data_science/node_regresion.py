import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import uniform, randint
import warnings 



def preprocesar_anime_dataset(final_anime_dataset: pd.DataFrame) -> pd.DataFrame:
    final_anime_dataset = final_anime_dataset.dropna(subset=['total_episodios'])
    final_anime_dataset = final_anime_dataset.dropna(subset=['posicion_anime'])
    final_anime_dataset['generos_anime'] = final_anime_dataset['generos_anime'].str.split(', ')
    # Luego, crear las columnas dummy para cada género
    generos_dummies = final_anime_dataset['generos_anime'].str.join('|').str.get_dummies()
    # Unir las nuevas columnas dummy al DataFrame original
    final_anime_dataset = pd.concat([final_anime_dataset, generos_dummies], axis=1)
    # Eliminar la columna original 'GenerosAnime' y la columna de lista temporal
    final_anime_dataset = final_anime_dataset.drop(columns=['generos_anime'])
    return final_anime_dataset



def Entrenar_modelo_regresion( 
    final_anime_dataset: pd.DataFrame, 
    parametros_regresion: dict
) -> tuple: 
    """
    Entrena varios modelos de regresión, aplica escalado y retorna los modelos 
    entrenados junto con los datos de prueba (X_test y y_test) para evaluación posterior.
    """

    features_base = parametros_regresion["feature"]

    generos_cols = [col for col in final_anime_dataset.columns 
                    if col not in features_base + [parametros_regresion["target"]] 
                    and pd.api.types.is_numeric_dtype(final_anime_dataset[col])]
    
    X = final_anime_dataset[features_base + generos_cols]
    y = final_anime_dataset[parametros_regresion["target"]]

    # 1. DIVISIÓN DE DATOS
    # X_test e y_test son la versión original (no escalada) aquí.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=parametros_regresion["test_size"],
        random_state=parametros_regresion["random_state"]
    )

    ## 2. ESCALADO DE DATOS Y RECONSTRUCCIÓN
    scaler = StandardScaler()
    
    # El resultado es una matriz NumPy
    X_train_scaled_array = scaler.fit_transform(X_train)
    X_test_scaled_array = scaler.transform(X_test)
    
    # Sobrescribimos X_train y X_test con la versión escalada (manteniendo los nombres)
    X_train = pd.DataFrame(X_train_scaled_array, columns=X_train.columns, index=X_train.index)
    # X_test AHORA contiene los datos ESCALADOS
    X_test = pd.DataFrame(X_test_scaled_array, columns=X_test.columns, index=X_test.index) 

    # 3. DEFINICIÓN DE MODELOS (Iteraciones aumentadas para Lasso/Ridge)
    modelos = {
        "LinearRegression": (LinearRegression(), {}),
        "Ridge": (Ridge(max_iter=10000, random_state=parametros_regresion["random_state"]), {"alpha": uniform(0.01, 10)}),
        "Lasso": (Lasso(max_iter=10000, random_state=parametros_regresion["random_state"]), {"alpha": uniform(0.01, 1)}),
        "RandomForest": (RandomForestRegressor(random_state=parametros_regresion["random_state"]), {
            "n_estimators": randint(10, 50),
            "max_depth": randint(2, 6)
        }),
        "GradientBoosting": (GradientBoostingRegressor(random_state=parametros_regresion["random_state"]), {
            "n_estimators": randint(10, 50),
            "learning_rate": uniform(0.01, 0.3),
            "max_depth": randint(2, 6)
        })
    }

    modelos_entrenados = {}
    
    # 4. ENTRENAMIENTO
    with warnings.catch_warnings():
        # Ignorar advertencias de Scikit-learn (las manejamos con max_iter)
        warnings.filterwarnings("ignore", category=UserWarning) 

        for nombre, (modelo, distribucion) in modelos.items():
            
            # Usamos X_train (ya escalado)
            if distribucion:
                print(f" Buscando mejores hiperparámetros para {nombre}...")
                search = RandomizedSearchCV(
                    modelo,
                    distribucion,
                    n_iter=parametros_regresion.get("n_iter", 5),
                    cv=parametros_regresion.get("cv", 5),
                    random_state=parametros_regresion["random_state"],
                    n_jobs=-1 
                )
                search.fit(X_train, y_train)
                mejor_modelo = search.best_estimator_
                
            else:
                mejor_modelo = modelo.fit(X_train, y_train)
            
            
            modelos_entrenados[nombre] = mejor_modelo
            print(f"✅ {nombre} entrenado.")
    y_test = y_test.to_frame()
    # 5. RETORNO DE RESULTADOS
    # Se retorna X_test que contiene los datos ESCALADOS listos para la evaluación.
    return modelos_entrenados, X_test, y_test