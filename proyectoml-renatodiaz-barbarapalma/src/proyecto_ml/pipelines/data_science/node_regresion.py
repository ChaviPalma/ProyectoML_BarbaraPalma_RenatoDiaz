import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import uniform, randint

# ... (tu función preprocesar_anime_dataset sigue igual) ...

def Entrenar_modelo_regresion( 
    final_anime_dataset: pd.DataFrame, 
    parametros_regresion: dict
) -> tuple: 
 

    features_base = parametros_regresion["feature"]

    generos_cols = [col for col in final_anime_dataset.columns 
                    if col not in features_base + [parametros_regresion["target"]] 
                    and pd.api.types.is_numeric_dtype(final_anime_dataset[col])]
    
    X = final_anime_dataset[features_base + generos_cols]
    y = final_anime_dataset[parametros_regresion["target"]]

    # Estos X_test, y_test ahora debemos retornarlos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=parametros_regresion["test_size"],
        random_state=parametros_regresion["random_state"]
    )

    modelos = {
        "LinearRegression": (LinearRegression(), {}),
        "Ridge": (Ridge(), {"alpha": uniform(0.01, 10)}),
        "Lasso": (Lasso(), {"alpha": uniform(0.01, 1)}),
        "RandomForest": (RandomForestRegressor(), {
            "n_estimators": randint(10, 50),
            "max_depth": randint(2, 6)
        }),
        "GradientBoosting": (GradientBoostingRegressor(), {
            "n_estimators": randint(10, 50),
            "learning_rate": uniform(0.01, 0.3),
            "max_depth": randint(2, 6)
        })
    }

    modelos_entrenados = {}
   

    for nombre, (modelo, distribucion) in modelos.items():
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


    return modelos_entrenados, X_test, y_test