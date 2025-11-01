
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import uniform, randint
from sklearn.preprocessing import StandardScaler



def preprocesar_anime_dataset(final_anime_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesa el DataFrame 'final_anime_dataset' para regresión:
      - Convierte variables categóricas en variables dummy.
    """

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

def Entrenar_modelo_regresion( final_anime_dataset: pd.DataFrame, parametros_regresion: dict) -> tuple[dict, dict]:
    """
    Entrena y evalúa múltiples modelos de regresión después de estandarizar
    las características para asegurar la convergencia de modelos lineales.
    """

    # 1. SELECCIÓN DE CARACTERÍSTICAS
    features_base = parametros_regresion["feature"]

    generos_cols = [col for col in final_anime_dataset.columns 
                  if col not in features_base + [parametros_regresion["target"]] 
                  and pd.api.types.is_numeric_dtype(final_anime_dataset[col])]
    
    X = final_anime_dataset[features_base + generos_cols]
    y = final_anime_dataset[parametros_regresion["target"]]

    # 2. DIVISIÓN DE DATOS
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=parametros_regresion["test_size"],
        random_state=parametros_regresion["random_state"]
    )

    ## PASO CRÍTICO: ESCALADO DE DATOS
    # Los modelos lineales requieren que los datos estén en la misma escala.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Restablecer nombres de columnas después del escalado para trazabilidad (opcional pero útil)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # 3. DEFINICIÓN DE MODELOS Y RANGOS DE HIPERPARÁMETROS
    modelos = {
        # Ridge y Lasso ahora tienen más iteraciones para asegurar la convergencia
        "LinearRegression": (LinearRegression(), {}),
        "Ridge": (Ridge(random_state=parametros_regresion["random_state"]), 
                  {"alpha": uniform(0.01, 1), "max_iter": [10000]}),
        "Lasso": (Lasso(random_state=parametros_regresion["random_state"]), 
                  {"alpha": uniform(0.01, 1), "max_iter": [10000]}),
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
    metricas_modelos = {}

    # 4. ENTRENAMIENTO Y EVALUACIÓN
    for nombre, (modelo, distribucion) in modelos.items():
        # Usar datos escalados solo para Ridge y Lasso, pero usaré los escalados para todos por consistencia de código
        if nombre in ["LinearRegression", "Ridge", "Lasso"]:
            X_train_data = X_train_scaled
            X_test_data = X_test_scaled
        else:
            # Los modelos basados en árboles (RandomForest, GradientBoosting) pueden usar los datos sin escalar
            X_train_data = X_train
            X_test_data = X_test

        if distribucion:
            print(f" Buscando mejores hiperparámetros para {nombre}...")
            search = RandomizedSearchCV(
                modelo,
                distribucion,
                n_iter=parametros_regresion.get("n_iter", 5),
                cv=parametros_regresion.get("cv", 5),
                random_state=parametros_regresion["random_state"],
                n_jobs=1
            )
            search.fit(X_train_data, y_train)
            mejor_modelo = search.best_estimator_
            
        else:
            mejor_modelo = modelo.fit(X_train_data, y_train)
            
        y_pred = mejor_modelo.predict(X_test_data)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred) 

        modelos_entrenados[nombre] = mejor_modelo
        metricas_modelos[nombre] = {"MSE": mse, "R2": r2}

        print(f" {nombre} entrenado - R2: {r2:.3f}, MSE: {mse:.3f}")

    # Uso 'GradientBoosting' como último modelo entrenado para la impresión final
    print(f"✅ Pipeline de entrenamiento finalizado. Revise las métricas para cada modelo.")

    return modelos_entrenados, metricas_modelos