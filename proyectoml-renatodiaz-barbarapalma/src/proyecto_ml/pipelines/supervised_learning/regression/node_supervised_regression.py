from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

def modelo_regresion_supervisada(df_model: pd.DataFrame, parametros_regresion:dict ):
    print("🔗 INICIANDO INTEGRACIÓN SUPERVISADA...")

    # 1. Preparamos los datos
    target_col = parametros_regresion["target"]

    # Variables Predictoras (Features)
    # Usamos las originales + LOS CLUSTERS NUEVOS
    feature_cols_base = ['total_episodios', 'miembros', 'favoritos', 'popularidad'] # Corregido
    feature_cols_con_cluster = feature_cols_base + ['cluster_dbscan']

    # Eliminamos filas sin target
    df_regresion = df_model.dropna(subset=[target_col] + feature_cols_con_cluster).copy()

    # 2. Split Train/Test
    X = df_regresion[feature_cols_con_cluster]
    y = df_regresion[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parametros_regresion["test_size"], random_state=parametros_regresion["random_state"])

    # 3. Entrenar Modelo SIN Cluster (Línea Base)
    rf_base = RandomForestRegressor(n_estimators=50, random_state=parametros_regresion["random_state"])
    rf_base.fit(X_train[feature_cols_base], y_train)
    preds_base = rf_base.predict(X_test[feature_cols_base])
    mae_base = mean_absolute_error(y_test, preds_base)

    # 4. Entrenar Modelo CON Cluster (Integrado)
    rf_cluster = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_cluster.fit(X_train, y_train)
    preds_cluster = rf_cluster.predict(X_test)
    mae_cluster = mean_absolute_error(y_test, preds_cluster)

    # 5. Comparación
    print(f"📊 Error (MAE) SIN Cluster: {mae_base:.4f}")
    print(f"📊 Error (MAE) CON Cluster: {mae_cluster:.4f}")

    mejora = (mae_base - mae_cluster) / mae_base * 100
    if mejora > 0:
        print(f"✅ ¡ÉXITO! Agregar los clusters redujo el error en un {mejora:.2f}%")
    else:
        print(f"⚠️ El cluster no mejoró el modelo directamente.")

    metricas_regresion_supervisada = {
        "mae_sin_cluster": mae_base,
        "mae_con_cluster": mae_cluster,
        "mejora_porcentaje": mejora
    }

    return metricas_regresion_supervisada