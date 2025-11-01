import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


def calcular_metricas_modelos(
    modelos_entrenados: dict, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> dict:
    
    print("Iniciando cálculo de métricas en el pipeline de reporting...")
    metricas_modelos = {}
    
    for nombre, modelo in modelos_entrenados.items():
        # Generamos las predicciones aquí
        y_pred = modelo.predict(X_test)
        
        # Calculamos las métricas aquí
        # Nota: y_test y y_pred se asume que están alineados por índice o son arrays compatibles.
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Guardamos las métricas en el diccionario
        metricas_modelos[nombre] = {
            "R2": r2, 
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse
        }
        
        print(f" Métricas calculadas para {nombre} - R2: {r2:.3f}, MSE: {mse:.3f}")
        
    # Devolvemos el diccionario de métricas
    return metricas_modelos

def plot_metricas_comparativas(metricas_modelos: dict) -> plt.Figure:
    """
    Toma el diccionario de métricas (creado por la función anterior)
    y crea un gráfico de barras comparativo.
    """
    # Convertir el diccionario a un DataFrame
    df_metricas = pd.DataFrame.from_dict(metricas_modelos, orient='index')
    
    # Crear la figura con subplots (R2 y MSE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Comparación de Métricas de Modelos de Regresión', fontsize=16)

    # Gráfico de Barras para R2
    df_r2_sorted = df_metricas.sort_values('R2', ascending=False)
    sns.barplot(x=df_r2_sorted.index, y='R2', data=df_r2_sorted, ax=ax1, palette='coolwarm')
    ax1.set_title('R-cuadrado (R2) - (Más alto es mejor)')
    ax1.set_ylabel('Puntuación R2')
    ax1.set_ylim(min(0, df_metricas['R2'].min() - 0.1), 1.0) 
    ax1.tick_params(axis='x', rotation=45)

    # Gráfico de Barras para MSE
    df_mse_sorted = df_metricas.sort_values('MSE', ascending=True)
    sns.barplot(x=df_mse_sorted.index, y='MSE', data=df_mse_sorted, ax=ax2, palette='viridis')
    ax2.set_title('Error Cuadrático Medio (MSE) - (Más bajo es mejor)')
    ax2.set_ylabel('Valor de MSE')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

# --- GRÁFICO 2: Función Genérica "Reales vs. Predichos" ---

def plot_actual_vs_predicted(
    modelo: any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    nombre_modelo: str
) -> plt.Figure:
    """
    Genera un gráfico de dispersión (scatter plot) de
    valores reales vs. valores predichos para UN modelo específico.
    """
    y_pred = modelo.predict(X_test)
    fig = plt.figure(figsize=(10, 8))
    
    plt.scatter(y_test, y_pred, alpha=0.3, s=10, label="Predicciones")
    
    # CORRECCIÓN CLAVE: Se usa .values.min() y np.min() para obtener escalares puros,
    # eliminando el error de "Series is ambiguous".
    y_test_min = y_test.values.min()
    y_pred_min = np.min(y_pred)
    
    y_test_max = y_test.values.max()
    y_pred_max = np.max(y_pred)

    lims = [
        min(y_test_min, y_pred_min),
        max(y_test_max, y_pred_max)
    ]
    
    plt.plot(lims, lims, 'r--', lw=2, label='Predicción Perfecta (y=x)')
    
    plt.xlabel("Valores Reales (y_test)")
    plt.ylabel("Valores Predichos (y_pred)")
    plt.title(f"Reales vs. Predichos - Modelo: {nombre_modelo}")
    plt.legend()
    plt.grid(True)
    
    return fig

# --- GRÁFICO 3: Función Genérica "Gráfico de Residuos" ---

def plot_residuos(modelo, X_test, y_test, nombre_modelo: str) -> plt.Figure:
    """
    Genera un gráfico de residuos (error) vs. valores predichos para UN modelo específico.
    """
    y_pred = modelo.predict(X_test)
    
    # Convertir a vectores 1D
    y_test = np.ravel(y_test)
    y_pred = np.ravel(y_pred)
    
    residuos = y_test - y_pred

    fig = plt.figure(figsize=(12, 7))
    sns.residplot(
        x=y_pred,
        y=residuos,
        lowess=True,
        scatter_kws={'alpha': 0.3, 's': 10},
        line_kws={'color': 'red', 'lw': 2, 'label': 'Tendencia de Error'}
    )
    
    plt.axhline(0, color='black', linestyle='--', lw=1)
    plt.title(f"Gráfico de Residuos - Modelo: {nombre_modelo}")
    plt.legend()
    
    return fig


# --- Función 4: "Ayudante" para extraer modelos del diccionario ---

def get_model_from_dict(model_dict: dict, model_name: str) -> any:
    """
    Extrae un modelo específico del diccionario de modelos entrenados.
    """
    # Si el input viene del formato params:, model_name ya está resuelto a su valor literal.
    if model_name not in model_dict:
        raise KeyError(f"Modelo '{model_name}' no encontrado en el diccionario. "
                       f"Modelos disponibles: {list(model_dict.keys())}")
    
    print(f"Reporting: Extrayendo modelo '{model_name}' del diccionario.")
    return model_dict[model_name]
