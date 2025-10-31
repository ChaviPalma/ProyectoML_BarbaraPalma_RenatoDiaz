# Asegúrate de tener estos imports al principio del archivo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.figure import Figure # Para el type hint

def crear_grafico_qq_residuos(y_test: pd.DataFrame, y_pred_slr: np.ndarray) -> Figure:
    """Crea un gráfico Q-Q de los residuos del modelo."""

    print("Creando gráfico Q-Q de residuos...")

    # Asegúrate que y_test sea una Serie 1D
    if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1:
        y_test_series = y_test.iloc[:, 0]
    else:
        y_test_series = y_test # Asume que ya es Serie

    # Calcula los residuos
    # y_pred_slr probablemente viene como array numpy, y_test_series es Serie pandas
    # Asegúrate que los índices coincidan si haces resta directa, o usa .values
    try:
        residuals_slr = y_test_series.values - y_pred_slr.flatten() # Usar .values y .flatten() para asegurar compatibilidad
    except AttributeError: # Si y_pred_slr ya es 1D
         residuals_slr = y_test_series.values - y_pred_slr


    # Crea la figura y el eje con Matplotlib
    fig, ax = plt.subplots(figsize=(7, 5))

    # Genera el gráfico Q-Q usando scipy.stats y ploteando en el eje 'ax'
    stats.probplot(residuals_slr, dist="norm", plot=ax)

    # Configura títulos y etiquetas en el eje 'ax'
    ax.set_title("Gráfico Q-Q de Residuos (Regresión Lineal Simple)")
    ax.set_xlabel("Cuantiles Teóricos")
    ax.set_ylabel("Cuantiles de los Residuos")
    ax.grid(True)

    # Importante: No uses plt.show() dentro de un nodo de Kedro
    # plt.show()

    print("Gráfico Q-Q creado.")
    # Devuelve el objeto Figure de Matplotlib
    return fig