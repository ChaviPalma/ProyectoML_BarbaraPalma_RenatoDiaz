import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)
import numpy as np

# --- NODO 1: Cálculo de Métricas ---

def calcular_metricas_clasificacion(
    modelos_entrenados: dict, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> tuple[dict, dict]:
    """
    Toma los modelos, los evalúa contra los datos de test
    y devuelve dos diccionarios: uno con métricas y otro con reportes.
    """
    print("Iniciando cálculo de métricas en el pipeline de reporting (clasificación)...")
    metricas_modelos = {}
    reportes_clasificacion = {}
    
    for nombre, modelo in modelos_entrenados.items():
        y_pred = modelo.predict(X_test)
        
        # Métricas principales
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metricas_modelos[nombre] = {
            "Accuracy": accuracy,
            "F1-Score (Weighted)": f1
        }
        
        # Reporte de clasificación 
        report = classification_report(
            y_test, y_pred, 
            target_names=['No Interesado (0)', 'Interesado (1)'],
            output_dict=True, 
            zero_division=0
        )
        reportes_clasificacion[nombre] = report
        
        print(f" Métricas calculadas para {nombre} - Accuracy: {accuracy:.3f}")
        
    return metricas_modelos, reportes_clasificacion

# --- NODO 2: Gráfico de Comparación de Métricas ---

def plot_metricas_comparativas_clasificacion(metricas_modelos: dict) -> plt.Figure:
    """
    Toma el diccionario de métricas y crea un gráfico de barras
    comparando el Accuracy y el F1-Score de todos los modelos.
    """
    df_metricas = pd.DataFrame.from_dict(metricas_modelos, orient='index')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Comparación de Métricas de Modelos de Clasificación', fontsize=16)

    # Gráfico de Barras para Accuracy
    df_acc_sorted = df_metricas.sort_values('Accuracy', ascending=False)
    sns.barplot(x=df_acc_sorted.index, y='Accuracy', data=df_acc_sorted, ax=ax1, palette='coolwarm')
    ax1.set_title('Accuracy - (Más alto es mejor)')
    ax1.set_ylabel('Puntuación Accuracy')
    ax1.set_ylim(min(0, df_metricas['Accuracy'].min() - 0.1), 1.0)
    ax1.tick_params(axis='x', rotation=45)

    # Gráfico de Barras para F1-Score (Weighted)
    df_f1_sorted = df_metricas.sort_values('F1-Score (Weighted)', ascending=False)
    sns.barplot(x=df_f1_sorted.index, y='F1-Score (Weighted)', data=df_f1_sorted, ax=ax2, palette='viridis')
    ax2.set_title('F1-Score (Weighted) - (Más alto es mejor)')
    ax2.set_ylabel('Puntuación F1')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

# --- NODO 3: Gráfico de Matriz de Confusión  ---

def plot_confusion_matrix(
    modelo: any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    nombre_modelo: str
) -> plt.Figure:
    """
    Genera y grafica una matriz de confusión para un modelo específico.
    """
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Pred No Interesado', 'Pred Interesado'],
        yticklabels=['Real No Interesado', 'Real Interesado']
    )
    ax.set_ylabel('Etiqueta Real')
    ax.set_xlabel('Etiqueta Predicha')
    ax.set_title(f"Matriz de Confusión - Modelo: {nombre_modelo}")
    return fig

# --- NODO 4: "Ayudante" para extraer modelos  ---

def get_model_from_dict(model_dict: dict, model_name: str) -> any:
    """
    Extrae un modelo específico del diccionario de modelos entrenados.
    """
    if model_name not in model_dict:
        raise KeyError(f"Modelo '{model_name}' no encontrado. Disponibles: {list(model_dict.keys())}")
    
    print(f"Reporting: Extrayendo modelo '{model_name}' del diccionario.")
    return model_dict[model_name]