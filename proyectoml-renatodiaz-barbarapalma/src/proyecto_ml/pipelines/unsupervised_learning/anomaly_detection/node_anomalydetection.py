from sklearn.ensemble import IsolationForest
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def detectar_anomalias(df_model: pd.DataFrame, X_scaled, parametros_clustering: dict) -> tuple:
    """
    Retorna: (df_model, fig_matplotlib, anomalias).
    La figura es un matplotlib.figure.Figure compatible con MatplotlibWriter (guarda PNG con savefig).
    """
    iso_forest = IsolationForest(
        contamination=parametros_clustering.get("contamination", "auto"),
        random_state=parametros_clustering.get("random_state", 42)
    )
    df_model['anomaly_iso'] = iso_forest.fit_predict(X_scaled)
    df_model['tipo_dato'] = df_model['anomaly_iso'].apply(lambda x: 'Anomalía' if x == -1 else 'Normal')

    cant_anomalias = (df_model['anomaly_iso'] == -1).sum()
    print(f"⚠️ Se detectaron {cant_anomalias} animes anómalos ({cant_anomalias/len(df_model):.2%}).")

    # --- Crear figura MATPLOTLIB (compatible con MatplotlibWriter) ---
    fig, ax = plt.subplots(figsize=(10, 7))
    groups = df_model.groupby('tipo_dato')
    colors = {'Normal': '#1f77b4', 'Anomalía': '#d62728'}
    for name, group in groups:
        ax.scatter(group['umap_1'], group['umap_2'],
                   s=20, alpha=0.7, label=name, c=colors.get(name, '#7f7f7f'))
    ax.set_title('Detección de Anomalías (Isolation Forest) sobre UMAP')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(title='Tipo')
    ax.grid(alpha=0.25)
    plt.tight_layout()

    anomalias = df_model[df_model['anomaly_iso'] == -1].copy()

    # Guardar anomalías como JSON si se proporciona ruta
    anom_path = parametros_clustering.get('save_anom_json')
    if anom_path:
        try:
            anomalias.to_json(anom_path, orient='records', force_ascii=False)
            print(f"✅ Anomalías guardadas en: {anom_path}")
        except Exception as e:
            print(f"⚠️ No se pudo guardar las anomalías: {e}")

    return df_model, fig, anomalias