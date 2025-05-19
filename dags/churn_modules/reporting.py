"""
Funciones para la generación de reportes y envío de notificaciones.

"""

import os
from datetime import datetime
import pandas as pd
import json

from .config import (
    logger, BASE_DIR, MODEL_PATH, METRICS_PATH, 
    MESSAGES, VISUALIZATION_CONFIG
)

def send_summary_report(**kwargs):
    """
    Genera un resumen del proceso y lo envía para notificación.
    En una implementación real, podría enviar correos electrónicos o notificaciones a Slack.
    """
    # Obtener resultados de las tareas anteriores
    ti = kwargs['ti']
    eval_results = ti.xcom_pull(task_ids='evaluate_model')
    
    if not eval_results:
        msg = "No se pudieron obtener resultados de evaluación"
        logger.warning(msg)
        return {"status": "warning", "message": msg}
    
    # Formatear resultados para notificación
    metrics = {
        'accuracy': eval_results.get('accuracy', 'N/A'),
        'precision': eval_results.get('precision', 'N/A'),
        'recall': eval_results.get('recall', 'N/A'),
        'f1': eval_results.get('f1', 'N/A'),
        'auc_roc': eval_results.get('auc_roc', 'N/A')
    }
    
    # Crear mensaje de resumen
    summary = generate_summary_text(metrics)
    
    # Guardar resumen en archivo para referencia
    summary_path = f"{BASE_DIR}/execution_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    # Intenta enviar notificaciones si están configuradas
    try:
        send_notifications(metrics, summary)
    except Exception as e:
        logger.warning(f"No se pudieron enviar notificaciones: {e}")
    
    return {
        'status': 'success',
        'message': MESSAGES['summary_complete'],
        'summary_path': summary_path
    }

def generate_summary_text(metrics):
    """Genera texto de resumen con las métricas del modelo"""
    summary = f"""
    ===========================================
    RESUMEN DE EJECUCIÓN: PREDICCIÓN DE CHURN
    ===========================================
    
    Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    MÉTRICAS DEL MODELO:
      - Accuracy: {metrics['accuracy']:.4f}
      - Precision: {metrics['precision']:.4f}
      - Recall: {metrics['recall']:.4f}
      - F1 Score: {metrics['f1']:.4f}
      - AUC-ROC: {metrics['auc_roc']:.4f}
    
    UBICACIÓN DE RESULTADOS:
      - Modelo guardado en: {MODEL_PATH}
      - Métricas detalladas en: {METRICS_PATH}
      - Visualizaciones en: {BASE_DIR}
    
    Para ver más detalles, consulte el dashboard de MLflow o
    los logs de ejecución en Airflow.
    """
    
    # Imprimir en logs
    logger.info(summary)
    
    return summary

def send_notifications(metrics, summary):
    """
    Envía notificaciones a diferentes canales.
    Implementación de ejemplo que puede personalizarse según necesidades.
    """
    # Ejemplo: Enviar email (requiere configuración adicional)
    try:
        send_email_notification(metrics, summary)
    except:
        logger.warning("No se pudo enviar email de notificación")
    
    # Ejemplo: Notificación a Slack (requiere configuración adicional)
    try:
        send_slack_notification(metrics, summary)
    except:
        logger.warning("No se pudo enviar notificación a Slack")
    
    # Ejemplo: Crear archivo para integración con otros sistemas
    try:
        create_integration_file(metrics)
    except Exception as e:
        logger.warning(f"No se pudo crear archivo de integración: {e}")

def send_email_notification(metrics, summary):
    """
    Envía notificación por email.
    Esta es una implementación simulada - para uso real, configurar servidor SMTP.
    """
    # En una implementación real, usar:
    # from airflow.utils.email import send_email
    # send_email(
    #     to=['team@example.com'],
    #     subject='Resumen de Ejecución: Modelo de Predicción de Churn',
    #     html_content=f"<pre>{summary}</pre>",
    #     files=[f"{BASE_DIR}/confusion_matrix.png"]
    # )
    
    logger.info("Simulando envío de email de notificación...")
    logger.info("Para implementar realmente, configurar SMTP en Airflow")

def send_slack_notification(metrics, summary):
    """
    Envía notificación a Slack.
    Esta es una implementación simulada - para uso real, configurar webhook.
    """
    # En una implementación real, usar:
    # from airflow.providers.slack.hooks.slack_webhook import SlackWebhookHook
    # webhook = SlackWebhookHook(
    #     http_conn_id='slack_webhook',
    #     message=f"Modelo de churn entrenado y evaluado: AUC-ROC={metrics['auc_roc']:.4f}"
    # )
    # webhook.execute()
    
    logger.info("Simulando envío de notificación a Slack...")
    logger.info("Para implementar realmente, configurar webhook de Slack en Airflow")

def create_integration_file(metrics):
    """Crea archivo JSON para integración con otros sistemas"""
    integration_data = {
        'model_version': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'execution_timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'model_path': MODEL_PATH,
        'status': 'success' if metrics['auc_roc'] > 0.7 else 'warning'
    }
    
    integration_path = f"{BASE_DIR}/integration_data.json"
    with open(integration_path, 'w') as f:
        json.dump(integration_data, f, indent=2)
    
    logger.info(f"Archivo de integración creado en {integration_path}")
    
    return integration_path

def generate_html_report():
    """
    Genera un reporte en formato HTML con visualizaciones incorporadas.
    En una implementación real, este reporte podría enviarse por email o
    publicarse en un servidor web interno.
    """
    # Cargar métricas
    try:
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
    except:
        metrics = {}
    
    # Cargar importancia de características
    try:
        importance_df = pd.read_csv(f"{BASE_DIR}/feature_importance.csv")
        top_features = importance_df.head(10).to_html(index=False)
    except:
        top_features = "No disponible"
    
    # Rutas de imágenes
    confusion_matrix_img = "./confusion_matrix.png"
    roc_curve_img = "./roc_curve.png"
    shap_plot_img = "./shap_summary_plot.png"
    
    # Crear HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporte de Predicción de Churn</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; }}
            .metrics {{ display: flex; flex-wrap: wrap; }}
            .metric-card {{ 
                border: 1px solid #ddd; 
                border-radius: 5px; 
                padding: 15px; 
                margin: 10px; 
                width: 200px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .metric-value {{ 
                font-size: 24px; 
                font-weight: bold; 
                margin: 10px 0; 
                color: #2980b9;
            }}
            .plots {{ display: flex; flex-wrap: wrap; }}
            .plot-container {{ margin: 15px; }}
            img {{ max-width: 100%; border: 1px solid #eee; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Reporte de Predicción de Churn</h1>
        <p>Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Métricas del Modelo</h2>
        <div class="metrics">
            <div class="metric-card">
                <h3>Accuracy</h3>
                <div class="metric-value">{metrics.get('accuracy', 'N/A'):.4f}</div>
            </div>
            <div class="metric-card">
                <h3>Precision</h3>
                <div class="metric-value">{metrics.get('precision', 'N/A'):.4f}</div>
            </div>
            <div class="metric-card">
                <h3>Recall</h3>
                <div class="metric-value">{metrics.get('recall', 'N/A'):.4f}</div>
            </div>
            <div class="metric-card">
                <h3>F1 Score</h3>
                <div class="metric-value">{metrics.get('f1', 'N/A'):.4f}</div>
            </div>
            <div class="metric-card">
                <h3>AUC-ROC</h3>
                <div class="metric-value">{metrics.get('auc_roc', 'N/A'):.4f}</div>
            </div>
        </div>
        
        <h2>Visualizaciones</h2>
        <div class="plots">
            <div class="plot-container">
                <h3>Matriz de Confusión</h3>
                <img src="{confusion_matrix_img}" alt="Matriz de Confusión">
            </div>
            <div class="plot-container">
                <h3>Curva ROC</h3>
                <img src="{roc_curve_img}" alt="Curva ROC">
            </div>
            <div class="plot-container">
                <h3>Análisis SHAP</h3>
                <img src="{shap_plot_img}" alt="SHAP Summary Plot">
            </div>
        </div>
        
        <h2>Top 10 Características Más Importantes</h2>
        {top_features}
        
        <h2>Interpretación del Modelo</h2>
        <p>
            El modelo ha identificado los factores más importantes que influyen en el abandono de clientes.
            Las características con mayor importancia deben ser el foco de las estrategias de retención.
            Se recomienda revisar el análisis SHAP para entender cómo cada variable afecta individualmente
            a la probabilidad de abandono de cada cliente.
        </p>
        
        <h2>Recomendaciones</h2>
        <ul>
            <li>Implementar campañas de retención dirigidas a los segmentos con mayor riesgo</li>
            <li>Abordar los problemas relacionados con las características más importantes</li>
            <li>Monitorizar continuamente la efectividad de las intervenciones</li>
            <li>Reevaluar el modelo periódicamente con datos actualizados</li>
        </ul>
        
        <hr>
        <footer>
            <p>Generado automáticamente por el pipeline de predicción de churn. Para más información, contacte al equipo de Ciencia de Datos.</p>
        </footer>
    </body>
    </html>
    """
    
    # Guardar HTML
    html_path = f"{BASE_DIR}/churn_report.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Reporte HTML generado en {html_path}")
    
    return html_path