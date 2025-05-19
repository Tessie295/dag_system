"""
Funciones para la evaluación del modelo de predicción de churn.
Cálculo de métricas, visualizaciones y explicabilidad SHAP.
"""

import os
import json
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import mlflow

from .config import (
    logger, BASE_DIR, MLFLOW_TRACKING_URI, MODEL_PATH, HOLDOUT_PATH, 
    METRICS_PATH, SHAP_PLOT_PATH, VISUALIZATION_CONFIG, SHAP_AVAILABLE, MESSAGES
)
from .db_operations import (
    save_metrics_to_postgres, save_feature_importance, 
    save_predictions, save_monitoring_event
)
from .utils import log_metrics_for_prometheus

def evaluate_model(**kwargs):
    """
    Evalúa el modelo entrenado:
    - Carga el modelo guardado y los datos de holdout
    - Realiza predicciones y evalúa métricas
    - Genera explicaciones SHAP para interpretabilidad
    - Registra resultados y genera visualizaciones
    """
    logger.info("Iniciando evaluación del modelo...")
    
    # Iniciar el tracking de MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("churn_prediction")
    
    # Iniciar una nueva ejecución
    with mlflow.start_run(run_name="model_evaluation") as run:
        # 1. Cargar el modelo
        logger.info(f"Cargando modelo desde {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            error_msg = f"No se encontró el modelo en {MODEL_PATH}. Asegúrese de que la tarea de entrenamiento se ejecutó correctamente."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        with open(MODEL_PATH, 'rb') as f:
            model_package = pickle.load(f)
        
        model = model_package['model']
        feature_names = model_package['feature_names']
        scaler = model_package['scaler']
        feature_importance = model_package['feature_importance']
        
        # 2. Cargar datos de holdout
        logger.info(f"Cargando datos de holdout desde {HOLDOUT_PATH}")
        if not os.path.exists(HOLDOUT_PATH):
            error_msg = f"No se encontraron los datos de holdout en {HOLDOUT_PATH}."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        with open(HOLDOUT_PATH, 'rb') as f:
            holdout_data = pickle.load(f)
        
        X_holdout = holdout_data['X_holdout']
        y_holdout = holdout_data['y_holdout']
        customer_ids = holdout_data.get('customer_ids', None)
        
        logger.info(f"Datos de holdout cargados: {X_holdout.shape[0]} muestras")
        
        # Registrar en MLflow
        mlflow.log_param("holdout_samples", X_holdout.shape[0])
        mlflow.log_param("holdout_features", X_holdout.shape[1])
        
        # 3. Realizar predicciones
        logger.info("Realizando predicciones...")
        y_pred = model.predict(X_holdout)
        y_pred_proba = model.predict_proba(X_holdout)[:, 1]
        
        # 4. Evaluar el modelo y registrar métricas
        metrics = calculate_metrics(y_holdout, y_pred, y_pred_proba)
        
        # 5. Generar visualizaciones de evaluación
        visualization_paths = generate_evaluation_plots(y_holdout, y_pred, y_pred_proba)
        
        # 6. Guardar métricas en diferentes formatos
        save_metrics(metrics, visualization_paths)
        
        # 7. Guardar en PostgreSQL
        model_id = save_results_to_postgres(metrics, feature_importance, 
                                          customer_ids, y_pred_proba)
        
        # 8. Interpretación con SHAP para explicabilidad
        if SHAP_AVAILABLE:
            shap_results = generate_shap_explanations(model, X_holdout, feature_names, y_pred_proba)
        else:
            logger.warning("SHAP no está disponible. Se omitió la generación de explicaciones SHAP.")
            logger.warning("Instale SHAP para habilitar análisis avanzados: pip install shap")
            shap_results = None
        
        # 9. Generar archivo con predicciones para uso posterior
        save_predictions_file(customer_ids, y_pred_proba, y_pred, y_holdout)
        
        # 10. Resumen final
        logger.info("\n========== EVALUACIÓN COMPLETADA ==========")
        logger.info(f"El modelo tiene un AUC-ROC de {metrics['auc_roc']:.4f} en el conjunto de holdout")
        logger.info("Los resultados detallados se han guardado para su análisis")
        
        # Registrar en MLflow que la ejecución fue exitosa
        mlflow.log_param("evaluation_status", "success")
        
        return {
            'status': 'success',
            'message': MESSAGES['evaluation_complete'],
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'auc_roc': float(metrics['auc_roc'])
        }

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calcula métricas de evaluación del modelo"""
    logger.info("Evaluando métricas del modelo...")
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    
    # Imprimir métricas en los logs
    logger.info("\n========== MÉTRICAS DE EVALUACIÓN DEL MODELO ==========")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"AUC-ROC: {auc_roc:.4f}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    logger.info("\nMatriz de Confusión:")
    logger.info(f"\n{cm}")
    
    # Informe de clasificación detallado
    class_report = classification_report(y_true, y_pred)
    logger.info("\nInforme de Clasificación:")
    logger.info(f"\n{class_report}")
    
    # Registrar métricas en MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("auc_roc", auc_roc)
    
    # Crear diccionario de métricas
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc_roc': float(auc_roc),
        'evaluation_date': datetime.now().isoformat()
    }
    
    return metrics

def generate_evaluation_plots(y_true, y_pred, y_pred_proba):
    """Genera visualizaciones para la evaluación del modelo"""
    logger.info("Generando visualizaciones...")
    
    plot_paths = {}
    
    # Matriz de confusión
    plt.figure(figsize=VISUALIZATION_CONFIG['figsize_medium'])
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusión')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predicho')
    cm_plot_path = f"{BASE_DIR}/confusion_matrix.png"
    plt.savefig(cm_plot_path)
    plt.close()
    plot_paths['confusion_matrix'] = cm_plot_path
    
    # Curva ROC
    plt.figure(figsize=VISUALIZATION_CONFIG['figsize_medium'])
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    roc_plot_path = f"{BASE_DIR}/roc_curve.png"
    plt.savefig(roc_plot_path)
    plt.close()
    plot_paths['roc_curve'] = roc_plot_path
    
    # Curva Precision-Recall
    plt.figure(figsize=VISUALIZATION_CONFIG['figsize_medium'])
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    plt.plot(recall_curve, precision_curve, label=f'AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend(loc='lower left')
    pr_plot_path = f"{BASE_DIR}/precision_recall_curve.png"
    plt.savefig(pr_plot_path)
    plt.close()
    plot_paths['precision_recall_curve'] = pr_plot_path
    
    # Distribución de probabilidades
    plt.figure(figsize=VISUALIZATION_CONFIG['figsize_medium'])
    sns.histplot(y_pred_proba, bins=30, kde=True)
    plt.xlabel('Probabilidad de Churn')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Probabilidades de Churn')
    dist_plot_path = f"{BASE_DIR}/probability_distribution.png"
    plt.savefig(dist_plot_path)
    plt.close()
    plot_paths['probability_distribution'] = dist_plot_path
    
    # Registrar visualizaciones en MLflow
    for plot_name, plot_path in plot_paths.items():
        mlflow.log_artifact(plot_path, "evaluation_plots")
    
    return plot_paths

def save_metrics(metrics, visualization_paths):
    """Guarda métricas en diferentes formatos"""
    # Guardar en JSON
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Métricas guardadas en {METRICS_PATH}")
    
    # Guardar métricas para monitoreo con Prometheus
    log_metrics_for_prometheus(metrics)
    
    # Registrar en MLflow
    mlflow.log_dict(metrics, "metrics.json")
    
    # Metricas agregadas para Prometheus
    churn_rate = metrics.get('recall', 0.0)
    prediction_distribution = {
        'churned': churn_rate,
        'retained': 1.0 - churn_rate
    }
    
    prometheus_metrics = {
        'churn_rate': churn_rate,
        'prediction_distribution_churned': prediction_distribution['churned'],
        'prediction_distribution_retained': prediction_distribution['retained']
    }
    
    log_metrics_for_prometheus(prometheus_metrics, metric_prefix='churn_model_additional')

def save_results_to_postgres(metrics, feature_importance, customer_ids, y_pred_proba):
    """Guarda resultados en PostgreSQL"""
    try:
        # Insertar métricas del modelo
        model_id = save_metrics_to_postgres(metrics, model_type='XGBoost')
        
        # Si tenemos model_id, guardar más información
        if model_id:
            # Guardar importancia de características
            feature_importance_pg = feature_importance.copy()
            feature_importance_pg['feature_name'] = feature_importance_pg['Feature']
            feature_importance_pg['importance_value'] = feature_importance_pg['Importance']
            feature_importance_pg['rank'] = range(1, len(feature_importance_pg) + 1)
            
            save_feature_importance(
                feature_importance_pg[['feature_name', 'importance_value', 'rank']], 
                model_id
            )
            
            # Guardar predicciones de clientes
            if customer_ids is not None:
                # Crear un DataFrame con las predicciones
                predictions_df = pd.DataFrame({
                    'customer_id': customer_ids,
                    'churn_probability': y_pred_proba,
                    'prediction_date': datetime.now()
                })
                
                save_predictions(predictions_df, model_id)
            
            # Guardar evento de monitoreo
            save_monitoring_event(
                model_id=model_id,
                event_type='model_evaluation',
                event_description='Evaluación de modelo en conjunto de holdout',
                metrics=metrics
            )
            
            logger.info(f"Resultados guardados en PostgreSQL (model_id: {model_id})")
            return model_id
    except Exception as e:
        logger.warning(f"No se pudo guardar en PostgreSQL: {e}")
        logger.warning("Continuando con el flujo normal...")
    
    return None

def generate_shap_explanations(model, X, feature_names, y_pred_proba):
    """Genera explicaciones SHAP para el modelo"""
    logger.info("\nGenerando explicaciones SHAP para interpretabilidad...")
    
    try:
        import shap
        
        # Crear explainer SHAP
        explainer = shap.TreeExplainer(model)
        
        # Calcular valores SHAP (usar una muestra para eficiencia)
        sample_size = min(200, X.shape[0])
        X_sample = X.sample(sample_size, random_state=42)
        shap_values = explainer.shap_values(X_sample)
        
        # Generar y guardar el gráfico de resumen SHAP
        plt.figure(figsize=VISUALIZATION_CONFIG['figsize_large'])
        shap.summary_plot(
            shap_values, 
            X_sample, 
            feature_names=feature_names, 
            show=False, 
            plot_size=VISUALIZATION_CONFIG['figsize_large']
        )
        plt.tight_layout()
        plt.savefig(SHAP_PLOT_PATH, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico SHAP guardado en {SHAP_PLOT_PATH}")
        
        # Registrar en MLflow
        mlflow.log_artifact(SHAP_PLOT_PATH, "shap_plots")
        
        # Crear un DataFrame con los valores SHAP promedio para cada característica
        shap_importance = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Value': np.abs(shap_values).mean(0)
        })
        shap_importance = shap_importance.sort_values('SHAP_Value', ascending=False)
        
        # Guardar importancia SHAP
        shap_importance_path = f'{BASE_DIR}/shap_importance.csv'
        shap_importance.to_csv(shap_importance_path, index=False)
        mlflow.log_artifact(shap_importance_path, "shap_analysis")
        
        # Registrar en MLflow
        mlflow.log_dict(
            {feature: float(value) for feature, value in 
            zip(shap_importance['Feature'], shap_importance['SHAP_Value'])},
            "shap_importance.json"
        )
        
        # Analizar el efecto de las variables en el modelo
        logger.info("\n========== EFECTO DE LAS VARIABLES EN EL MODELO ==========")
        logger.info("Top 10 características según valores SHAP:")
        
        for i, (feature, importance) in enumerate(zip(shap_importance['Feature'].head(10), 
                                                  shap_importance['SHAP_Value'].head(10))):
            logger.info(f"{i+1}. {feature}: {importance:.4f}")
        
        # Analizar clientes específicos
        specific_clients_analysis = analyze_specific_clients(explainer, X, feature_names, y_pred_proba)
        
        return {
            'shap_importance': shap_importance,
            'specific_clients_analysis': specific_clients_analysis
        }
        
    except Exception as e:
        logger.error(f"Error al generar explicaciones SHAP: {e}")
        logger.error("Continuando con la evaluación sin análisis SHAP...")
        return None

def analyze_specific_clients(explainer, X, feature_names, y_pred_proba):
    """Analiza clientes específicos para entender factores individuales"""
    logger.info("\n========== ANÁLISIS DE CLIENTES ESPECÍFICOS ==========")
    
    # Identificar ejemplos de alta probabilidad de churn y baja probabilidad
    high_churn_idx = y_pred_proba.argmax()
    low_churn_idx = y_pred_proba.argmin()
    
    # Analizar cliente con alto riesgo de churn
    logger.info("\nCliente con ALTO riesgo de abandono:")
    logger.info(f"Probabilidad: {y_pred_proba[high_churn_idx]:.4f}")
    
    client_shap = explainer.shap_values(X.iloc[[high_churn_idx]])[0]
    
    # Ordenar características por impacto
    feature_impact = pd.DataFrame({
        'Feature': feature_names,
        'Impact': client_shap,
        'Value': X.iloc[high_churn_idx].values
    })
    
    # Ordenar por impacto absoluto
    feature_impact['Abs_Impact'] = np.abs(feature_impact['Impact'])
    feature_impact = feature_impact.sort_values('Abs_Impact', ascending=False)
    
    # Mostrar factores que más influyen
    logger.info("Factores que más influyen en la predicción (TOP 5):")
    for i, row in feature_impact.head(5).iterrows():
        direction = "INCREMENTA" if row['Impact'] > 0 else "REDUCE"
        logger.info(f"  - {row['Feature']}: {direction} el riesgo de abandono "
              f"(impacto: {row['Impact']:.4f}, valor: {row['Value']:.2f})")
    
    # Analizar cliente con bajo riesgo de churn
    logger.info("\nCliente con BAJO riesgo de abandono:")
    logger.info(f"Probabilidad: {y_pred_proba[low_churn_idx]:.4f}")
    
    client_shap_low = explainer.shap_values(X.iloc[[low_churn_idx]])[0]
    
    # Ordenar características por impacto
    feature_impact_low = pd.DataFrame({
        'Feature': feature_names,
        'Impact': client_shap_low,
        'Value': X.iloc[low_churn_idx].values
    })
    
    # Ordenar por impacto absoluto
    feature_impact_low['Abs_Impact'] = np.abs(feature_impact_low['Impact'])
    feature_impact_low = feature_impact_low.sort_values('Abs_Impact', ascending=False)
    
    # Mostrar factores que más influyen
    logger.info("Factores que más influyen en la predicción (TOP 5):")
    for i, row in feature_impact_low.head(5).iterrows():
        direction = "INCREMENTA" if row['Impact'] > 0 else "REDUCE"
        logger.info(f"  - {row['Feature']}: {direction} el riesgo de abandono "
              f"(impacto: {row['Impact']:.4f}, valor: {row['Value']:.2f})")
    
    # Guardar análisis de clientes específicos
    high_risk_path = f'{BASE_DIR}/high_churn_client_analysis.csv'
    low_risk_path = f'{BASE_DIR}/low_churn_client_analysis.csv'
    
    feature_impact.to_csv(high_risk_path, index=False)
    feature_impact_low.to_csv(low_risk_path, index=False)
    
    # Registrar en MLflow
    mlflow.log_artifact(high_risk_path, "client_analysis")
    mlflow.log_artifact(low_risk_path, "client_analysis")
    
    # Crear visualización para cliente de alto riesgo
    plt.figure(figsize=VISUALIZATION_CONFIG['figsize_medium'])
    # Mostrar solo las 10 características principales por claridad
    topn = min(10, len(feature_impact))
    impact_data = feature_impact.head(topn).copy()
    impact_data['Color'] = impact_data['Impact'].apply(lambda x: 'red' if x > 0 else 'blue')
    sns.barplot(x='Impact', y='Feature', data=impact_data, palette=impact_data['Color'])
    plt.title(f'Factores que Influyen en la Predicción - Cliente con Alto Riesgo')
    plt.tight_layout()
    high_risk_plot_path = f"{BASE_DIR}/high_risk_client_factors.png"
    plt.savefig(high_risk_plot_path)
    plt.close()
    
    # Registrar en MLflow
    mlflow.log_artifact(high_risk_plot_path, "client_analysis")
    
    return {
        'high_risk': feature_impact,
        'low_risk': feature_impact_low
    }

def save_predictions_file(customer_ids, y_pred_proba, y_pred, y_true):
    """Guarda archivo con predicciones para uso posterior"""
    if customer_ids is not None:
        predictions_output = pd.DataFrame({
            'customer_id': customer_ids,
            'churn_probability': y_pred_proba,
            'predicted_churn': y_pred,
            'actual_churn': y_true.values
        })
        predictions_path = f"{BASE_DIR}/churn_predictions.csv"
        predictions_output.to_csv(predictions_path, index=False)
        logger.info(f"Predicciones guardadas en {predictions_path}")
        
        # Registrar en MLflow
        mlflow.log_artifact(predictions_path, "predictions")