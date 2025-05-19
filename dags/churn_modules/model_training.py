"""
Funciones para el entrenamiento del modelo de predicción de churn.
Validación cruzada, optimización de hiperparámetros y registro con MLflow.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import mlflow
import mlflow.sklearn

from .config import (
    logger, BASE_DIR, MLFLOW_TRACKING_URI, PROCESSED_DATA_PATH, 
    MODEL_PATH, FEATURE_IMPORTANCE_PATH, XGBOOST_PARAMS, MESSAGES,
    VISUALIZATION_CONFIG
)
from .db_operations import save_to_postgres

def train_model(**kwargs):
    """
    Entrena un modelo XGBoost para la predicción de churn:
    - Carga los datos preprocesados
    - Entrena el modelo con validación cruzada
    - Guarda el modelo entrenado en formato pickle
    """
    logger.info("Iniciando entrenamiento del modelo...")
    
    # Iniciar el tracking de MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("churn_prediction")
    
    # Iniciar una nueva ejecución
    with mlflow.start_run(run_name="model_training") as run:
        # 1. Cargar datos procesados
        logger.info(f"Cargando datos desde {PROCESSED_DATA_PATH}")
        if not os.path.exists(PROCESSED_DATA_PATH):
            error_msg = f"No se encontraron los datos procesados en {PROCESSED_DATA_PATH}. Asegúrese de que la tarea de preparación de datos se ejecutó correctamente."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        with open(PROCESSED_DATA_PATH, 'rb') as f:
            train_data = pickle.load(f)
        
        X_train = train_data['X_train']
        y_train = train_data['y_train']
        feature_names = train_data['feature_names']
        scaler = train_data['scaler']
        
        logger.info(f"Datos cargados: {X_train.shape[0]} muestras, {X_train.shape[1]} características")
        
        # Registrar en MLflow
        mlflow.log_param("num_features", X_train.shape[1])
        mlflow.log_param("num_samples", X_train.shape[0])
        
        # 2. Configuración de hiperparámetros para XGBoost
        logger.info("Configurando hiperparámetros...")
        
        # Calculamos el peso para balancear las clases
        scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
        
        # Hiperparámetros
        params = XGBOOST_PARAMS.copy()
        params['scale_pos_weight'] = scale_pos_weight
        # params['eval_metric'] = 'auc'
        
        # Registrar hiperparámetros en MLflow
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # 3. Validación cruzada para evaluación previa
        logger.info("Realizando validación cruzada...")
        
        model_cv = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=params['random_state'])
        cv_scores = cross_val_score(model_cv, X_train, y_train, cv=cv, scoring='roc_auc')
        
        logger.info(f"AUC-ROC en validación cruzada: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Registrar en MLflow
        mlflow.log_metric("cv_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_auc_std", cv_scores.std())
        
        # 4. Entrenamiento del modelo final
        logger.info("Entrenando modelo XGBoost final...")
        
        # Definimos el modelo
        model = xgb.XGBClassifier(**params)
        
        # Entrenamos el modelo
        model.fit(
            X_train, 
            y_train, 
            verbose=True
        )
        
        logger.info("Modelo entrenado exitosamente")
        
        # 5. Calcular y guardar importancia de características
        feature_importance = get_feature_importance(model, feature_names)
        
        # 6. Crear visualizaciones
        create_feature_importance_plots(feature_importance)
        
        # 7. Guardar en PostgreSQL
        try_save_to_postgres(feature_importance)
        
        # 8. Guardar el modelo
        logger.info(f"Guardando modelo en {MODEL_PATH}")
        model_package = {
            'model': model,
            'feature_names': feature_names,
            'scaler': scaler,
            'feature_importance': feature_importance,
            'hyperparameters': params,
            'training_date': mlflow.active_run().info.start_time
        }
        
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model_package, f)
        
        logger.info("Modelo guardado exitosamente")
        
        # 9. Registrar modelo en MLflow
        mlflow.xgboost.log_model(model, "xgboost_model")
        mlflow.log_artifact(MODEL_PATH, "pickled_model")
        
        return {
            'status': 'success',
            'message': MESSAGES['training_complete'],
            'model_type': 'XGBoost',
            'cv_auc_mean': float(cv_scores.mean()),
            'top_features': feature_importance['Feature'].head(5).tolist()
        }

def get_feature_importance(model, feature_names):
    """Obtiene y formatea la importancia de características del modelo"""
    logger.info("Calculando importancia de características...")
    
    # Importancia basada en ganancia
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    logger.info("Top 10 características más importantes:")
    for i, (feature, importance) in enumerate(zip(feature_importance['Feature'].head(10), 
                                                feature_importance['Importance'].head(10))):
        logger.info(f"{i+1}. {feature}: {importance:.4f}")
    
    # Guardar importancia de características
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    logger.info(f"Importancia de características guardada en {FEATURE_IMPORTANCE_PATH}")
    
    return feature_importance

def create_feature_importance_plots(feature_importance):
    """Crea y guarda visualizaciones de importancia de características"""
    # Crear figura para la importancia de características
    plt.figure(figsize=VISUALIZATION_CONFIG['figsize_medium'])
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Top 20 Características Más Importantes')
    plt.tight_layout()
    
    # Guardar figura y registrar en MLflow
    importance_fig_path = f"{BASE_DIR}/feature_importance_plot.png"
    plt.savefig(importance_fig_path)
    plt.close()
    
    mlflow.log_artifact(importance_fig_path, "plots")
    mlflow.log_dict(
        {feature: float(importance) for feature, importance in 
        zip(feature_importance['Feature'], feature_importance['Importance'])},
        "feature_importance_full.json"
    )

def try_save_to_postgres(feature_importance):
    """Intenta guardar datos en PostgreSQL"""
    try:
        # Preparar datos para PostgreSQL
        feature_importance_pg = feature_importance.copy()
        feature_importance_pg['rank'] = feature_importance_pg.reset_index().index + 1
        feature_importance_pg['model_id'] = 1  # ID temporal, se actualizará después
        feature_importance_pg.rename(columns={'Importance': 'importance_value'}, inplace=True)
        
        # Guardar en PostgreSQL
        save_to_postgres(
            feature_importance_pg[['model_id', 'Feature', 'importance_value', 'rank']], 
            'churn_feature_importance_temp', 
            if_exists='replace'
        )
        logger.info("Importancia de características guardada en PostgreSQL")
        return True
    except Exception as e:
        logger.warning(f"No se pudo guardar en PostgreSQL: {e}")
        return False