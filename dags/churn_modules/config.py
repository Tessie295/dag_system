"""
Configuración centralizada para el proyecto de predicción de churn.
Contiene variables, constantes y configuraciones utilizadas por todos los módulos.
"""

import os
import logging

# Configuración de registro para depuración
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración de rutas y conexiones
BASE_DIR = '/opt/airflow/data' #'dag_telecom/data' 
MLFLOW_TRACKING_URI = "http://mlflow:5000"  # Servicio MLflow en Docker
MLFLOW_ARTIFACT_ROOT = "/mlflow/artifacts"  # Directorio accesible para Airflow
POSTGRES_CONN_ID = "postgres_default"  # Conexión definida en Airflow
DATA_PATH = f'{BASE_DIR}/dataset.csv'
PROCESSED_DATA_PATH = f'{BASE_DIR}/processed_data.pkl'
MODEL_PATH = f'{BASE_DIR}/churn_model.pkl'
HOLDOUT_PATH = f'{BASE_DIR}/holdout_data.pkl'
METRICS_PATH = f'{BASE_DIR}/model_metrics.json'
FEATURE_IMPORTANCE_PATH = f'{BASE_DIR}/feature_importance.csv'
SHAP_PLOT_PATH = f'{BASE_DIR}/shap_summary_plot.png'

# Crear directorio base si no existe
os.makedirs(BASE_DIR, exist_ok=True)
# Crear el directorio de MLflow si no existe
os.makedirs(MLFLOW_ARTIFACT_ROOT, exist_ok=True)

# Intentar importar shap con manejo de errores
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP no está disponible. Se omitirá la generación de explicaciones SHAP.")

# Parámetros del modelo XGBoost
XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'learning_rate': 0.05,
    'max_depth': 5,
    'min_child_weight': 2,
    'gamma': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 200,
    'random_state': 42
}

# Configuración general para análisis de datos
DATA_PROCESSING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'top_features_limit': 30,
    'outlier_percentile_cutoff': 0.01
}

# Configuración para visualizaciones
VISUALIZATION_CONFIG = {
    'figsize_large': (12, 10),
    'figsize_medium': (10, 8),
    'figsize_small': (8, 6),
    'dpi': 300
}

# Texto para mensajes estándar
MESSAGES = {
    'setup_complete': "Configuración de PostgreSQL completada",
    'data_prep_complete': "Preparación de datos completada exitosamente",
    'training_complete': "Entrenamiento de modelo completado exitosamente",
    'evaluation_complete': "Evaluación del modelo completada exitosamente",
    'summary_complete': "Resumen generado correctamente"
}