"""
Funciones de utilidad para el proyecto de predicción de churn.

"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow

from .config import logger, BASE_DIR, MLFLOW_TRACKING_URI, MLFLOW_ARTIFACT_ROOT

def format_numeric_columns(df):
    """
    Formatea columnas numéricas que usan coma como separador decimal.
    Maneja el caso común en datos de origen español/europeo.
    """
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = df[col].str.replace(',', '.').astype(float)
        except:
            pass
    return df

def log_metrics_for_prometheus(metrics_dict, metric_prefix='churn_model'):
    """
    Registra métricas en un formato compatible con Prometheus.
    Estas métricas pueden ser recogidas por un exportador de Prometheus.
    """
    metrics_file = f"{BASE_DIR}/prometheus_metrics.txt"
    
    with open(metrics_file, 'w') as f:
        timestamp = datetime.now().timestamp() * 1000  # timestamp en milisegundos
        for metric_name, value in metrics_dict.items():
            if isinstance(value, (int, float)):  # Solo métricas numéricas
                f.write(f"{metric_prefix}_{metric_name} {value} {timestamp}\n")
    
    logger.info(f"Métricas para Prometheus guardadas en {metrics_file}")
    return metrics_file

def create_timestamp_id():
    """Crea un identificador basado en timestamp"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def create_versioned_filename(base_name, extension, version=None):
    """
    Crea un nombre de archivo versionado.
    Útil para guardar diferentes versiones de modelos o datos.
    """
    if version is None:
        version = create_timestamp_id()
    
    return f"{base_name}_{version}.{extension}"

def ensure_directory_exists(directory_path):
    """Asegura que un directorio existe, creándolo si es necesario"""
    os.makedirs(directory_path, exist_ok=True)

def memory_usage_info(df):
    """
    Calcula información sobre el uso de memoria de un DataFrame.
    Útil para optimizar el almacenamiento.
    """
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    memory_per_column = {col: memory_usage[i] for i, col in enumerate(df.columns)}
    
    usage_info = {
        'total_memory_mb': total_memory / 1024 / 1024,
        'memory_per_column_mb': {k: v / 1024 / 1024 for k, v in memory_per_column.items()}
    }
    
    return usage_info

def optimize_dtypes(df):
    """
    Optimiza los tipos de datos de un DataFrame para reducir uso de memoria.
    Útil para conjuntos grandes de datos.
    """
    optimized_df = df.copy()
    
    # Para columnas enteras
    int_columns = df.select_dtypes(include=['int']).columns
    for col in int_columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        # Seleccionar el tipo de dato más eficiente
        if col_min >= 0:
            if col_max < 2**8:
                optimized_df[col] = df[col].astype(np.uint8)
            elif col_max < 2**16:
                optimized_df[col] = df[col].astype(np.uint16)
            elif col_max < 2**32:
                optimized_df[col] = df[col].astype(np.uint32)
        else:
            if col_min > -2**7 and col_max < 2**7:
                optimized_df[col] = df[col].astype(np.int8)
            elif col_min > -2**15 and col_max < 2**15:
                optimized_df[col] = df[col].astype(np.int16)
            elif col_min > -2**31 and col_max < 2**31:
                optimized_df[col] = df[col].astype(np.int32)
    
    # Para columnas float
    float_columns = df.select_dtypes(include=['float']).columns
    for col in float_columns:
        optimized_df[col] = df[col].astype(np.float32)
    
    # Para columnas categóricas
    cat_columns = df.select_dtypes(include=['object']).columns
    for col in cat_columns:
        if df[col].nunique() / df.shape[0] < 0.5:  # Si la cardinalidad es baja
            optimized_df[col] = df[col].astype('category')
    
    return optimized_df

def generate_data_profile(df):
    """
    Genera un perfil básico del DataFrame.
    Incluye estadísticas, distribuciones y otros insights.
    """
    profile = {}
    
    # Información básica
    profile['shape'] = df.shape
    profile['memory_usage'] = memory_usage_info(df)
    profile['dtypes'] = df.dtypes.astype(str).to_dict()
    
    # Estadísticas descriptivas
    try:
        profile['summary'] = df.describe(include='all').to_dict()
    except:
        profile['summary'] = "Error generando resumen estadístico"
    
    # Valores nulos
    missing_values = df.isnull().sum()
    profile['missing_values'] = {
        'total': missing_values.sum(),
        'per_column': missing_values.to_dict(),
        'percentage': (missing_values / len(df) * 100).to_dict()
    }
    
    # Cardinalidad de cada columna
    profile['cardinality'] = {col: df[col].nunique() for col in df.columns}
    
    return profile

def batch_generator(df, batch_size=1000):
    """
    Generador de lotes para procesar DataFrames grandes.
    Útil para procesar grandes volúmenes de datos sin agotar memoria.
    """
    n_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        yield df.iloc[start_idx:end_idx]

def get_model_info(model):
    """
    Extrae información útil de un modelo scikit-learn/XGBoost.
    Útil para documentación y registro de experimentos.
    """
    model_info = {
        'model_type': type(model).__name__,
        'parameters': model.get_params()
    }
    
    if hasattr(model, 'feature_importances_'):
        model_info['has_feature_importances'] = True
    else:
        model_info['has_feature_importances'] = False
    
    if hasattr(model, 'classes_'):
        model_info['classes'] = model.classes_.tolist()
    
    return model_info

def safe_mlflow_start(experiment_name, run_name=None):
    """Inicia de forma segura una ejecución de MLflow"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        return mlflow.start_run(run_name=run_name)
    except Exception as e:
        logger.warning(f"Error al iniciar MLflow: {e}")
        from contextlib import nullcontext
        return nullcontext()

def safe_mlflow_log_param(key, value):
    """Registra un parámetro en MLflow de forma segura"""
    try:
        mlflow.log_param(key, value)
        return True
    except Exception as e:
        logger.warning(f"Error al registrar parámetro {key} en MLflow: {e}")
        return False

def safe_mlflow_log_metric(key, value):
    """Registra una métrica en MLflow de forma segura"""
    try:
        mlflow.log_metric(key, value)
        return True
    except Exception as e:
        logger.warning(f"Error al registrar métrica {key} en MLflow: {e}")
        return False

def safe_mlflow_log_artifact(local_path, artifact_path=None):
    """Registra un artefacto en MLflow de forma segura"""
    try:
        mlflow.log_artifact(local_path, artifact_path)
        return True
    except Exception as e:
        logger.warning(f"Error al registrar artefacto {local_path} en MLflow: {e}")
        return False

def safe_mlflow_log_dict(dictionary, artifact_file):
    """Registra un diccionario en MLflow de forma segura"""
    try:
        # Primero guardamos el diccionario localmente
        import json
        import tempfile
        
        # Crear un archivo temporal
        temp_file = os.path.join(MLFLOW_ARTIFACT_ROOT, f"{artifact_file}")
        os.makedirs(os.path.dirname(temp_file), exist_ok=True)
        
        with open(temp_file, 'w') as f:
            json.dump(dictionary, f, indent=2, default=str)
        
        # Luego intentamos registrarlo en MLflow
        return safe_mlflow_log_artifact(temp_file)
    except Exception as e:
        logger.warning(f"Error al registrar diccionario en MLflow: {e}")
        return False