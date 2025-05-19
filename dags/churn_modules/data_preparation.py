"""
Funciones para la preparación y procesamiento de datos para el proyecto de predicción de churn.
Incluye limpieza, transformación, selección de características y división de datos.
"""

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
from datetime import datetime

from .config import (
    BASE_DIR, MLFLOW_TRACKING_URI, MLFLOW_ARTIFACT_ROOT, DATA_PATH, PROCESSED_DATA_PATH, 
    HOLDOUT_PATH, DATA_PROCESSING_CONFIG, logger, MESSAGES
)
from .db_operations import save_to_postgres
from .utils import format_numeric_columns

def prepare_data(**kwargs):
    """
    Prepara los datos para el modelado: 
    - Carga datos
    - Limpia y formatea
    - Selecciona características
    - Divide en entrenamiento y holdout
    - Guarda los conjuntos procesados
    """
    logger.info("Iniciando preparación de datos...")
    
    # Verificar que el archivo existe
    if not os.path.exists(DATA_PATH):
        error_msg = f"El archivo de datos no existe en {DATA_PATH}. Verifique que el volumen está correctamente montado."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Iniciar el tracking de MLflow para este experimento
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("churn_prediction")
    mlflow.set_experiment_tag("artifact_location", MLFLOW_ARTIFACT_ROOT)
    
    # Iniciar una nueva ejecución en MLflow
    with mlflow.start_run(run_name="data_preparation") as run:
        mlflow.log_param("data_source", DATA_PATH)
        
        # Cargar datos
        logger.info(f"Cargando datos desde {DATA_PATH}")
        try:
            df = pd.read_csv(DATA_PATH, sep=';')
            df = format_numeric_columns(df)
            
            # Registrar información básica en MLflow
            mlflow.log_param("data_rows", df.shape[0])
            mlflow.log_param("data_columns", df.shape[1])
            
            logger.info(f"Datos cargados: {df.shape[0]} filas y {df.shape[1]} columnas")
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            raise
        
        # =========================================================================
        # Preprocesamiento de datos
        # =========================================================================
        
        # 1. Manejo de valores faltantes
        missing_values = handle_missing_values(df)
        
        # 2. Manejo de outliers (capping al percentil 1-99)
        outlier_stats = handle_outliers(df)
        
        # 3. Codificación de variables categóricas
        df_encoded, categorical_stats = encode_categorical_variables(df)
        
        # Guardar en PostgreSQL para su posterior análisis
        try:
            # Guardar solo una muestra para no sobrecargar la base de datos
            sample_size = min(10000, df_encoded.shape[0])
            sample_df = df_encoded.sample(sample_size, random_state=DATA_PROCESSING_CONFIG['random_state'])
            
            # Añadir timestamp de procesamiento
            from datetime import datetime
            sample_df['processed_date'] = datetime.now()
            
            # Guardar en PostgreSQL
            save_to_postgres(sample_df, 'churn_raw_data', if_exists='replace')
            logger.info(f"Muestra de datos guardada en PostgreSQL (tabla: churn_raw_data)")
        except Exception as e:
            logger.warning(f"No se pudo guardar en PostgreSQL: {e}")
            logger.warning("Continuando con el flujo normal...")
        
        # 4. Separar características y variable objetivo
        customer_ids = df_encoded['Customer_ID'] if 'Customer_ID' in df_encoded.columns else None
        X = df_encoded.drop(['churn', 'Customer_ID'], axis=1, errors='ignore')
        y = df_encoded['churn']
        
        # 5. Selección de características
        X_selected, feature_importance = select_features(X, y)
        
        # 6. Escalado de características y división
        X_train, X_holdout, y_train, y_holdout, customer_ids_holdout, scaler = scale_and_split(
            X_selected, y, customer_ids
        )
        
        # 7. Guardar información de procesamiento en MLflow
        log_processing_info_to_mlflow(
            missing_values, outlier_stats, categorical_stats,
            feature_importance, X_train, X_holdout, y_train, y_holdout
        )
        
        # 8. Guardar los datos procesados
        logger.info("Guardando datos procesados...")
        save_processed_data(
            X_train, y_train, X_holdout, y_holdout, 
            customer_ids_holdout, scaler, feature_importance
        )
        
        # 9. Guardar versión procesada en PostgreSQL
        try_save_processed_data_to_postgres(X_train, y_train)
        
    return {
        'status': 'success',
        'message': MESSAGES['data_prep_complete'],
        'num_features': X_selected.shape[1],
        'train_samples': X_train.shape[0],
        'holdout_samples': X_holdout.shape[0]
    }

def handle_missing_values(df):
    """Maneja valores faltantes en el DataFrame"""
    logger.info("Manejo de valores faltantes...")
    missing_values = df.isnull().sum()
    columns_with_missing = missing_values[missing_values > 0].index.tolist()
    
    missing_stats = {}
    for col in columns_with_missing:
        # Verificar si la columna es numérica
        is_numeric = pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype in ['int64', 'float64']
        
        if is_numeric:
            # Para variables numéricas, usar mediana
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            logger.info(f"  - {col}: Rellenado con la mediana: {median_value}")
            missing_stats[col] = {"method": "median", "value": float(median_value)}
        else:
            # Para variables categóricas, usar moda
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
            logger.info(f"  - {col}: Rellenado con la moda: {mode_value}")
            missing_stats[col] = {"method": "mode", "value": mode_value}
    
    # Registrar en MLflow
    mlflow.log_dict(missing_stats, "missing_values_treatment.json")
    
    return missing_stats

def handle_outliers(df):
    """Aplica tratamiento de outliers al DataFrame"""
    logger.info("Aplicando tratamiento de outliers...")
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'churn' in numeric_columns:
        numeric_columns.remove('churn')
    if 'Customer_ID' in numeric_columns:
        numeric_columns.remove('Customer_ID')
    
    # Umbral para percentiles
    lower_percentile = DATA_PROCESSING_CONFIG['outlier_percentile_cutoff']
    upper_percentile = 1 - lower_percentile
    
    outlier_stats = {}
    for col in numeric_columns:
        try:
            # Calcular percentiles para capping
            p01 = df[col].quantile(lower_percentile)
            p99 = df[col].quantile(upper_percentile)
            
            # Contar outliers antes del capping
            outliers_mask = (df[col] < p01) | (df[col] > p99)
            outliers_count = outliers_mask.sum()
            outliers_percentage = (outliers_count / len(df)) * 100
            
            if outliers_percentage > 2:  # Si hay más del 2% de outliers
                logger.info(f"  - {col}: {outliers_count} outliers ({outliers_percentage:.2f}%)")
                
                # Aplicar capping
                df[col] = np.where(df[col] < p01, p01, df[col])
                df[col] = np.where(df[col] > p99, p99, df[col])
                
                logger.info(f"    → Valores limitados al rango [{p01:.2f}, {p99:.2f}]")
                outlier_stats[col] = {
                    "outliers_count": int(outliers_count),
                    "outliers_percentage": float(outliers_percentage),
                    "lower_bound": float(p01),
                    "upper_bound": float(p99)
                }
        except Exception as e:
            logger.error(f"  - Error al procesar outliers en {col}: {e}")
    
    # Registrar en MLflow
    mlflow.log_dict(outlier_stats, "outlier_treatment.json")
    
    return outlier_stats

def encode_categorical_variables(df):
    """Codifica variables categóricas usando one-hot encoding"""
    logger.info("Codificación de variables categóricas...")
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'Customer_ID' in categorical_columns:
        categorical_columns.remove('Customer_ID')
    
    categorical_stats = {}
    if categorical_columns:
        logger.info(f"Variables categóricas identificadas: {len(categorical_columns)}")
        
        for col in categorical_columns:
            unique_values = df[col].nunique()
            logger.info(f"  - {col}: {unique_values} valores únicos")
            categorical_stats[col] = {"unique_values": int(unique_values)}
        
        # Registrar en MLflow
        mlflow.log_dict(categorical_stats, "categorical_variables.json")
        
        # Aplicar One-Hot Encoding
        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        
        logger.info(f"Dimensiones antes de la codificación: {df.shape}")
        logger.info(f"Dimensiones después de la codificación: {df_encoded.shape}")
    else:
        logger.info("No se detectaron variables categóricas.")
        df_encoded = df.copy()
    
    return df_encoded, categorical_stats

def select_features(X, y):
    """Selecciona las características más importantes usando Random Forest"""
    logger.info("Selección de características...")
    
    # Utilizamos Random Forest para seleccionar características importantes
    feature_selector = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        random_state=DATA_PROCESSING_CONFIG['random_state'], 
        n_jobs=-1
    )
    feature_selector.fit(X, y)
    
    # Obtener importancia de características
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_selector.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Seleccionar las N características más importantes o todas si hay menos de N
    top_n = min(DATA_PROCESSING_CONFIG['top_features_limit'], len(feature_importance))
    selected_features = feature_importance.head(top_n)['Feature'].tolist()
    
    logger.info(f"Se seleccionaron {len(selected_features)} características de {X.shape[1]} disponibles")
    
    # Filtrar para mantener solo las características seleccionadas
    X_selected = X[selected_features]
    
    return X_selected, feature_importance

def scale_and_split(X_selected, y, customer_ids=None):
    """Escala las características y divide en conjuntos de entrenamiento y holdout"""
    logger.info("Escalado de características...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_selected)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_selected.columns, index=X_selected.index)
    
    # Guardar información del scaler en MLflow
    mlflow.sklearn.log_model(scaler, "robust_scaler")
    
    # División en conjuntos de entrenamiento y holdout
    logger.info("División en conjuntos de entrenamiento y holdout...")
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_scaled_df, y, 
        test_size=DATA_PROCESSING_CONFIG['test_size'], 
        random_state=DATA_PROCESSING_CONFIG['random_state'], 
        stratify=y
    )
    
    # Si tenemos Customer_IDs, dividirlos también
    customer_ids_holdout = None
    if customer_ids is not None:
        _, customer_ids_holdout = train_test_split(
            customer_ids, 
            test_size=DATA_PROCESSING_CONFIG['test_size'], 
            random_state=DATA_PROCESSING_CONFIG['random_state'], 
            stratify=y
        )
    
    logger.info(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
    logger.info(f"Conjunto de holdout: {X_holdout.shape[0]} muestras")
    logger.info(f"Distribución de clases en entrenamiento: {y_train.value_counts(normalize=True)}")
    logger.info(f"Distribución de clases en holdout: {y_holdout.value_counts(normalize=True)}")
    
    return X_train, X_holdout, y_train, y_holdout, customer_ids_holdout, scaler

def log_processing_info_to_mlflow(missing_values, outlier_stats, categorical_stats, 
                                 feature_importance, X_train, X_holdout, y_train, y_holdout):
    """Registra información de procesamiento en MLflow"""
    # Registrar parámetros y métricas
    mlflow.log_param("train_samples", X_train.shape[0])
    mlflow.log_param("holdout_samples", X_holdout.shape[0])
    mlflow.log_param("train_churn_rate", float(y_train.mean()))
    mlflow.log_param("holdout_churn_rate", float(y_holdout.mean()))
    
    # Guardar feature importance
    feature_importance_path = f"{BASE_DIR}/initial_feature_importance.csv"
    feature_importance.to_csv(feature_importance_path, index=False)
    mlflow.log_artifact(feature_importance_path, "feature_importance")
    
    # Crear y guardar visualización de feature importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 20 Características Más Importantes')
    plt.tight_layout()
    
    feat_importance_plot_path = f"{BASE_DIR}/feature_importance_plot.png"
    plt.savefig(feat_importance_plot_path)
    plt.close()
    
    mlflow.log_artifact(feat_importance_plot_path, "plots")
    
    # Registrar información detallada
    mlflow.log_dict(
        {feature: float(importance) for feature, importance in 
         zip(feature_importance['Feature'].head(20), feature_importance['Importance'].head(20))},
        "feature_importance_top20.json"
    )
    
    # Registrar ubicación de datos procesados
    mlflow.log_param("processed_data_path", PROCESSED_DATA_PATH)
    mlflow.log_param("holdout_data_path", HOLDOUT_PATH)

def save_processed_data(X_train, y_train, X_holdout, y_holdout, customer_ids_holdout, scaler, feature_importance):
    """Guarda los datos procesados en archivos pickle"""
    train_data = {
        'X_train': X_train,
        'y_train': y_train,
        'feature_names': X_train.columns.tolist(),
        'scaler': scaler
    }
    
    holdout_data = {
        'X_holdout': X_holdout,
        'y_holdout': y_holdout,
        'customer_ids': customer_ids_holdout
    }
    
    # Guardar en archivos pickle
    with open(PROCESSED_DATA_PATH, 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(HOLDOUT_PATH, 'wb') as f:
        pickle.dump(holdout_data, f)
    
    logger.info(f"Datos de entrenamiento guardados en {PROCESSED_DATA_PATH}")
    logger.info(f"Datos de holdout guardados en {HOLDOUT_PATH}")
    
    # Guardar información de características para consulta
    feature_importance.to_csv(f'{BASE_DIR}/initial_feature_importance.csv', index=False)

def try_save_processed_data_to_postgres(X_train, y_train):
    """Intenta guardar una muestra de datos procesados en PostgreSQL"""
    try:
        # Preparar datos para PostgreSQL
        processed_data_sample = X_train.sample(min(5000, X_train.shape[0]), random_state=42).copy()
        processed_data_sample['churn'] = y_train.reindex(processed_data_sample.index)
        processed_data_sample['processed_date'] = datetime.now()
        
        # Guardar en PostgreSQL
        save_to_postgres(processed_data_sample, 'churn_processed_data', if_exists='replace')
        logger.info("Datos procesados guardados en PostgreSQL")
        return True
    except Exception as e:
        logger.warning(f"No se pudo guardar en PostgreSQL: {e}")
        return False