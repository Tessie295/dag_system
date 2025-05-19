"""
Operaciones relacionadas con la base de datos para el proyecto de predicción de churn.
Configuración de tablas, guardado y recuperado de datos.
"""

import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook
from .config import POSTGRES_CONN_ID, logger

def get_postgres_connection():
    """Obtiene una conexión a PostgreSQL"""
    try:
        pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        return pg_hook.get_conn()
    except Exception as e:
        logger.error(f"Error al conectar a PostgreSQL: {e}")
        raise

def save_to_postgres(df, table_name, if_exists='replace'):
    """Guarda un DataFrame en PostgreSQL"""
    try:
        pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        df.to_sql(table_name, pg_hook.get_sqlalchemy_engine(), if_exists=if_exists, index=False)
        logger.info(f"Datos guardados en tabla {table_name} de PostgreSQL")
        return True
    except Exception as e:
        logger.error(f"Error al guardar datos en PostgreSQL: {e}")
        logger.error(f"Tabla: {table_name}, Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
        return False

def run_sql_query(query, parameters=None, return_results=True):
    """Ejecuta una consulta SQL en PostgreSQL"""
    try:
        pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        if return_results:
            return pg_hook.get_records(query, parameters)
        else:
            pg_hook.run(query, parameters)
            return True
    except Exception as e:
        logger.error(f"Error al ejecutar consulta SQL: {e}")
        logger.error(f"Query: {query}")
        return None if return_results else False

def setup_postgres_tables(**kwargs):
    """
    Configura las tablas necesarias en PostgreSQL para el proyecto.
    """
    logger.info("Configurando tablas en PostgreSQL...")
    
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    
    # Crear tabla para los datos procesados
    pg_hook.run("""
    CREATE TABLE IF NOT EXISTS churn_processed_data (
        id SERIAL PRIMARY KEY,
        customer_id VARCHAR(50),
        churn INTEGER,
        processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Crear tabla para las métricas del modelo
    pg_hook.run("""
    CREATE TABLE IF NOT EXISTS churn_model_metrics (
        model_id SERIAL PRIMARY KEY,
        model_type VARCHAR(50),
        accuracy FLOAT,
        precision FLOAT,
        recall FLOAT,
        f1_score FLOAT,
        auc_roc FLOAT,
        training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Crear tabla para la importancia de características
    pg_hook.run("""
    CREATE TABLE IF NOT EXISTS churn_feature_importance (
        id SERIAL PRIMARY KEY,
        model_id INT REFERENCES churn_model_metrics(model_id),
        feature_name VARCHAR(100),
        importance_value FLOAT,
        rank INT
    );
    """)
    
    # Crear tabla para importancia de características temporal
    pg_hook.run("""
    CREATE TABLE IF NOT EXISTS churn_feature_importance_temp (
        id SERIAL PRIMARY KEY,
        model_id INT,
        feature_name VARCHAR(100),
        importance_value FLOAT,
        rank INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Crear tabla para predicciones individuales
    pg_hook.run("""
    CREATE TABLE IF NOT EXISTS churn_predictions (
        prediction_id SERIAL PRIMARY KEY,
        customer_id VARCHAR(50),
        model_id INT REFERENCES churn_model_metrics(model_id),
        churn_probability FLOAT,
        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Crear tabla para valores SHAP (explicabilidad)
    pg_hook.run("""
    CREATE TABLE IF NOT EXISTS churn_shap_values (
        id SERIAL PRIMARY KEY,
        prediction_id INTEGER REFERENCES churn_predictions(prediction_id),
        feature_name VARCHAR(100) NOT NULL,
        feature_value FLOAT,
        shap_value FLOAT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Crear tabla para eventos de monitoreo
    pg_hook.run("""
    CREATE TABLE IF NOT EXISTS churn_model_monitoring (
        id SERIAL PRIMARY KEY,
        model_id INTEGER REFERENCES churn_model_metrics(model_id),
        event_type VARCHAR(50) NOT NULL,
        event_description TEXT,
        metrics JSONB,
        event_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    logger.info("Tablas en PostgreSQL configuradas correctamente.")
    
    return "Configuración de PostgreSQL completada"

def save_metrics_to_postgres(metrics, model_type='XGBoost'):
    """Guarda métricas de evaluación del modelo en PostgreSQL y devuelve el ID del modelo"""
    from datetime import datetime
    
    try:
        # Insertar métricas del modelo
        model_id_query = run_sql_query("""
            INSERT INTO churn_model_metrics 
            (model_type, accuracy, precision, recall, f1_score, auc_roc, training_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING model_id;
        """, parameters=(
            model_type,
            metrics['accuracy'], 
            metrics['precision'], 
            metrics['recall'], 
            metrics['f1'],
            metrics['auc_roc'],
            datetime.now()
        ))
        
        if model_id_query and len(model_id_query) > 0:
            model_id = model_id_query[0][0]
            logger.info(f"Métricas guardadas en PostgreSQL (model_id: {model_id})")
            return model_id
        else:
            logger.warning("No se pudo obtener model_id después de insertar métricas")
            return None
    except Exception as e:
        logger.error(f"Error al guardar métricas en PostgreSQL: {e}")
        return None

def save_feature_importance(feature_importance_df, model_id):
    """Guarda la importancia de características en PostgreSQL"""
    if model_id is None:
        logger.warning("No se puede guardar importancia de características sin model_id")
        return False
    
    try:
        # Añadir model_id y rank si no existen
        if 'model_id' not in feature_importance_df.columns:
            feature_importance_df['model_id'] = model_id
        
        if 'rank' not in feature_importance_df.columns:
            feature_importance_df['rank'] = feature_importance_df.reset_index().index + 1
        
        # Guardar en PostgreSQL
        return save_to_postgres(
            feature_importance_df[['model_id', 'feature_name', 'importance_value', 'rank']], 
            'churn_feature_importance', 
            'append'
        )
    except Exception as e:
        logger.error(f"Error al guardar importancia de características: {e}")
        return False

def save_predictions(predictions_df, model_id):
    """Guarda predicciones en PostgreSQL"""
    if model_id is None:
        logger.warning("No se puede guardar predicciones sin model_id")
        return False
    
    try:
        # Añadir model_id si no existe
        if 'model_id' not in predictions_df.columns:
            predictions_df['model_id'] = model_id
        
        # Guardar en PostgreSQL
        return save_to_postgres(
            predictions_df[['customer_id', 'model_id', 'churn_probability']], 
            'churn_predictions', 
            'append'
        )
    except Exception as e:
        logger.error(f"Error al guardar predicciones: {e}")
        return False

def save_monitoring_event(model_id, event_type, event_description, metrics=None):
    """Registra un evento de monitoreo en PostgreSQL"""
    import json
    
    if model_id is None:
        logger.warning("No se puede registrar evento sin model_id")
        return False
    
    try:
        metrics_json = json.dumps(metrics) if metrics else None
        
        run_sql_query("""
            INSERT INTO churn_model_monitoring 
            (model_id, event_type, event_description, metrics)
            VALUES (%s, %s, %s, %s);
        """, parameters=(
            model_id,
            event_type,
            event_description,
            metrics_json
        ), return_results=False)
        
        logger.info(f"Evento de monitoreo registrado: {event_type}")
        return True
    except Exception as e:
        logger.error(f"Error al registrar evento de monitoreo: {e}")
        return False