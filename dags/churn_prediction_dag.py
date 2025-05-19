"""
DAG Principal para Predicción de Churn en Telecomunicaciones
===========================================================

Este DAG orquesta el flujo completo de trabajo para la predicción de abandono (churn)
implementando una arquitectura modular con integración de MLflow, PostgreSQL y monitoreo.

Autor: Teresa
Fecha: Mayo 2025
"""

from datetime import datetime, timedelta
import os
import sys

# Añadir el directorio de módulos al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'churn_modules'))

# Importar módulos personalizados
from churn_modules.config import (
    BASE_DIR, MLFLOW_TRACKING_URI, POSTGRES_CONN_ID, DATA_PATH, 
    PROCESSED_DATA_PATH, MODEL_PATH, HOLDOUT_PATH, METRICS_PATH
)
from churn_modules.data_preparation import prepare_data
from churn_modules.model_training import train_model
from churn_modules.model_evaluation import evaluate_model
from churn_modules.db_operations import setup_postgres_tables
from churn_modules.reporting import send_summary_report

# Importaciones de Airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Argumentos por defecto para el DAG
default_args = {
    'owner': 'data_science_team',
    'depends_on_past': False,
    'email': ['data_science@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=1),
}

# Definición del DAG
dag = DAG(
    'churn_prediction_advanced',
    default_args=default_args,
    description='Pipeline de predicción de churn para telecomunicaciones',
    schedule_interval='@weekly',  # Ejecutar semanalmente
    start_date=days_ago(1),
    catchup=False,  # No ejecutar ejecuciones atrasadas
    tags=['churn', 'telecomunicaciones', 'prediccion', 'xgboost', 'mlflow', 'postgresql'],
)

# Definir la tarea de configuración de PostgreSQL
task_setup_postgres = PythonOperator(
    task_id='setup_postgres_tables',
    python_callable=setup_postgres_tables,
    provide_context=True,
    dag=dag,
)

# Definir tarea de preparación de datos
task_prepare_data = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data,
    provide_context=True,
    dag=dag,
)

# Definir tarea de entrenamiento del modelo
task_train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)

# Definir tarea de evaluación del modelo
task_evaluate_model = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    provide_context=True,
    dag=dag,
)

# Tarea para enviar resumen
task_send_summary = PythonOperator(
    task_id='send_summary',
    python_callable=send_summary_report,
    provide_context=True,
    dag=dag,
)

# Definir el orden de ejecución de las tareas
task_setup_postgres >> task_prepare_data >> task_train_model >> task_evaluate_model >> task_send_summary