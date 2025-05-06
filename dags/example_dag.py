"""
Ejemplo básico de un DAG en Airflow 3.0.0

Este DAG es un ejemplo simple que demuestra la sintaxis básica
y funcionalidades principales de un DAG en Apache Airflow.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Argumentos por defecto para el DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Definición del DAG
dag = DAG(
    'ejemplo_basico',
    default_args=default_args,
    description='Un DAG de ejemplo simple',
    schedule='@daily',  # Nuevo formato en Airflow 3.0.0
    start_date=datetime(2022, 1, 1),
    catchup=False,
    tags=['ejemplo'],
)

# Definir una función para el operador de Python
def print_hello(**kwargs):
    """Función de ejemplo que imprime un mensaje"""
    return 'Hola desde el operador de Python!'

# Tarea 1: Ejecutar un comando bash
tarea_bash = BashOperator(
    task_id='tarea_bash',
    bash_command='echo "Hola, esto es un ejemplo básico de Airflow!" > /opt/airflow/logs/ejemplo_salida.txt',
    dag=dag,
)

# Tarea 2: Ejecutar una función Python
tarea_python = PythonOperator(
    task_id='tarea_python',
    python_callable=print_hello,
    dag=dag,
)

# Tarea 3: Otra tarea bash que depende de las anteriores
tarea_final = BashOperator(
    task_id='tarea_final',
    bash_command='echo "Todas las tareas anteriores se completaron con éxito."',
    dag=dag,
)

# Definir el orden de las tareas (flujo de trabajo)
tarea_bash >> tarea_python >> tarea_final