# Proyecto de Predicción de Churn en Telecomunicaciones (Ficticio)

Este proyecto implementa un pipeline completo de Machine Learning para predecir el abandono de clientes (churn) en empresas de telecomunicaciones usando Apache Airflow, XGBoost, MLflow y PostgreSQL.

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Arquitectura](#️-arquitectura)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Configuración](#️-configuración)
- [Ejecución](#-ejecución)
- [Monitoreo](#-monitoreo)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Explicación del Pipeline](#-explicación-del-pipeline)
- [Métricas y Evaluación](#-métricas-y-evaluación)
- [Troubleshooting](#-troubleshooting)
- [Contribución](#-contribución)

## 🎯 Descripción del Proyecto

Este proyecto desarrolla un modelo predictivo para identificar clientes con alta probabilidad de abandono en una empresa de telecomunicaciones. El objetivo es implementar un pipeline de MLOps completo que incluya:

- **Preparación automatizada de datos**
- **Entrenamiento de modelo XGBoost**
- **Evaluación y explicabilidad (SHAP)**
- **Registro de experimentos (MLflow)**
- **Monitoreo en tiempo real (Prometheus/Grafana)**
- **Almacenamiento de resultados (PostgreSQL)**

## 🏗️ Arquitectura

```
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   Apache        │   │   PostgreSQL    │   │    MLflow       │
│   Airflow       │   │   (Datos +      │   │  (Experimentos) │
│  (Orquestación) │←→ │   Resultados)   │→  │                 │
└─────────────────┘   └─────────────────┘   └─────────────────┘
         ↓                       ↓                       ↓
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   Prometheus    │   │     Grafana     │   │   XGBoost       │
│  (Métricas)     │←→ │  (Dashboard)    │   │   (Modelo)      │
└─────────────────┘   └─────────────────┘   └─────────────────┘
```

## 📋 Requisitos

### Software
- **Docker** >= 20.10.0
- **Docker Compose** >= 2.0.0
- **Git**
- Al menos **8GB de RAM** disponible
- Al menos **10GB de espacio en disco**

### Sistema Operativo
- Linux (recomendado)
- macOS
- Windows 10/11 con WSL2

## 🔧 Instalación

### 1. Clonar el Repositorio

```bash
git clone <url-del-repositorio>
cd churn-prediction-project
```

### 2. Configurar el Entorno

```bash
# Crear directorios necesarios
mkdir -p dags logs plugins data

# Configurar permisos (Linux/macOS)
export AIRFLOW_UID=$(id -u)
echo -e "AIRFLOW_UID=${AIRFLOW_UID}" > .env

# Para Windows (PowerShell)
# $AIRFLOW_UID = 1000
# echo "AIRFLOW_UID=$AIRFLOW_UID" > .env
```

### 3. Preparar los Datos

```bash
# Copiar el dataset al directorio de datos
cp dataset.csv ./data/dataset.csv

# Verificar que el archivo está en la ubicación correcta
ls -la ./data/
```

## ⚙️ Configuración

### Configuración de Airflow

```bash
# Inicializar la base de datos de Airflow
docker-compose up airflow-init

# Esperar a que la inicialización termine (ver mensaje de éxito)
```

### Configuración de Conexiones

Una vez que Airflow esté ejecutándose, configurar las conexiones:

1. **Acceder a Airflow Web UI**: http://localhost:8080
   - Usuario: `airflow`
   - Contraseña: `airflow`

2. **Configurar conexión PostgreSQL**:
   - Ir a `Admin` → `Connections`
   - Crear nueva conexión:
     - **Connection Id**: `postgres_default`
     - **Connection Type**: `Postgres`
     - **Host**: `postgres-ml`
     - **Database**: `churn_prediction`
     - **Login**: `mluser`
     - **Password**: `mlpassword`
     - **Port**: `5432`

## 🚀 Ejecución

### 1. Iniciar Todos los Servicios

```bash
# Iniciar todos los contenedores
docker-compose up -d

# Verificar que todos los servicios están ejecutándose
docker-compose ps
```

### 2. Verificar el Estado de los Servicios

Esperar a que todos los servicios estén saludables:

```bash
# Verificar logs de Airflow
docker-compose logs airflow-webserver

# Verificar logs de PostgreSQL
docker-compose logs postgres-ml

# Verificar logs de MLflow
docker-compose logs mlflow
```

### 3. Acceso a las Interfaces Web

| Servicio | URL | Credenciales |
|----------|-----|--------------|
| **Airflow** | http://localhost:8080 | airflow/airflow |
| **MLflow** | http://localhost:5000 | No requiere |
| **Grafana** | http://localhost:3000 | admin/admin |
| **Prometheus** | http://localhost:9090 | No requiere |

### 4. Ejecutar el Pipeline

1. **En Airflow Web UI**:
   - Buscar el DAG: `churn_prediction_dag`
   - Activar el DAG (toggle ON)
   - Hacer clic en "Trigger DAG" para ejecutar manualmente

2. **Monitorear la Ejecución**:
   - Ver el progreso en la vista de Grid o Graph
   - Revisar logs de cada tarea
   - Verificar que todas las tareas se completen exitosamente

## 📊 Monitoreo

### MLflow Tracking

- **Experimentos**: http://localhost:5000
- Ver métricas, parámetros y artefactos de cada ejecución
- Comparar diferentes versiones del modelo

### Dashboard de Grafana

- **Dashboard**: http://localhost:3000
- Username: `admin`, Password: `admin`
- Ver métricas en tiempo real del modelo
- Monitorear rendimiento y predicciones

### Base de Datos PostgreSQL

```bash
# Conectar a PostgreSQL para consultar resultados
docker exec -it <postgres-ml-container> psql -U mluser -d churn_prediction

# Consultas útiles
\dt  # Listar tablas
SELECT * FROM churn_model_metrics ORDER BY training_date DESC LIMIT 5;
SELECT * FROM churn_feature_importance WHERE model_id = 1;
```

## 📁 Estructura del Proyecto

```
churn-prediction-project/
├── docker-compose.yaml          # Configuración de servicios
├── Dockerfile                   # Imagen personalizada de Airflow
├── Dockerfile.mlflow           # Imagen personalizada de MLflow
├── init-ml-db.sql              # Script de inicialización de BD
├── prometheus/          
│   └──prometheus.yml             # Configuración de Prometheus
├── grafana/          
│   └──grafana-dashboard.json     # Dashboard de Grafana
├── data/                       # Datos del proyecto
│   └── dataset.csv            # Dataset principal
├── dags/                       # DAGs de Airflow
│   ├── churn_prediction_dag.py # DAG principal
│   └── churn_modules/          # Módulos del proyecto
│       ├── __init__.py
│       ├── config.py           # Configuración centralizada
│       ├── data_preparation.py # Preparación de datos
│       ├── model_training.py   # Entrenamiento del modelo
│       ├── model_evaluation.py # Evaluación del modelo
│       ├── db_operations.py    # Operaciones de BD
│       ├── reporting.py        # Generación de reportes
│       └── utils.py            # Utilidades
├── logs/                       # Logs de Airflow
└── plugins/                    # Plugins personalizados
```

## 🔍 Explicación del Pipeline

### 1. Setup PostgreSQL (`setup_postgres_tables`)
- Crea las tablas necesarias en PostgreSQL
- Configura índices para optimización
- Establece permisos adecuados

### 2. Preparación de Datos (`prepare_data`)
- Carga el dataset CSV
- Limpia y formatea datos
- Maneja valores faltantes y outliers
- Codifica variables categóricas
- Selecciona características importantes
- Escala datos y divide en train/holdout
- Registra todo en MLflow

### 3. Entrenamiento del Modelo (`train_model`)
- Configura hiperparámetros de XGBoost
- Realiza validación cruzada
- Entrena el modelo final
- Calcula importancia de características
- Guarda el modelo entrenado
- Registra métricas en MLflow

### 4. Evaluación del Modelo (`evaluate_model`)
- Evalúa el modelo en conjunto holdout
- Calcula métricas de rendimiento
- Genera visualizaciones (ROC, Confusion Matrix)
- Crea explicaciones SHAP
- Analiza clientes específicos
- Guarda resultados en PostgreSQL

### 5. Reporte Summary (`send_summary`)
- Genera reporte final
- Envía notificaciones (simuladas)
- Crea archivos de integración

## 📈 Métricas y Evaluación

### Métricas del Modelo
- **Accuracy**: Precisión general del modelo
- **Precision**: Proporción de verdaderos positivos
- **Recall**: Capacidad de detectar churners
- **F1-Score**: Media armónica de precision y recall
- **AUC-ROC**: Área bajo la curva ROC

### Explicabilidad SHAP
- Análisis de importancia de características
- Explicaciones a nivel individual
- Visualizaciones de factores de riesgo

### Monitoreo Continuo
- Deriva de datos (data drift)
- Rendimiento del modelo en producción
- Métricas de negocio

## 🔧 Troubleshooting

### Problemas Comunes

#### Error: "No se puede conectar a PostgreSQL"
```bash
# Verificar que el contenedor está ejecutándose
docker ps | grep postgres

# Revisar logs
docker-compose logs postgres-ml

# Reiniciar el servicio
docker-compose restart postgres-ml
```

#### Error: "Airflow DAG no aparece"
```bash
# Verificar que los archivos están en dags/
ls -la ./dags/

# Reiniciar scheduler
docker-compose restart airflow-scheduler

# Verificar logs
docker-compose logs airflow-scheduler
```

#### Error: "MLflow no registra experimentos"
```bash
# Verificar conectividad
curl http://localhost:5000

# Revisar logs
docker-compose logs mlflow

# Verificar permisos de volúmenes
ls -la mlflow-artifacts/
```

#### Memoria Insuficiente
```bash
# Verificar uso de memoria
docker stats

# Reducir worker de Airflow si es necesario
# En docker-compose.yaml, comentar airflow-worker
```

### Logs Útiles

```bash
# Ver todos los logs
docker-compose logs -f

# Logs específicos por servicio
docker-compose logs -f airflow-webserver
docker-compose logs -f postgres-ml
docker-compose logs -f mlflow

# Logs del DAG específico
docker-compose exec airflow-webserver airflow dags list
docker-compose exec airflow-webserver airflow tasks list churn_prediction_dag
```

### Reinicio Completo

```bash
# Parar todos los servicios
docker-compose down

# Limpiar volúmenes (CUIDADO: borra todos los datos)
docker-compose down -v

# Limpiar todo y reiniciar
docker-compose up airflow-init
docker-compose up -d
```

## 🧪 Testing

### Verificar la Instalación

```bash
# Test 1: Verificar servicios
curl -f http://localhost:8080/health
curl -f http://localhost:5000/health
curl -f http://localhost:3000/api/health

# Test 2: Verificar DAG
docker-compose exec airflow-webserver airflow dags list | grep churn_prediction_dag

# Test 3: Verificar conexión a BD
docker exec postgres-ml-container psql -U mluser -d churn_prediction -c "SELECT 1;"
```

### Ejecución de Prueba

```bash
# Ejecutar DAG programáticamente
docker-compose exec airflow-webserver airflow dags trigger churn_prediction_dag

# Verificar estado
docker-compose exec airflow-webserver airflow dags state churn_prediction_dag $(date +%Y-%m-%d)
```

## 📚 Documentación y Webgrafía

### Recursos MLflow
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)

### Recursos Airflow
- [Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)

### Recursos XGBoost
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
