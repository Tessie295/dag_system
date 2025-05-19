-- Script de inicialización de la base de datos para el proyecto de predicción de churn
-- Este script se ejecutará automáticamente al iniciar el contenedor postgres-ml

-- Crear tablas para almacenar datos del proyecto
CREATE TABLE IF NOT EXISTS churn_raw_data (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    -- Añadir aquí todas las columnas originales del dataset
    -- Estas serán completadas dinámicamente por el DAG
    churn INTEGER,
    processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS churn_processed_data (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    -- Las columnas finales se añadirán dinámicamente por el DAG
    churn INTEGER,
    processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para versiones de modelos entrenados
CREATE TABLE IF NOT EXISTS churn_model_metrics (
    model_id SERIAL PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,
    accuracy FLOAT NOT NULL,
    precision FLOAT NOT NULL,
    recall FLOAT NOT NULL,
    f1_score FLOAT NOT NULL,
    auc_roc FLOAT NOT NULL,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hyperparameters JSONB,
    description TEXT
);

-- Tabla para importancia de características
CREATE TABLE IF NOT EXISTS churn_feature_importance (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES churn_model_metrics(model_id),
    feature_name VARCHAR(100) NOT NULL,
    importance_value FLOAT NOT NULL,
    rank INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla temporal para importancia de características (uso intermedio)
CREATE TABLE IF NOT EXISTS churn_feature_importance_temp (
    id SERIAL PRIMARY KEY,
    model_id INTEGER,
    feature_name VARCHAR(100) NOT NULL,
    importance_value FLOAT NOT NULL,
    rank INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para predicciones individuales
CREATE TABLE IF NOT EXISTS churn_predictions (
    prediction_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    model_id INTEGER REFERENCES churn_model_metrics(model_id),
    churn_probability FLOAT NOT NULL,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para valores SHAP (explicabilidad)
CREATE TABLE IF NOT EXISTS churn_shap_values (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER REFERENCES churn_predictions(prediction_id),
    feature_name VARCHAR(100) NOT NULL,
    feature_value FLOAT,
    shap_value FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para eventos de monitoreo
CREATE TABLE IF NOT EXISTS churn_model_monitoring (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES churn_model_metrics(model_id),
    event_type VARCHAR(50) NOT NULL,
    event_description TEXT,
    metrics JSONB,
    event_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla para planes de acción basados en predicciones
CREATE TABLE IF NOT EXISTS churn_action_plans (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    prediction_id INTEGER REFERENCES churn_predictions(prediction_id),
    action_type VARCHAR(50) NOT NULL,
    action_description TEXT,
    priority INTEGER,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Índices para mejorar rendimiento
CREATE INDEX IF NOT EXISTS idx_customer_id ON churn_predictions(customer_id);
CREATE INDEX IF NOT EXISTS idx_model_id ON churn_predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_prediction_date ON churn_predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_churn_probability ON churn_predictions(churn_probability);

-- Permisos
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mluser;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO mluser;

-- Crear tabla para esquema de MLflow (si aún no existe)
-- MLflow creará sus propias tablas, pero podemos ayudar con la inicialización

-- Mensaje de éxito
SELECT 'Database initialized successfully for Churn Prediction Project' as status;