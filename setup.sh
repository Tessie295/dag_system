#!/bin/bash

# Colores para mejor visualización
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Configurando entorno de Apache Airflow con Docker ===${NC}"

# Crear estructura de carpetas
echo -e "${GREEN}Creando estructura de carpetas...${NC}"
mkdir -p ./dags ./logs ./plugins
chmod -R 777 ./dags ./logs ./plugins

# Verificar si docker compose.yaml existe
if [ ! -f docker-compose.yaml ]; then
    echo -e "${GREEN}Descargando docker-compose.yaml...${NC}"
    curl -LfO 'https://airflow.apache.org/docs/apache-airflow/stable/docker-compose.yaml'
    
    # Modificar el archivo para usar LocalExecutor en lugar de CeleryExecutor
    echo -e "${GREEN}Configurando para usar LocalExecutor...${NC}"
    sed -i 's/CeleryExecutor/LocalExecutor/g' docker-compose.yaml
    
    # Desactivar ejemplos de carga
    sed -i "s/AIRFLOW__CORE__LOAD_EXAMPLES: 'true'/AIRFLOW__CORE__LOAD_EXAMPLES: 'false'/g" docker-compose.yaml
else
    echo -e "${GREEN}El archivo docker-compose.yaml ya existe.${NC}"
fi

# Inicializar Airflow
echo -e "${GREEN}Inicializando la base de datos de Airflow...${NC}"
docker compose up airflow-init

# Instrucciones finales
echo -e "${BLUE}=== Configuración completada ===${NC}"
echo -e "${GREEN}Para iniciar Airflow, ejecuta:${NC} docker compose up -d"
echo -e "${GREEN}Accede a la interfaz web:${NC} http://localhost:8080"
echo -e "${GREEN}Usuario:${NC} airflow"
echo -e "${GREEN}Contraseña:${NC} airflow"
echo -e "${GREEN}Tus DAGs deben colocarse en la carpeta:${NC} ./dags"
