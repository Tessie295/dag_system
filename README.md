# Configuración de Airflow con Docker

Este proyecto contiene la configuración necesaria para ejecutar Apache Airflow utilizando Docker.

## Estructura de Carpetas

```
airflow-project/
├── dags/           # Carpeta para almacenar los DAGs
├── logs/           # Carpeta para almacenar los logs
├── plugins/        # Carpeta para almacenar plugins
└── docker-compose.yaml  # Archivo de configuración de Docker Compose
```

## Requisitos Previos

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)
- Si estás en Windows 10, se recomienda usar WSL2

## Pasos para Iniciar Airflow

1. **Inicializar la base de datos**:
   ```bash
   docker-compose up airflow-init
   ```

2. **Iniciar Airflow**:
   ```bash
   docker-compose up
   ```

3. **Acceder a la Interfaz Web**:
   - URL: http://localhost:8080/
   - Usuario: airflow
   - Contraseña: airflow

## Comandos Útiles

- Para detener los contenedores:
  ```bash
  docker-compose down
  ```

- Para reiniciar los contenedores:
  ```bash
  docker-compose restart
  ```

- Para ver los logs:
  ```bash
  docker-compose logs
  ```

- Para entrar al shell del contenedor webserver:
  ```bash
  docker exec -it airflow_webserver bash
  ```

## Trabajando con DAGs

Los DAGs deben colocarse en la carpeta `dags/`. Una vez que agregues un nuevo DAG, este aparecerá automáticamente en la interfaz web después de un breve período de tiempo.

## Notas Importantes

- Esta configuración utiliza el ejecutor `LocalExecutor` que es adecuado para entornos de desarrollo y prueba.
- Asegúrate de tener suficiente memoria asignada a Docker (al menos 4GB, idealmente 8GB).
- Para entornos de producción, considera usar Kubernetes con el Chart oficial de Airflow.
