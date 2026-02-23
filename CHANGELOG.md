# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

## [Unreleased]

## [1.1.0] - 2026-02-22

### Added

- Nuevo notebook para exploración de datos (`notebooks/2-exploration/01_LMG_exploration_22_02_2026.ipynb`)
    - Exploración, limpieza y validación de datos cardíacos
    - Detección de valores no estándar, espacios, saltos de línea y valores nulos explícitos
    - Conversión de tipos de datos: nominales, ordinales, binarios y numéricos
    - Documentación del orden de columnas categóricas ordinales (`rest_ecg`, `slope`, `ca`)
    - Guardado de datos limpios en `data/03_primary/corazon_primary.parquet`

### Changed

- Mejora en el proceso de limpieza de datos con manejo avanzado de anomalías

### Fixed

- Corrección en la conversión de columnas ordinales a tipo `category` con orden

## [1.0.0] - 2026-02-21

### Added

- Nuevo notebook para la descarga de datos del corazón (`notebooks/1-data/01_LMG_download_data_21_02_2026.ipynb`)
    - Implementación del pipeline de descarga de datos from external sources
    - Integración con el flujo de trabajo de preparación de datos

### Changed

### Deprecated

### Removed

### Fixed

### Security
