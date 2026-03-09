# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

## [Unreleased]

## [1.0.0] - 2026-02-21

### Added

- Nuevo notebook para la descarga de datos del corazón (`notebooks/1-data/01_LMG_download_data_21_02_2026.ipynb`)
    - Implementación del pipeline de descarga de datos from external sources
    - Integración con el flujo de trabajo de preparación de datos

### Changed

- Refactorización del notebook `01_LMG_download_data_21_02_2026.ipynb`:
    - Implementación de static typing (type hints) en funciones
    - Aplicación del Single Responsibility Principle con funciones `load_data()` y `explore_data()`
    - Uso de `pathlib.Path` para manejo de rutas siguiendo buenas prácticas
    - Consolidación de exploración de datos en función reutilizable
    - Simplificación de análisis y conclusiones

### Deprecated

### Removed

### Fixed

### Security
