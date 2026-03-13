# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

## [Unreleased]

## [1.1.0] - 2026-03-13

### Added

- Módulo de carga de datos (`src/data/loader.py`):
    - Funciones `load_data()`, `explore_data()` y `save_data()` reutilizables
    - Soporte de archivos CSV y exportación a Parquet
    - Manejo seguro de DataFrames pequeños y vacíos

- Suite completa de tests unitarios (`tests/data/test_loader.py`):
    - 23 pruebas pytest organizadas en 3 clases (TestLoadData, TestExploreData, TestSaveData)
    - Cobertura integral: carga de datos, exploración, guardado y casos edge
    - 6 fixtures pytest en `tests/data/conftest.py` con datos de prueba variados
    - Tests independientes, rápidos (0.25s) y con nombres descriptivos

### Changed

- Refactorización del código de datos:
    - Extracción de funciones a módulo reutilizable `src/data/loader.py`
    - Mejora de manejo de DataFrames pequeños en `explore_data()` con lógica segura de muestreo

### Fixed

- Resolución de 6 violaciones de ruff linter:
    - Reemplazo de magic numbers (5, 14, 63, 100) con constantes nombradas
    - Corrección de rutas inseguras de archivos temporales usando pytest fixtures
    - Eliminación de variable no utilizada en tests de sobrescritura

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
