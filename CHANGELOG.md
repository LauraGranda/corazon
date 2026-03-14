# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

## [Unreleased]

### Added

- Módulo de limpieza y validación de datos (`src/data/clean_data.py`):
    - 6 funciones de limpieza reutilizables: `standardize_nulls()`, `clean_strings()`, `invalidate_categorical()`, `invalidate_numeric()`, `validate_dataframe()`, `cast_types()`
    - 8 constantes de esquema (VALID_*) basadas en `datos_corazon_Info.txt`
    - Soporte para nullable Int64, boolean, y tipos categoriales ordenados/desordenados
    - Type hints completos y docstrings en estilo Google

- Suite completa de tests para limpieza de datos (`tests/data/test_clean_data.py`):
    - 51 pruebas pytest organizadas en 5 clases (TestStandardizeNulls, TestCleanStrings, TestInvalidateCategorical, TestInvalidateNumeric, TestValidateDataframe, TestCastTypes)
    - Cobertura integral: estandarización de nulls, limpieza de strings, validación de categorías/numéricos, casting de tipos
    - Pruebas de inmutabilidad, NaN preservation, y casting a tipos nullable

- Nuevas fixtures pytest en `tests/data/conftest.py`:
    - `messy_dataframe`: DataFrame con 6 filas con problemas de calidad (nulls irregulares, espacios, valores basura)
    - `pre_cast_dataframe`: DataFrame en estado pre-cast para tests de type casting

### Changed

- Refactorización del pipeline de limpieza de datos:
    - Extracción de funciones del notebook `notebooks/2-exploration/01_LMG_explore.ipynb` a módulo reutilizable `src/data/clean_data.py`
    - Aplicación del Single Responsibility Principle: funciones pequeñas y decoupled
    - Mejora de mantenibilidad y testabilidad del código de limpieza

### Fixed

- Resolución de 10 violaciones de ruff linter:
    - Reemplazo de comparaciones directas con True/False por aserciones Pythónicas (`assert result` en lugar de `assert result == True`)
    - Eliminación de valores mágicos en tests (PLR2004): uso de constantes nombradas (AGE_TEST_VALUE, REST_BP_TEST_VALUE, etc.)
    - Formateo correcto de archivos según ruff formatter
    - Importaciones innecesarias removidas

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
