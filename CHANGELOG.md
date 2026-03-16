# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

## [Unreleased]

## [1.6.0] - 2026-03-15

### Added

- Módulo de training pipeline (`src/pipelines/training_pipeline/train_pipeline.py`):
    - Script de producción para entrenar modelo LogisticRegression con GridSearchCV
    - Feature engineering integrado con `create_preprocessor()` del feature pipeline
    - Hyperparameter tuning con cv=5 en parámetros `penalty` (l1, l2) y `C` (0.1, 0.5, 1.0, 5.0)
    - Validación de modelo contra baseline score de 0.70 usando recall metric
    - Serialización de modelo con joblib protocol=5
    - Type hints completos y docstrings

- Suite completa de tests para training pipeline (`tests/pipelines/training_pipeline/test_simple_train_pipeline.py`):
    - Tests para validar existencia del script de pipeline
    - Tests para validar generación del archivo joblib del modelo

- Notebooks de selección y evaluación de modelos (`notebooks/5-models/`):
    - **01-baseline_model.ipynb**: Baseline con LogisticRegression, cross-validation, learning curves y evaluación
    - **01-model_selection.ipynb**: Comparación de 5 modelos (LogisticRegression, RandomForest, SVC, GradientBoosting, KNeighborsClassifier)
    - **01-first_model_logistic_regression.ipynb**: GridSearchCV tuning, MLflow experiment tracking y model serialization

- Notebook de interpretación del modelo (`notebooks/6-interpretation/01-model_interpretation.ipynb`):
    - Feature importance basada en coeficientes absolutos
    - Permutation importance en test set usando recall metric
    - Análisis clínico detallado de hallazgos del modelo

- Integración MLflow para experiment tracking:
    - Configuración de tracking URI en `mlruns/`
    - Logging de modelos con signature inference
    - Logging de métricas y hiperparámetros

## [1.5.0] - 2026-03-15

### Added

- Módulo de feature engineering pipeline (`src/pipelines/feature_pipeline/build_features.py`):
    - Función principal `create_preprocessor() -> ColumnTransformer` que retorna un preprocessor unfitted
    - 3 funciones privadas para construcción de sub-pipelines especializados:
        - `_build_numeric_pipeline()`: SimpleImputer(median) para age, max_hr, old_peak
        - `_build_categorical_pipeline()`: SimpleImputer(most_frequent) + OneHotEncoder para chest_pain, sex
        - `_build_ordinal_pipeline()`: SimpleImputer(most_frequent) + OrdinalEncoder para thal, slope, ca, exang
    - Constantes de feature groups (NUMERIC_FEATURES, CATEGORICAL_FEATURES, ORDINAL_FEATURES)
    - Type hints completos y docstrings en estilo Google

- Suite completa de tests para feature engineering pipeline (`tests/pipelines/feature_pipeline/`):
    - 5 pruebas pytest en clase TestCreatePreprocessor
    - Fixture `dummy_df` en `conftest.py` con datos mínimos para pruebas
    - Cobertura de: instancia correcta, número de transformers, nombres de transformers, fit/transform sin excepciones, forma de output correcta
    - Constantes de test (EXPECTED_NUM_TRANSFORMERS, EXPECTED_TRANSFORMER_NAMES, EXPECTED_NUM_FEATURES) para evitar magic numbers

- Notebook prototipo de feature engineering (`notebooks/4-feat_eng/01-feature_engineering_pipeline.ipynb`):
    - Pipeline completo de: carga → selección → limpieza → análisis → split → fit/transform
    - Documentación de EDA a producción workflow
    - 15 características ingenieriles generadas (3 numéricas + 2 one-hot categóricas + 2 one-hot sexo + 4 ordinales)

### Changed

- Refactorización del pipeline de feature engineering:
    - Extracción de lógica del notebook a módulos reutilizables en `src/pipelines/feature_pipeline/`
    - Aplicación del Single Responsibility Principle: funciones puras con inputs/outputs claros
    - Mejora de testabilidad y reutilización del código en múltiples contextos
    - Estructura modular compatible con FTI pipeline pattern (Feature/Training/Inference)

- Actualización de dependencias del proyecto:
    - Agregación de scikit-learn >=1.5.0 como dependencia principal
    - Optimización de `uv.lock` (removidas dependencias innecesarias de jupyter)
    - Actualización de configuración pre-commit para trabajar correctamente con uv

### Fixed

- Resolución de violaciones de ruff linter en feature pipeline:
    - Eliminación de valores mágicos (3, números en aserciones) reemplazados con constantes nombradas
    - Formateo correcto de código según ruff formatter (line length 100, double quotes)
    - Renombramiento de directorio de tests de `feature-pipeline` a `feature_pipeline` para compatibilidad mypy

- Corrección de errores en `pyproject.toml`:
    - Formateo correcto de sección `authors` usando diccionarios en lugar de strings
    - Actualización de configuración pre-commit para usar dependencias gestionadas por uv

## [1.3.0] - 2026-03-14

### Added

- Módulo de limpieza y validación de datos (`src/data/clean_data.py`):
    - Funciones para limpieza y validación de datos cardíacos
    - Refactorización de lógica del notebook de exploración a módulo reutilizable
    - Manejo avanzado de anomalías y valores no estándar

- Suite de tests unitarios para data cleaning (`tests/data/test_clean_data.py`):
    - Tests exhaustivos para funciones de limpieza y validación

- Módulos de análisis de datos (`src/analysis/`):
    - Análisis univariado de variables
    - Análisis bivariable de relaciones entre variables
    - Análisis multivariado para patrones complejos

### Changed

- Refactorización del notebook de exploración:
    - Extracción de lógica de limpieza a módulo reutilizable
    - Aplicación del Single Responsibility Principle
    - Mejora de testabilidad del código de análisis

### Fixed

- Resolución de violaciones de ruff linter en módulos de análisis y limpieza

## [1.2.0] - 2026-03-13

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
