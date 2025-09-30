"""
Configuración centralizada para el proyecto de generación de datos sintéticos.

Este módulo contiene todas las constantes, configuraciones y parámetros
utilizados en el proyecto, centralizando la configuración para facilitar
el mantenimiento y la modificación.
"""

from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class PathsConfig:
    """Configuración de rutas del proyecto."""

    # Rutas de entrada
    INPUT_FILE: str = "data/Titanic-Dataset.csv"

    # Rutas de salida
    OUTPUT_DIR: str = "results/data/"
    GRAPHICS_DIR: str = "results/graphics/"
    REPORTS_DIR: str = "results/reports/"

    # Subdirectorios
    COMPARE_DIR: str = "results/graphics/compare/"

    def __post_init__(self):
        """Crear directorios si no existen."""
        for path in [
            self.OUTPUT_DIR,
            self.GRAPHICS_DIR,
            self.REPORTS_DIR,
            self.COMPARE_DIR,
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Configuración relacionada con el procesamiento de datos."""

    # Columna objetivo
    TARGET_COLUMN: str = "survived"

    # Configuración de limpieza
    MISSING_VALUE_MARKERS: Optional[List[str]] = None
    ID_LIKE_COLUMNS: Optional[List[str]] = None

    # Configuración de división de datos
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    # Configuración de balanceo
    BALANCING_METHOD: str = "RandomUnderSampler"
    BALANCING_STRATEGY: str = "all"

    def __post_init__(self):
        if self.MISSING_VALUE_MARKERS is None:
            self.MISSING_VALUE_MARKERS = ["?", "", "NA", "NaN"]

        if self.ID_LIKE_COLUMNS is None:
            self.ID_LIKE_COLUMNS = ["passengerid", "name", "ticket", "cabin"]


@dataclass
class GANConfig:
    """Configuración del modelo GAN."""

    # Parámetros de generación
    SYNTHETIC_SAMPLES: int = 1000
    EPOCHS: int = 1400
    BATCH_SIZE: int = 64
    LATENT_DIM: int = 100
    LEARNING_RATE: float = 0.0002

    # Configuración de arquitectura
    GENERATOR_HIDDEN_SIZE: int = 128
    DISCRIMINATOR_HIDDEN_SIZE: int = 128

    # Configuración de entrenamiento
    STRATIFIED_TRAINING: bool = True
    LOG_INTERVAL: int = 100

    # Configuración de normalización
    NORMALIZATION_RANGE: tuple = (-1, 1)
    EPSILON: float = 1e-8


@dataclass
class AnalysisConfig:
    """Configuración para análisis y comparación."""

    # Configuración de gráficos
    FIGURE_SIZE: tuple = (10, 6)
    DPI: int = 300
    BBOX_INCHES: str = "tight"

    # Configuración de comparación
    MAX_CATEGORIES: int = 30
    DRIFT_WARN_THRESHOLD: float = 0.3
    DRIFT_HIGH_THRESHOLD: float = 0.3
    CATEGORICAL_DRIFT_THRESHOLD: float = 0.2

    # Configuración de histogramas
    HISTOGRAM_BINS: int = 30

    # Configuración de correlación
    STRONG_CORRELATION_THRESHOLD: float = 0.5
    VERY_STRONG_CORRELATION_THRESHOLD: float = 0.7


@dataclass
class LoggingConfig:
    """Configuración de logging."""

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    ENCODING: str = "utf-8"


@dataclass
class ProjectConfig:
    """Configuración principal del proyecto."""

    paths: PathsConfig
    data: DataConfig
    gan: GANConfig
    analysis: AnalysisConfig
    logging: LoggingConfig

    def __init__(
        self,
        paths: Optional[PathsConfig] = None,
        data: Optional[DataConfig] = None,
        gan: Optional[GANConfig] = None,
        analysis: Optional[AnalysisConfig] = None,
        logging: Optional[LoggingConfig] = None,
    ):
        self.paths = paths or PathsConfig()
        self.data = data or DataConfig()
        self.gan = gan or GANConfig()
        self.analysis = analysis or AnalysisConfig()
        self.logging = logging or LoggingConfig()


# Instancia global de configuración
config = ProjectConfig()


# Constantes de colores para gráficos
class Colors:
    """Paleta de colores unificada para gráficos."""

    REAL = "#4e79a7"
    SYNTHETIC = "#e15759"
    BALANCED = "#59a14f"
    NEUTRAL = "#9c755f"

    # Colores para correlación
    POSITIVE_CORRELATION = "blue"
    NEGATIVE_CORRELATION = "red"


# Configuraciones específicas por tipo de dato
class DataTypes:
    """Configuraciones específicas por tipo de dato."""

    NUMERIC_TYPES = ["float64", "int64", "float32", "int32"]
    CATEGORICAL_TYPES = ["object", "category"]
    BINARY_THRESHOLD = {0, 1}  # Para identificar variables binarias


# Configuraciones de validación
class ValidationConfig:
    """Configuraciones para validación de datos."""

    MIN_SAMPLES_FOR_ANALYSIS = 10
    MAX_MISSING_PERCENTAGE = 0.5
    MIN_CORRELATION_SAMPLES = 30


# Exportar configuración principal
__all__ = [
    "config",
    "PathsConfig",
    "DataConfig",
    "GANConfig",
    "AnalysisConfig",
    "LoggingConfig",
    "ProjectConfig",
    "Colors",
    "DataTypes",
    "ValidationConfig",
]
