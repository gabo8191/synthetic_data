"""
Módulo de validación para el proyecto de generación de datos sintéticos.

Este módulo contiene funciones de validación para datos, parámetros y configuraciones,
asegurando que el pipeline funcione correctamente con datos válidos.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path
import logging

from src.config import config, ValidationConfig


class ValidationError(Exception):
    """Excepción personalizada para errores de validación."""

    pass


def validate_file_path(
    file_path: str, required_extensions: Optional[List[str]] = None
) -> None:
    """
    Valida que un archivo existe y tiene la extensión correcta.

    Args:
        file_path: Ruta del archivo
        required_extensions: Extensiones permitidas (por defecto: .csv, .xlsx, .json)

    Raises:
        ValidationError: Si el archivo no es válido
    """
    if required_extensions is None:
        required_extensions = [".csv", ".xlsx", ".xls", ".json"]

    path = Path(file_path)

    if not path.exists():
        raise ValidationError(f"Archivo no encontrado: {file_path}")

    if not path.is_file():
        raise ValidationError(f"La ruta no es un archivo: {file_path}")

    if path.suffix.lower() not in required_extensions:
        raise ValidationError(
            f"Extensión no soportada: {path.suffix}. "
            f"Extensiones permitidas: {required_extensions}"
        )


def validate_dataframe_basic(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Validación básica de un DataFrame.

    Args:
        df: DataFrame a validar
        name: Nombre descriptivo para mensajes de error

    Raises:
        ValidationError: Si el DataFrame no es válido
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"{name} debe ser un pandas DataFrame")

    if df.empty:
        raise ValidationError(f"{name} está vacío")

    if len(df) < ValidationConfig.MIN_SAMPLES_FOR_ANALYSIS:
        raise ValidationError(
            f"{name} tiene muy pocas muestras: {len(df)} < {ValidationConfig.MIN_SAMPLES_FOR_ANALYSIS}"
        )


def validate_target_column(df: pd.DataFrame, target_column: str) -> None:
    """
    Valida que la columna objetivo existe y es válida.

    Args:
        df: DataFrame a validar
        target_column: Nombre de la columna objetivo

    Raises:
        ValidationError: Si la columna objetivo no es válida
    """
    if target_column not in df.columns:
        # Intentar variantes comunes
        candidates = {c.lower(): c for c in df.columns}
        if target_column.lower() in candidates:
            logging.warning(
                f"Columna objetivo '{target_column}' no encontrada, usando '{candidates[target_column.lower()]}'"
            )
        else:
            raise ValidationError(
                f"Columna objetivo '{target_column}' no encontrada. "
                f"Columnas disponibles: {list(df.columns)}"
            )

    # Validar que no tenga demasiados valores faltantes
    missing_percentage = df[target_column].isnull().sum() / len(df)
    if missing_percentage > ValidationConfig.MAX_MISSING_PERCENTAGE:
        raise ValidationError(
            f"Columna objetivo '{target_column}' tiene demasiados valores faltantes: "
            f"{missing_percentage:.2%} > {ValidationConfig.MAX_MISSING_PERCENTAGE:.2%}"
        )


def validate_gan_parameters(
    epochs: int, batch_size: int, latent_dim: int, learning_rate: float, cantidad: int
) -> None:
    """
    Valida parámetros del GAN.

    Args:
        epochs: Número de épocas
        batch_size: Tamaño del batch
        latent_dim: Dimensión latente
        learning_rate: Tasa de aprendizaje
        cantidad: Cantidad de muestras a generar

    Raises:
        ValidationError: Si algún parámetro no es válido
    """
    if epochs <= 0:
        raise ValidationError(f"Épocas debe ser positivo: {epochs}")

    if batch_size <= 0:
        raise ValidationError(f"Batch size debe ser positivo: {batch_size}")

    if latent_dim <= 0:
        raise ValidationError(f"Dimensión latente debe ser positiva: {latent_dim}")

    if not 0 < learning_rate < 1:
        raise ValidationError(f"Learning rate debe estar entre 0 y 1: {learning_rate}")

    if cantidad <= 0:
        raise ValidationError(f"Cantidad debe ser positiva: {cantidad}")


def validate_data_split_parameters(
    test_size: float, random_state: Optional[int] = None
) -> None:
    """
    Valida parámetros para división de datos.

    Args:
        test_size: Proporción de datos de prueba
        random_state: Semilla aleatoria

    Raises:
        ValidationError: Si algún parámetro no es válido
    """
    if not 0 < test_size < 1:
        raise ValidationError(f"Test size debe estar entre 0 y 1: {test_size}")

    if random_state is not None and random_state < 0:
        raise ValidationError(f"Random state debe ser no negativo: {random_state}")


def validate_correlation_analysis(df: pd.DataFrame) -> None:
    """
    Valida que un DataFrame es adecuado para análisis de correlación.

    Args:
        df: DataFrame a validar

    Raises:
        ValidationError: Si el DataFrame no es adecuado para correlación
    """
    validate_dataframe_basic(df, "DataFrame para correlación")

    if len(df) < ValidationConfig.MIN_CORRELATION_SAMPLES:
        raise ValidationError(
            f"DataFrame tiene muy pocas muestras para correlación: "
            f"{len(df)} < {ValidationConfig.MIN_CORRELATION_SAMPLES}"
        )

    # Verificar que hay al menos 2 columnas numéricas
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2:
        raise ValidationError(
            f"DataFrame debe tener al menos 2 columnas numéricas para correlación. "
            f"Encontradas: {len(numeric_cols)}"
        )


def validate_balancing_parameters(
    df: pd.DataFrame, target_column: str, method: str
) -> None:
    """
    Valida parámetros para balanceo de datos.

    Args:
        df: DataFrame a balancear
        target_column: Columna objetivo
        method: Método de balanceo

    Raises:
        ValidationError: Si los parámetros no son válidos
    """
    validate_dataframe_basic(df, "DataFrame para balanceo")
    validate_target_column(df, target_column)

    if method not in ["RandomUnderSampler", "SMOTE", "ADASYN"]:
        raise ValidationError(f"Método de balanceo no soportado: {method}")

    # Verificar que hay al menos 2 clases
    unique_classes = df[target_column].nunique()
    if unique_classes < 2:
        raise ValidationError(
            f"Columna objetivo debe tener al menos 2 clases únicas. "
            f"Encontradas: {unique_classes}"
        )


def validate_comparison_parameters(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    max_categories: int,
    drift_threshold: float,
) -> None:
    """
    Valida parámetros para comparación de datos.

    Args:
        real_df: DataFrame con datos reales
        synthetic_df: DataFrame con datos sintéticos
        max_categories: Máximo número de categorías
        drift_threshold: Umbral de drift

    Raises:
        ValidationError: Si los parámetros no son válidos
    """
    validate_dataframe_basic(real_df, "DataFrame real")
    validate_dataframe_basic(synthetic_df, "DataFrame sintético")

    if max_categories <= 0:
        raise ValidationError(f"Max categories debe ser positivo: {max_categories}")

    if not 0 < drift_threshold < 1:
        raise ValidationError(
            f"Drift threshold debe estar entre 0 y 1: {drift_threshold}"
        )


def validate_directory_path(path: str, create_if_missing: bool = True) -> None:
    """
    Valida que un directorio existe o puede ser creado.

    Args:
        path: Ruta del directorio
        create_if_missing: Si crear el directorio si no existe

    Raises:
        ValidationError: Si el directorio no es válido
    """
    dir_path = Path(path)

    if not dir_path.exists():
        if create_if_missing:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValidationError(f"No se pudo crear directorio {path}: {e}")
        else:
            raise ValidationError(f"Directorio no existe: {path}")

    if not dir_path.is_dir():
        raise ValidationError(f"La ruta no es un directorio: {path}")


def validate_configuration() -> None:
    """
    Valida la configuración global del proyecto.

    Raises:
        ValidationError: Si la configuración no es válida
    """
    # Validar rutas
    validate_file_path(config.paths.INPUT_FILE)
    validate_directory_path(config.paths.OUTPUT_DIR)
    validate_directory_path(config.paths.GRAPHICS_DIR)
    validate_directory_path(config.paths.REPORTS_DIR)

    # Validar parámetros de datos
    validate_data_split_parameters(config.data.TEST_SIZE, config.data.RANDOM_STATE)

    # Validar parámetros del GAN
    validate_gan_parameters(
        config.gan.EPOCHS,
        config.gan.BATCH_SIZE,
        config.gan.LATENT_DIM,
        config.gan.LEARNING_RATE,
        config.gan.SYNTHETIC_SAMPLES,
    )

    # Validar parámetros de análisis
    if config.analysis.MAX_CATEGORIES <= 0:
        raise ValidationError(
            f"MAX_CATEGORIES debe ser positivo: {config.analysis.MAX_CATEGORIES}"
        )

    if not 0 < config.analysis.DRIFT_WARN_THRESHOLD < 1:
        raise ValidationError(
            f"DRIFT_WARN_THRESHOLD debe estar entre 0 y 1: {config.analysis.DRIFT_WARN_THRESHOLD}"
        )


def validate_data_quality(df: pd.DataFrame, name: str = "DataFrame") -> Dict[str, Any]:
    """
    Realiza una validación completa de calidad de datos.

    Args:
        df: DataFrame a validar
        name: Nombre descriptivo

    Returns:
        Diccionario con reporte de calidad

    Raises:
        ValidationError: Si los datos no pasan la validación
    """
    validate_dataframe_basic(df, name)

    quality_report = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df)).to_dict(),
        "duplicates": df.duplicated().sum(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "data_types": df.dtypes.astype(str).to_dict(),
        "numeric_columns": df.select_dtypes(include=np.number).columns.tolist(),
        "categorical_columns": df.select_dtypes(include="object").columns.tolist(),
    }

    # Verificar columnas con demasiados valores faltantes
    high_missing_cols = [
        col
        for col, pct in quality_report["missing_percentage"].items()
        if pct > ValidationConfig.MAX_MISSING_PERCENTAGE
    ]

    if high_missing_cols:
        logging.warning(
            f"{name}: Columnas con muchos valores faltantes: {high_missing_cols}"
        )

    return quality_report


# Exportar funciones principales
__all__ = [
    "ValidationError",
    "validate_file_path",
    "validate_dataframe_basic",
    "validate_target_column",
    "validate_gan_parameters",
    "validate_data_split_parameters",
    "validate_correlation_analysis",
    "validate_balancing_parameters",
    "validate_comparison_parameters",
    "validate_directory_path",
    "validate_configuration",
    "validate_data_quality",
]
