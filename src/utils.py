"""
Utilidades comunes para el proyecto de generación de datos sintéticos.

Este módulo contiene funciones de utilidad que son utilizadas por múltiples
módulos del proyecto, centralizando la lógica común y evitando duplicación.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from src.config import config, Colors


def setup_logging(reports_dir: str) -> logging.Logger:
    """
    Configura el sistema de logging del proyecto.

    Args:
        reports_dir: Directorio donde guardar los logs

    Returns:
        Logger configurado
    """
    os.makedirs(reports_dir, exist_ok=True)
    log_path = os.path.join(reports_dir, "pipeline.log")

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(getattr(logging, config.logging.LOG_LEVEL))

    formatter = logging.Formatter(
        fmt=config.logging.LOG_FORMAT,
        datefmt=config.logging.DATE_FORMAT,
    )

    # Handler para consola
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, config.logging.LOG_LEVEL))
    ch.setFormatter(formatter)

    # Handler para archivo
    fh = logging.FileHandler(log_path, encoding=config.logging.ENCODING)
    fh.setLevel(getattr(logging, config.logging.LOG_LEVEL))
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logging.getLogger("pipeline")


def save_json_report(data: Dict[str, Any], filepath: str) -> None:
    """
    Guarda un diccionario como archivo JSON con manejo de errores.

    Args:
        data: Diccionario a guardar
        filepath: Ruta del archivo
    """
    try:
        # Convertir objetos no serializables
        serializable_data = _make_json_serializable(data)

        with open(filepath, "w", encoding=config.logging.ENCODING) as f:
            json.dump(serializable_data, f, indent=4, ensure_ascii=False)

    except Exception as e:
        logging.error(f"Error guardando reporte JSON en {filepath}: {e}")
        raise


def _make_json_serializable(obj: Any) -> Any:
    """
    Convierte objetos no serializables a JSON a formatos serializables.

    Args:
        obj: Objeto a convertir

    Returns:
        Objeto serializable
    """
    if hasattr(obj, "tolist"):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_dataframe(df: pd.DataFrame, filepath: str, **kwargs: Any) -> None:
    """
    Guarda un DataFrame con manejo de errores.

    Args:
        df: DataFrame a guardar
        filepath: Ruta del archivo
        **kwargs: Argumentos adicionales para pandas.to_csv()
    """
    try:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Valores por defecto
        default_kwargs = {"index": False}
        default_kwargs.update(kwargs)

        # Usar solo argumentos válidos para to_csv
        df.to_csv(filepath, index=default_kwargs.get("index", False))

    except Exception as e:
        logging.error(f"Error guardando DataFrame en {filepath}: {e}")
        raise


def identify_continuous_columns(df: pd.DataFrame) -> List[str]:
    """
    Identifica columnas numéricas continuas (no binarias).

    Args:
        df: DataFrame a analizar

    Returns:
        Lista de nombres de columnas continuas
    """
    continuous_cols = []
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        unique_values = set(pd.Series(df[col].dropna().unique()).tolist())
        if not unique_values.issubset({0, 1}):
            continuous_cols.append(col)

    return continuous_cols


def identify_id_like_columns(
    df: pd.DataFrame, id_patterns: Optional[List[str]] = None
) -> List[str]:
    """
    Identifica columnas que parecen identificadores.

    Args:
        df: DataFrame a analizar
        id_patterns: Patrones a buscar (por defecto usa config.data.ID_LIKE_COLUMNS)

    Returns:
        Lista de nombres de columnas tipo ID
    """
    if id_patterns is None:
        id_patterns = config.data.ID_LIKE_COLUMNS or []

    id_like_cols = []

    for col in df.columns:
        col_lower = str(col).lower()
        if any(pattern.lower() in col_lower for pattern in id_patterns):
            id_like_cols.append(col)
        else:
            # Verificar alta unicidad
            nunique_ratio = float(df[col].nunique()) / max(1.0, float(len(df)))
            if nunique_ratio > 0.9:
                id_like_cols.append(col)

    return id_like_cols


def validate_dataframe(df: pd.DataFrame, min_samples: int = 10) -> bool:
    """
    Valida que un DataFrame cumpla con los requisitos mínimos.

    Args:
        df: DataFrame a validar
        min_samples: Número mínimo de muestras

    Returns:
        True si es válido, False en caso contrario
    """
    if len(df) < min_samples:
        logging.warning(
            f"DataFrame tiene muy pocas muestras: {len(df)} < {min_samples}"
        )
        return False

    if df.empty:
        logging.error("DataFrame está vacío")
        return False

    return True


def get_color_for_label(label: str) -> str:
    """
    Obtiene el color apropiado para una etiqueta en gráficos.

    Args:
        label: Etiqueta del gráfico

    Returns:
        Color en formato hexadecimal
    """
    label_lower = label.lower()

    if "real" in label_lower:
        return Colors.REAL
    elif "synthetic" in label_lower:
        return Colors.SYNTHETIC
    elif "balanced" in label_lower:
        return Colors.BALANCED
    else:
        return Colors.NEUTRAL


def calculate_drift_metrics(
    real_values: pd.Series, synthetic_values: pd.Series, threshold: float = 0.3
) -> Dict[str, float]:
    """
    Calcula métricas de drift entre datos reales y sintéticos.

    Args:
        real_values: Valores reales
        synthetic_values: Valores sintéticos
        threshold: Umbral para considerar drift alto

    Returns:
        Diccionario con métricas de drift
    """
    real_mean = float(real_values.mean())
    syn_mean = float(synthetic_values.mean())
    real_std = float(real_values.std()) or 1e-8
    syn_std = float(synthetic_values.std()) or 1e-8

    rel_mean_diff = abs(syn_mean - real_mean) / (abs(real_mean) + 1e-8)
    std_rel_diff = abs(syn_std - real_std) / real_std

    return {
        "real_mean": real_mean,
        "synthetic_mean": syn_mean,
        "real_std": real_std,
        "synthetic_std": syn_std,
        "rel_mean_diff": rel_mean_diff,
        "std_rel_diff": std_rel_diff,
        "high_drift": rel_mean_diff > threshold or std_rel_diff > threshold,
    }


def align_categorical_series(
    series1: pd.Series, series2: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Alinea dos series categóricas para comparación.

    Args:
        series1: Primera serie
        series2: Segunda serie

    Returns:
        Tupla con las series alineadas
    """
    all_categories = sorted(set(series1.index.tolist()) | set(series2.index.tolist()))

    aligned_series1 = series1.reindex(all_categories, fill_value=0.0)
    aligned_series2 = series2.reindex(all_categories, fill_value=0.0)

    return aligned_series1, aligned_series2


def create_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Crea estadísticas resumen de un DataFrame.

    Args:
        df: DataFrame a analizar

    Returns:
        Diccionario con estadísticas resumen
    """
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "memory_usage": df.memory_usage(deep=True).sum(),
        "numeric_columns": df.select_dtypes(include=np.number).columns.tolist(),
        "categorical_columns": df.select_dtypes(include="object").columns.tolist(),
    }


def ensure_directory_exists(path: str) -> None:
    """
    Asegura que un directorio existe, creándolo si es necesario.

    Args:
        path: Ruta del directorio
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def log_dataframe_info(df: pd.DataFrame, name: str, logger: logging.Logger) -> None:
    """
    Registra información básica de un DataFrame.

    Args:
        df: DataFrame a registrar
        name: Nombre descriptivo del DataFrame
        logger: Logger a usar
    """
    logger.info(f"{name}: {df.shape[0]} filas, {df.shape[1]} columnas")

    if hasattr(df, "memory_usage"):
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        logger.info(f"{name} memoria: {memory_mb:.2f} MB")


# Exportar funciones principales
__all__ = [
    "setup_logging",
    "save_json_report",
    "save_dataframe",
    "identify_continuous_columns",
    "identify_id_like_columns",
    "validate_dataframe",
    "get_color_for_label",
    "calculate_drift_metrics",
    "align_categorical_series",
    "create_summary_stats",
    "ensure_directory_exists",
    "log_dataframe_info",
]
