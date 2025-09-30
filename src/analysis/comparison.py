"""
Módulo de comparación entre datos reales y sintéticos.

Este módulo contiene funciones para comparar datos reales y sintéticos,
generando reportes detallados y visualizaciones comparativas.
"""

import pandas as pd
import os
import logging
from typing import Dict, List, Optional

from src.analysis.plotting import (
    plot_numeric_comparison,
    plot_categorical_comparison,
    plot_target_comparison,
)
from src.analysis.metrics import (
    calculate_drift_metrics,
    calculate_categorical_drift,
    calculate_discrimination_percentage,
)


def identify_common_numeric_columns(
    real_df: pd.DataFrame, synthetic_df: pd.DataFrame, skip_id_like: bool = True
) -> List[str]:
    """
    Identifica columnas numéricas comunes entre datasets reales y sintéticos.

    Args:
        real_df: DataFrame con datos reales
        synthetic_df: DataFrame con datos sintéticos
        skip_id_like: Si omitir columnas tipo ID

    Returns:
        Lista de nombres de columnas numéricas comunes
    """
    logger = logging.getLogger("comparison")
    common_numeric: List[str] = []

    for col in real_df.columns:
        if col not in synthetic_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(real_df[col]):
            continue

        if skip_id_like:
            name_l = str(col).lower()
            id_name_patterns = ["id", "ticket", "name", "cabin", "passengerid"]
            if any(p in name_l for p in id_name_patterns):
                logger.info(f"Omitiendo columna numérica tipo ID: {col}")
                continue

            nunique_ratio = float(real_df[col].nunique()) / max(
                1.0, float(len(real_df))
            )
            if nunique_ratio > 0.9:
                logger.info(f"Omitiendo columna numérica de alta unicidad: {col}")
                continue

        common_numeric.append(col)

    return common_numeric


def compare_numeric_variables(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    numeric_columns: List[str],
    compare_dir: str,
    drift_warn_threshold: float = 0.3,
) -> Dict:
    """
    Compara variables numéricas entre datos reales y sintéticos.

    Args:
        real_df: DataFrame con datos reales
        synthetic_df: DataFrame con datos sintéticos
        numeric_columns: Lista de columnas numéricas a comparar
        compare_dir: Directorio donde guardar gráficos
        drift_warn_threshold: Umbral para warnings de drift

    Returns:
        Diccionario con reporte de comparación numérica
    """
    logger = logging.getLogger("comparison")
    comparison_report: Dict = {}

    for col in numeric_columns:
        # Generar gráfico comparativo
        real_vals = real_df[col].astype(float)
        syn_vals = synthetic_df[col].astype(float)

        plot_numeric_comparison(
            real_vals, syn_vals, col, os.path.join(compare_dir, f"compare_{col}.png")
        )

        # Calcular métricas de drift
        drift_metrics = calculate_drift_metrics(
            real_vals, syn_vals, drift_warn_threshold
        )

        comp_entry = {
            "type": "numeric",
            "real_mean": drift_metrics["real_mean"],
            "synthetic_mean": drift_metrics["synthetic_mean"],
            "real_std": drift_metrics["real_std"],
            "synthetic_std": drift_metrics["synthetic_std"],
            "rel_mean_diff": drift_metrics["rel_mean_diff"],
            "std_rel_diff": drift_metrics["std_rel_diff"],
            "high_drift": drift_metrics["high_drift"],
        }

        # Log warning si hay alto drift
        if drift_metrics["high_drift"]:
            logger.warning(
                f"Alto drift en '{col}': rel_mean_diff={drift_metrics['rel_mean_diff']:.2f}, "
                f"std_rel_diff={drift_metrics['std_rel_diff']:.2f}"
            )

        comparison_report[col] = comp_entry

    return comparison_report


def compare_categorical_variables(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    categorical_columns: List[str],
    compare_dir: str,
    max_categories: int = 30,
) -> Dict:
    """
    Compara variables categóricas entre datos reales y sintéticos.

    Args:
        real_df: DataFrame con datos reales
        synthetic_df: DataFrame con datos sintéticos
        categorical_columns: Lista de columnas categóricas a comparar
        compare_dir: Directorio donde guardar gráficos
        max_categories: Máximo número de categorías a mostrar

    Returns:
        Diccionario con reporte de comparación categórica
    """
    logger = logging.getLogger("comparison")
    comparison_report: Dict = {}

    for col in categorical_columns:
        real_counts = real_df[col].astype(str).value_counts(normalize=True)

        # Identificar si en sintético existe one-hot de este col (prefijo "col_")
        prefix = f"{col}_"
        oh_cols = [c for c in synthetic_df.columns if c.startswith(prefix)]

        if len(oh_cols) == 0:
            # Si no hay one-hot, intentar que exista la misma columna categórica
            if col in synthetic_df.columns:
                syn_counts = synthetic_df[col].astype(str).value_counts(normalize=True)
            else:
                # No hay forma de comparar; continuar
                logger.info(
                    f"Omitiendo categórica sin correspondencia en sintético: {col}"
                )
                continue
        else:
            # Mapear one-hot a categorías
            syn_counts = (
                synthetic_df[oh_cols].mean().rename(lambda x: x.replace(prefix, ""))
            )

        # Verificar número de categorías
        all_cats = sorted(
            set(real_counts.index.tolist()) | set(syn_counts.index.tolist())
        )
        if len(all_cats) > max_categories:
            logger.info(
                f"Omitiendo categórica con muchas categorías ({len(all_cats)}): {col}"
            )
            continue

        # Generar gráfico comparativo
        plot_categorical_comparison(
            real_counts,
            syn_counts,
            col,
            os.path.join(compare_dir, f"compare_{col}.png"),
        )

        # Calcular métricas de drift categórico
        real_freq_dict = {k: float(v) for k, v in real_counts.items()}
        syn_freq_dict = {k: float(v) for k, v in syn_counts.items()}

        drift_metrics = calculate_categorical_drift(real_freq_dict, syn_freq_dict)

        comparison_report[col] = {
            "type": "categorical",
            "categories": all_cats,
            "real_freq": real_freq_dict,
            "synthetic_freq": syn_freq_dict,
            "max_difference": drift_metrics["max_difference"],
            "high_drift": drift_metrics["high_drift"],
        }

    return comparison_report


def compare_target_variable(
    real_target: pd.Series,
    synthetic_target: pd.Series,
    target_column: str,
    compare_dir: str,
) -> Dict:
    """
    Compara la variable objetivo entre datos reales y sintéticos.

    Args:
        real_target: Valores reales del target
        synthetic_target: Valores sintéticos del target
        target_column: Nombre de la columna objetivo
        compare_dir: Directorio donde guardar gráficos

    Returns:
        Diccionario con reporte de comparación del target
    """
    logger = logging.getLogger("comparison")

    try:
        # Generar gráfico comparativo
        plot_target_comparison(
            real_target,
            synthetic_target,
            target_column,
            os.path.join(compare_dir, f"compare_{target_column}.png"),
        )

        # Calcular frecuencias
        real_t = real_target.astype(str).value_counts(normalize=True)
        syn_t = synthetic_target.astype(str).value_counts(normalize=True)
        all_t = sorted(set(real_t.index.tolist()) | set(syn_t.index.tolist()))
        real_t = real_t.reindex(all_t, fill_value=0.0)
        syn_t = syn_t.reindex(all_t, fill_value=0.0)

        # Calcular métricas de drift
        real_freq_dict = {k: float(v) for k, v in real_t.items()}
        syn_freq_dict = {k: float(v) for k, v in syn_t.items()}

        drift_metrics = calculate_categorical_drift(real_freq_dict, syn_freq_dict)

        return {
            "type": "target",
            "categories": all_t,
            "real_freq": real_freq_dict,
            "synthetic_freq": syn_freq_dict,
            "max_difference": drift_metrics["max_difference"],
            "high_drift": drift_metrics["high_drift"],
        }

    except Exception as e:
        logger.warning(f"No se pudo comparar target '{target_column}': {e}")
        return {}


def compare_real_vs_synthetic(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    graphics_dir: str = "results/graphics/",
    *,
    target_column: Optional[str] = None,
    real_target: Optional[pd.Series] = None,
    synthetic_target: Optional[pd.Series] = None,
    max_categories: int = 30,
    skip_id_like: bool = True,
    drift_warn_threshold: float = 0.3,
) -> Dict:
    """
    Función principal para comparar datos reales y sintéticos.

    Args:
        real_df: DataFrame con datos reales
        synthetic_df: DataFrame con datos sintéticos
        graphics_dir: Directorio base para gráficos
        target_column: Nombre de la columna objetivo
        real_target: Valores reales del target
        synthetic_target: Valores sintéticos del target
        max_categories: Máximo número de categorías
        skip_id_like: Si omitir columnas tipo ID
        drift_warn_threshold: Umbral para warnings de drift

    Returns:
        Diccionario con reporte completo de comparación
    """
    compare_dir = os.path.join(graphics_dir, "compare")
    os.makedirs(compare_dir, exist_ok=True)

    logger = logging.getLogger("comparison")
    logger.info("Iniciando comparación de datos reales vs sintéticos...")

    # 1. Identificar y comparar variables numéricas
    numeric_columns = identify_common_numeric_columns(
        real_df, synthetic_df, skip_id_like
    )
    logger.info(f"Comparando {len(numeric_columns)} variables numéricas")

    numeric_report = compare_numeric_variables(
        real_df, synthetic_df, numeric_columns, compare_dir, drift_warn_threshold
    )

    # 2. Identificar y comparar variables categóricas
    categorical_columns = list(real_df.select_dtypes(include="object").columns)
    logger.info(f"Comparando {len(categorical_columns)} variables categóricas")

    categorical_report = compare_categorical_variables(
        real_df, synthetic_df, categorical_columns, compare_dir, max_categories
    )

    # 3. Comparar variable objetivo si se proporciona
    target_report = {}
    if target_column and real_target is not None and synthetic_target is not None:
        logger.info(f"Comparando variable objetivo: {target_column}")
        target_report = compare_target_variable(
            real_target, synthetic_target, target_column, compare_dir
        )

    # 4. Combinar reportes (solo incluir entradas válidas)
    comparison_report = {}

    # Agregar reportes numéricos
    for key, value in numeric_report.items():
        if isinstance(value, dict):
            comparison_report[key] = value

    # Agregar reportes categóricos
    for key, value in categorical_report.items():
        if isinstance(value, dict):
            comparison_report[key] = value

    # Agregar reporte de target si es válido
    if target_report and isinstance(target_report, dict):
        for key, value in target_report.items():
            if isinstance(value, dict):
                comparison_report[key] = value

    # 5. Calcular métricas de discriminación
    discrimination_metrics = calculate_discrimination_percentage(comparison_report)
    comparison_report["_metadata"] = discrimination_metrics

    logger.info(
        f"Comparación completada. Porcentaje discriminador: {discrimination_metrics['discrimination_percentage']}%"
    )

    return comparison_report


__all__ = [
    "identify_common_numeric_columns",
    "compare_numeric_variables",
    "compare_categorical_variables",
    "compare_target_variable",
    "compare_real_vs_synthetic",
]
