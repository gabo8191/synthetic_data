"""
Módulo orquestador de análisis de datos.

Este módulo coordina el análisis completo de datos, incluyendo balanceo,
visualización y comparación entre datos reales y sintéticos.
"""

import pandas as pd
import os
import logging
from typing import Dict, Tuple

from src.config import config
from src.analysis.balancing import (
    analyze_class_distribution,
    balance_dataset,
    validate_balancing_result,
)
from src.analysis.plotting import (
    plot_class_distribution,
    plot_correlation_matrix,
    plot_survived_correlation,
)
from src.analysis.comparison import compare_real_vs_synthetic
from src.analysis.metrics import (
    generate_correlation_report,
    calculate_correlation_matrix,
)


def analyze_data(
    df: pd.DataFrame, target_column: str, graphics_dir: str = "results/graphics/"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Realiza análisis básico y balanceo por undersampling.

    Args:
        df: DataFrame con los datos
        target_column: Nombre de la columna objetivo
        graphics_dir: Directorio donde guardar gráficos

    Returns:
        Tupla con (DataFrame balanceado, reporte de análisis)
    """
    logger = logging.getLogger("analyzer")
    report: Dict = {}
    os.makedirs(graphics_dir, exist_ok=True)

    try:
        logger.info("Iniciando análisis de datos...")

        # 1. Análisis de distribución de clases
        logger.info("Analizando distribución de clases...")
        class_analysis = analyze_class_distribution(df, target_column)
        report["class_analysis"] = class_analysis

        # 2. Generar gráfico de distribución original
        logger.info("Generando gráfico de distribución original...")
        plot_class_distribution(
            df[target_column], os.path.join(graphics_dir, "original_dist.png")
        )

        # 3. Balancear datos
        logger.info("Balanceando datos...")
        balanced_df, balancing_report = balance_dataset(
            df,
            target_column,
            method="RandomUnderSampler",
            random_state=config.data.RANDOM_STATE,
        )
        report["balancing"] = balancing_report

        # 4. Validar resultado del balanceo
        validation_report = validate_balancing_result(df, balanced_df, target_column)
        report["validation"] = validation_report

        # 5. Generar gráfico de distribución balanceada
        logger.info("Generando gráfico de distribución balanceada...")
        plot_class_distribution(
            balanced_df[target_column], os.path.join(graphics_dir, "balanced_dist.png")
        )

        logger.info(f"Análisis completado. Muestras: {len(df)} -> {len(balanced_df)}")
        return balanced_df, report

    except Exception as e:
        logger.error(f"Error en análisis de datos: {str(e)}")
        report["error"] = str(e)
        raise RuntimeError(f"Error en analyzer: {str(e)}")


def generate_correlation_analysis(
    df: pd.DataFrame, graphics_dir: str = "results/graphics/"
) -> Dict:
    """
    Genera análisis completo de correlación.

    Args:
        df: DataFrame con los datos
        graphics_dir: Directorio donde guardar gráficos

    Returns:
        Diccionario con reporte de correlación
    """
    logger = logging.getLogger("analyzer")

    try:
        logger.info("Iniciando análisis de correlación...")

        # 1. Generar reporte de correlación
        correlation_report = generate_correlation_report(df, graphics_dir)

        # 2. Calcular matriz de correlación para gráficos
        correlation_matrix, _ = calculate_correlation_matrix(df)

        # 3. Generar gráfico de matriz de correlación
        logger.info("Generando gráfico de matriz de correlación...")
        plot_correlation_matrix(
            correlation_matrix,
            os.path.join(graphics_dir, "correlation_matrix.png"),
            "Matriz de Correlación - Dataset Titanic",
        )

        # 4. Generar gráfico de correlación con supervivencia si existe
        if "survived" in correlation_matrix.columns:
            logger.info("Generando gráfico de correlación con supervivencia...")
            survived_corr = (
                correlation_matrix["survived"]
                .drop("survived")
                .sort_values(key=abs, ascending=False)
            )
            plot_survived_correlation(
                survived_corr,
                os.path.join(graphics_dir, "survived_correlation.png"),
                "Correlación de Variables con la Supervivencia",
            )

        logger.info("Análisis de correlación completado")
        return correlation_report

    except Exception as e:
        logger.error(f"Error en análisis de correlación: {str(e)}")
        raise RuntimeError(f"Error en análisis de correlación: {str(e)}")


# Re-exportar funciones de otros módulos para compatibilidad
from src.analysis.plotting import plot_class_distribution
from src.analysis.comparison import compare_real_vs_synthetic


__all__ = [
    "analyze_data",
    "generate_correlation_analysis",
    "compare_real_vs_synthetic",
    "plot_class_distribution",
]
