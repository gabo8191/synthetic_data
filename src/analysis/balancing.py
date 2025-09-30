"""
Módulo de balanceo de datos.

Este módulo contiene funciones para el balanceo de datasets desbalanceados
usando diferentes técnicas de muestreo.
"""

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from typing import Dict, Tuple
import logging


def analyze_class_distribution(df: pd.DataFrame, target_column: str) -> Dict:
    """
    Analiza la distribución de clases en el dataset.

    Args:
        df: DataFrame con los datos
        target_column: Nombre de la columna objetivo

    Returns:
        Diccionario con análisis de distribución de clases
    """
    class_dist = df[target_column].value_counts().to_dict()
    minority_class = min(class_dist.items(), key=lambda kv: kv[1])[0]
    imbalance_ratio = float(min(class_dist.values()) / max(class_dist.values()))

    return {
        "original_distribution": class_dist,
        "minority_class": minority_class,
        "imbalance_ratio": imbalance_ratio,
    }


def balance_dataset(
    df: pd.DataFrame,
    target_column: str,
    method: str = "RandomUnderSampler",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Balancea un dataset usando la técnica especificada.

    Args:
        df: DataFrame con los datos
        target_column: Nombre de la columna objetivo
        method: Método de balanceo a usar
        random_state: Semilla aleatoria para reproducibilidad

    Returns:
        Tupla con (DataFrame balanceado, reporte de balanceo)
    """
    logger = logging.getLogger("balancing")

    # Separar características y target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Aplicar balanceo según el método
    if method == "RandomUnderSampler":
        undersampler = RandomUnderSampler(
            sampling_strategy="all", random_state=random_state
        )
        X_res, y_res = undersampler.fit_resample(X, y)

        # Reconstruir DataFrame
        balanced_df = pd.DataFrame(X_res, columns=X.columns)
        balanced_df[target_column] = y_res

        # Crear reporte
        report = {
            "method": "RandomUnderSampler",
            "new_distribution": Counter(y_res),
            "samples_removed": int(len(y) - len(y_res)),
            "original_samples": len(y),
            "balanced_samples": len(y_res),
        }

        logger.info(f"Balanceo completado: {len(y)} -> {len(y_res)} muestras")

    else:
        raise ValueError(f"Método de balanceo no soportado: {method}")

    return balanced_df, report


def validate_balancing_result(
    original_df: pd.DataFrame, balanced_df: pd.DataFrame, target_column: str
) -> Dict:
    """
    Valida el resultado del balanceo.

    Args:
        original_df: DataFrame original
        balanced_df: DataFrame balanceado
        target_column: Nombre de la columna objetivo

    Returns:
        Diccionario con métricas de validación
    """
    original_dist = original_df[target_column].value_counts(normalize=True)
    balanced_dist = balanced_df[target_column].value_counts(normalize=True)

    # Calcular métricas de balanceo
    original_imbalance = max(original_dist.values) / min(original_dist.values)
    balanced_imbalance = max(balanced_dist.values) / min(balanced_dist.values)

    return {
        "original_imbalance_ratio": original_imbalance,
        "balanced_imbalance_ratio": balanced_imbalance,
        "improvement_factor": original_imbalance / balanced_imbalance,
        "is_balanced": balanced_imbalance <= 1.1,  # Tolerancia del 10%
    }


def get_balancing_strategy(df: pd.DataFrame, target_column: str) -> str:
    """
    Recomienda una estrategia de balanceo basada en el dataset.

    Args:
        df: DataFrame con los datos
        target_column: Nombre de la columna objetivo

    Returns:
        Estrategia recomendada
    """
    class_dist = df[target_column].value_counts()
    imbalance_ratio = min(class_dist.values) / max(class_dist.values)

    if imbalance_ratio > 0.3:
        return "RandomUnderSampler"  # Desbalanceo moderado
    elif imbalance_ratio > 0.1:
        return "RandomUnderSampler"  # Desbalanceo severo
    else:
        return "RandomUnderSampler"  # Desbalanceo extremo


__all__ = [
    "analyze_class_distribution",
    "balance_dataset",
    "validate_balancing_result",
    "get_balancing_strategy",
]
