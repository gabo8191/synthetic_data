"""
Módulo de procesamiento de datos para GAN.

Este módulo contiene funciones para normalización, desnormalización
y preparación de datos para el entrenamiento del GAN.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional

from src.config import config


def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normaliza los datos al rango [-1, 1].

    Args:
        data: Array de datos a normalizar

    Returns:
        Tupla con (datos_normalizados, valores_mínimos, valores_máximos)
    """
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # Evitar división por cero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = config.gan.EPSILON

    normalized_data = 2 * (data - min_vals) / range_vals - 1

    return normalized_data, min_vals, max_vals


def denormalize_data(
    normalized_data: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray
) -> np.ndarray:
    """
    Desnormaliza los datos del rango [-1, 1] al rango original.

    Args:
        normalized_data: Datos normalizados
        min_vals: Valores mínimos originales
        max_vals: Valores máximos originales

    Returns:
        Datos desnormalizados
    """
    # Evitar división por cero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = config.gan.EPSILON

    # Desnormalizar: [-1, 1] -> [0, 1] -> [min, max]
    denormalized = (normalized_data + 1) / 2
    denormalized = denormalized * range_vals + min_vals

    return denormalized


def prepare_data_for_training(
    df: pd.DataFrame, batch_size: int = 64, shuffle: bool = True
) -> Tuple[DataLoader, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepara los datos para el entrenamiento del GAN.

    Args:
        df: DataFrame con los datos
        batch_size: Tamaño del batch
        shuffle: Si mezclar los datos

    Returns:
        Tupla con (DataLoader, datos_normalizados, valores_mínimos, valores_máximos)
    """
    # Convertir a numpy y normalizar
    data = df.values.astype(np.float32)
    normalized_data, min_vals, max_vals = normalize_data(data)

    # Crear DataLoader
    dataset = TensorDataset(torch.tensor(normalized_data))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,  # Evitar batches incompletos
    )

    return dataloader, normalized_data, min_vals, max_vals


def create_latent_noise(
    batch_size: int, latent_dim: int, device: Optional[str] = None
) -> torch.Tensor:
    """
    Crea ruido latente para el generador.

    Args:
        batch_size: Tamaño del batch
        latent_dim: Dimensión del espacio latente
        device: Dispositivo donde crear el tensor

    Returns:
        Tensor con ruido latente
    """
    noise = torch.randn(batch_size, latent_dim)
    if device:
        noise = noise.to(device)
    return noise


def create_labels(
    batch_size: int, real: bool = True, device: Optional[str] = None
) -> torch.Tensor:
    """
    Crea etiquetas para el entrenamiento.

    Args:
        batch_size: Tamaño del batch
        real: Si las etiquetas son para datos reales (True) o falsos (False)
        device: Dispositivo donde crear el tensor

    Returns:
        Tensor con etiquetas
    """
    labels = torch.ones(batch_size, 1) if real else torch.zeros(batch_size, 1)
    if device:
        labels = labels.to(device)
    return labels


def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    Valida la calidad de los datos para entrenamiento GAN.

    Args:
        df: DataFrame a validar

    Returns:
        Diccionario con métricas de calidad
    """
    quality_report = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().sum(),
        "infinite_values": np.isinf(df.select_dtypes(include=[np.number])).sum().sum(),
        "constant_columns": [],
        "data_types": df.dtypes.to_dict(),
    }

    # Identificar columnas constantes
    for col in df.columns:
        if df[col].nunique() <= 1:
            quality_report["constant_columns"].append(col)

    # Estadísticas básicas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        quality_report["numeric_stats"] = {
            "mean": df[numeric_cols].mean().to_dict(),
            "std": df[numeric_cols].std().to_dict(),
            "min": df[numeric_cols].min().to_dict(),
            "max": df[numeric_cols].max().to_dict(),
        }

    return quality_report


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesa un DataFrame para entrenamiento GAN.

    Args:
        df: DataFrame a preprocesar

    Returns:
        DataFrame preprocesado
    """
    processed_df = df.copy()

    # Manejar valores faltantes
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        processed_df[numeric_cols] = processed_df[numeric_cols].fillna(
            processed_df[numeric_cols].median()
        )

    # Manejar valores infinitos
    processed_df = processed_df.replace([np.inf, -np.inf], np.nan)
    if len(numeric_cols) > 0:
        processed_df[numeric_cols] = processed_df[numeric_cols].fillna(
            processed_df[numeric_cols].median()
        )

    return processed_df


__all__ = [
    "normalize_data",
    "denormalize_data",
    "prepare_data_for_training",
    "create_latent_noise",
    "create_labels",
    "validate_data_quality",
    "preprocess_dataframe",
]
