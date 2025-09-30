"""
Módulo de generación de datos sintéticos.

Este módulo contiene funciones para generar datos sintéticos usando
GANs entrenados, incluyendo generación estándar y estratificada.
"""

import numpy as np
import pandas as pd
import torch
import logging
from typing import Optional

from src.config import config
from src.models.gan_models import Generator
from src.models.data_processing import (
    denormalize_data,
    create_latent_noise,
    preprocess_dataframe,
)
from src.models.training import GANTrainer


def generate_synthetic_data(
    generator: Generator,
    quantity: int,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Genera datos sintéticos usando un generador entrenado.

    Args:
        generator: Generador entrenado
        quantity: Cantidad de muestras a generar
        min_vals: Valores mínimos para desnormalización
        max_vals: Valores máximos para desnormalización
        device: Dispositivo donde está el generador

    Returns:
        Array con datos sintéticos generados
    """
    if device is None:
        device = next(generator.parameters()).device

    generator.eval()

    with torch.no_grad():
        # Crear ruido latente
        z = create_latent_noise(quantity, generator.latent_dim, device)

        # Generar datos
        synthetic_normalized = generator(z).cpu().numpy()

    # Desnormalizar
    synthetic_data = denormalize_data(synthetic_normalized, min_vals, max_vals)

    return synthetic_data


def generate_standard_data(
    df: pd.DataFrame,
    quantity: int,
    epochs: int = 1200,
    batch_size: int = 64,
    latent_dim: int = 100,
    log_prefix: str = "",
) -> pd.DataFrame:
    """
    Genera datos sintéticos usando un solo GAN.

    Args:
        df: DataFrame con datos de entrenamiento
        quantity: Cantidad de muestras a generar
        epochs: Número de épocas de entrenamiento
        batch_size: Tamaño del batch
        latent_dim: Dimensión del espacio latente
        log_prefix: Prefijo para logs

    Returns:
        DataFrame con datos sintéticos generados
    """
    logger = logging.getLogger("generation")

    # Preprocesar datos
    processed_df = preprocess_dataframe(df)

    # Crear entrenador
    trainer = GANTrainer(
        latent_dim=latent_dim,
        hidden_size=config.gan.GENERATOR_HIDDEN_SIZE,
        learning_rate=config.gan.LEARNING_RATE,
    )

    # Preparar datos para entrenamiento
    from src.models.data_processing import prepare_data_for_training

    dataloader, _, min_vals, max_vals = prepare_data_for_training(
        processed_df, batch_size
    )

    # Configurar modelos
    input_dim = processed_df.shape[1]
    trainer.setup_models(input_dim)

    # Entrenar
    logger.info(f"Iniciando entrenamiento estándar: {epochs} épocas")
    for epoch in range(epochs):
        trainer.train_epoch(dataloader, epoch, log_prefix)

    # Generar datos sintéticos
    synthetic_data = generate_synthetic_data(
        trainer.generator, quantity, min_vals, max_vals, trainer.device
    )

    return pd.DataFrame(synthetic_data, columns=df.columns)


def generate_stratified_data(
    df: pd.DataFrame,
    quantity: int,
    labels: pd.Series,
    epochs: int = 1200,
    batch_size: int = 64,
    latent_dim: int = 100,
) -> pd.DataFrame:
    """
    Genera datos sintéticos usando entrenamiento estratificado por clase.

    Args:
        df: DataFrame con datos de entrenamiento
        quantity: Cantidad de muestras a generar
        labels: Etiquetas para estratificación
        epochs: Número de épocas de entrenamiento
        batch_size: Tamaño del batch
        latent_dim: Dimensión del espacio latente

    Returns:
        DataFrame con datos sintéticos generados
    """
    logger = logging.getLogger("generation")

    unique_classes = sorted(labels.unique())
    num_classes = len(unique_classes)
    per_class = max(1, quantity // num_classes)

    logger.info(
        f"Generación estratificada: {num_classes} clases, {per_class} muestras por clase"
    )

    generated_parts: list[pd.DataFrame] = []

    for cls in unique_classes:
        mask = labels == cls
        df_cls = df.loc[mask]

        if len(df_cls) == 0:
            logger.warning(f"No hay datos para la clase {cls}, omitiendo...")
            continue

        logger.info(f"Entrenando GAN para clase={cls} con {len(df_cls)} muestras...")

        part = generate_standard_data(
            df_cls,
            quantity=per_class,
            epochs=epochs,
            batch_size=batch_size,
            latent_dim=latent_dim,
            log_prefix=f"[cls={cls}]",
        )
        generated_parts.append(part)

    if not generated_parts:
        raise ValueError("No se pudieron generar datos para ninguna clase")

    # Combinar todas las partes
    df_sintetico = pd.concat(generated_parts, axis=0, ignore_index=True)

    # Generar muestras adicionales si es necesario
    if len(df_sintetico) < quantity:
        remaining = quantity - len(df_sintetico)
        logger.info(f"Generando {remaining} muestras adicionales...")

        # Usar la primera clase disponible para muestras adicionales
        first_cls = unique_classes[0]
        mask = labels == first_cls
        df_cls = df.loc[mask]

        extra = generate_standard_data(
            df_cls,
            quantity=remaining,
            epochs=epochs,
            batch_size=batch_size,
            latent_dim=latent_dim,
            log_prefix=f"[cls={first_cls}]",
        )
        df_sintetico = pd.concat([df_sintetico, extra], axis=0, ignore_index=True)

    return df_sintetico


def generate_with_custom_gan(
    trainer: GANTrainer,
    df: pd.DataFrame,
    quantity: int,
    epochs: int = 1200,
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Genera datos sintéticos usando un entrenador GAN personalizado.

    Args:
        trainer: Entrenador GAN ya configurado
        df: DataFrame con datos de entrenamiento
        quantity: Cantidad de muestras a generar
        epochs: Número de épocas de entrenamiento
        batch_size: Tamaño del batch

    Returns:
        DataFrame con datos sintéticos generados
    """
    logger = logging.getLogger("generation")

    # Preprocesar datos
    processed_df = preprocess_dataframe(df)

    # Preparar datos para entrenamiento
    from src.models.data_processing import prepare_data_for_training

    dataloader, _, min_vals, max_vals = prepare_data_for_training(
        processed_df, batch_size
    )

    # Configurar modelos si no están configurados
    if not trainer._validate_models():
        input_dim = processed_df.shape[1]
        trainer.setup_models(input_dim)

    # Entrenar
    logger.info(f"Iniciando entrenamiento personalizado: {epochs} épocas")
    for epoch in range(epochs):
        trainer.train_epoch(dataloader, epoch)

    # Generar datos sintéticos
    synthetic_data = generate_synthetic_data(
        trainer.generator, quantity, min_vals, max_vals, trainer.device
    )

    return pd.DataFrame(synthetic_data, columns=df.columns)


def validate_generated_data(
    original_df: pd.DataFrame, synthetic_df: pd.DataFrame
) -> dict:
    """
    Valida la calidad de los datos sintéticos generados.

    Args:
        original_df: DataFrame original
        synthetic_df: DataFrame sintético

    Returns:
        Diccionario con métricas de validación
    """
    validation_report = {
        "shape_comparison": {
            "original": original_df.shape,
            "synthetic": synthetic_df.shape,
        },
        "column_comparison": {
            "original_columns": list(original_df.columns),
            "synthetic_columns": list(synthetic_df.columns),
            "columns_match": list(original_df.columns) == list(synthetic_df.columns),
        },
        "data_types_match": original_df.dtypes.equals(synthetic_df.dtypes),
        "missing_values": {
            "original": original_df.isnull().sum().sum(),
            "synthetic": synthetic_df.isnull().sum().sum(),
        },
    }

    # Comparar estadísticas básicas para columnas numéricas
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        validation_report["numeric_stats_comparison"] = {}

        for col in numeric_cols:
            if col in synthetic_df.columns:
                validation_report["numeric_stats_comparison"][col] = {
                    "original_mean": float(original_df[col].mean()),
                    "synthetic_mean": float(synthetic_df[col].mean()),
                    "original_std": float(original_df[col].std()),
                    "synthetic_std": float(synthetic_df[col].std()),
                    "mean_difference": abs(
                        original_df[col].mean() - synthetic_df[col].mean()
                    ),
                }

    return validation_report


__all__ = [
    "generate_synthetic_data",
    "generate_standard_data",
    "generate_stratified_data",
    "generate_with_custom_gan",
    "validate_generated_data",
]
