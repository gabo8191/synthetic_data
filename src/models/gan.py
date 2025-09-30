"""
Módulo orquestador GAN.

Este módulo coordina la generación de datos sintéticos usando GANs,
integrando todos los componentes especializados en una API unificada.
"""

import pandas as pd
import logging
from typing import Optional

from src.config import config
from src.validation import validate_gan_parameters
from src.models.gan_models import Generator, Discriminator
from src.models.training import GANTrainer
from src.models.generation import (
    generate_standard_data,
    generate_stratified_data,
    validate_generated_data,
)


def simple_gan_generator(
    df: pd.DataFrame,
    cantidad: int,
    epochs: int = 1200,
    batch_size: int = 64,
    latent_dim: int = 100,
    *,
    labels: Optional[pd.Series] = None,
    stratified: bool = False,
) -> pd.DataFrame:
    """
    Genera datos sintéticos con un GAN sencillo.

    Args:
        df: DataFrame con datos de entrenamiento
        cantidad: Cantidad de muestras a generar
        epochs: Número de épocas de entrenamiento
        batch_size: Tamaño del batch
        latent_dim: Dimensión del espacio latente
        labels: Etiquetas para entrenamiento estratificado
        stratified: Si usar entrenamiento estratificado por clase

    Returns:
        DataFrame con datos sintéticos generados
    """
    logger = logging.getLogger("gan")

    # Validar parámetros
    validate_gan_parameters(
        epochs, batch_size, latent_dim, config.gan.LEARNING_RATE, len(df)
    )

    logger.info(f"Iniciando generación GAN: {cantidad} muestras, {epochs} épocas")

    try:
        if stratified and labels is not None:
            logger.info("Usando entrenamiento estratificado")
            synthetic_df = generate_stratified_data(
                df, cantidad, labels, epochs, batch_size, latent_dim
            )
        else:
            logger.info("Usando entrenamiento estándar")
            synthetic_df = generate_standard_data(
                df, cantidad, epochs, batch_size, latent_dim
            )

        # Validar datos generados
        validation_report = validate_generated_data(df, synthetic_df)
        logger.info(f"Datos sintéticos generados: {synthetic_df.shape}")
        logger.info(
            f"Validación: {validation_report['column_comparison']['columns_match']}"
        )

        return synthetic_df

    except Exception as e:
        logger.error(f"Error en generación GAN: {str(e)}")
        raise RuntimeError(f"Error en generación GAN: {str(e)}")


def create_gan_trainer(
    latent_dim: int = 100,
    hidden_size: int = 128,
    learning_rate: float = 0.0002,
    device: Optional[str] = None,
) -> GANTrainer:
    """
    Crea un entrenador GAN personalizado.

    Args:
        latent_dim: Dimensión del espacio latente
        hidden_size: Tamaño de la capa oculta
        learning_rate: Tasa de aprendizaje
        device: Dispositivo para entrenamiento

    Returns:
        Entrenador GAN configurado
    """
    return GANTrainer(
        latent_dim=latent_dim,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        device=device,
    )


def train_gan_with_custom_config(
    df: pd.DataFrame, trainer: GANTrainer, epochs: int = 1200, batch_size: int = 64
) -> pd.DataFrame:
    """
    Entrena un GAN con configuración personalizada.

    Args:
        df: DataFrame con datos de entrenamiento
        trainer: Entrenador GAN personalizado
        epochs: Número de épocas
        batch_size: Tamaño del batch

    Returns:
        DataFrame con datos sintéticos generados
    """
    from src.models.generation import generate_with_custom_gan

    logger = logging.getLogger("gan")
    logger.info(f"Entrenando GAN personalizado: {epochs} épocas")

    return generate_with_custom_gan(trainer, df, len(df), epochs, batch_size)


# Re-exportar clases y funciones para compatibilidad
from src.models.gan_models import Generator, Discriminator
from src.models.training import GANTrainer


__all__ = [
    "simple_gan_generator",
    "create_gan_trainer",
    "train_gan_with_custom_config",
    "Generator",
    "Discriminator",
    "GANTrainer",
]
