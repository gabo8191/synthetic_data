"""
Módulo de modelos GAN.

Este módulo contiene las definiciones de los modelos Generator y Discriminator
para el GAN, incluyendo sus arquitecturas y configuraciones.
"""

import torch
import torch.nn as nn
from typing import Optional


class Generator(nn.Module):
    """
    Generador del GAN.

    Arquitectura:
    - Capa de entrada: latent_dim -> hidden_size
    - Activación: ReLU
    - Capa de salida: hidden_size -> input_dim
    - Activación: Tanh (para normalizar salida a [-1, 1])
    """

    def __init__(self, latent_dim: int, input_dim: int, hidden_size: int = 128):
        """
        Inicializa el generador.

        Args:
            latent_dim: Dimensión del espacio latente
            input_dim: Dimensión de entrada (número de características)
            hidden_size: Tamaño de la capa oculta
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del generador.

        Args:
            z: Tensor de ruido latente

        Returns:
            Tensor con datos generados
        """
        return self.model(z)

    def get_model_info(self) -> dict:
        """
        Obtiene información del modelo.

        Returns:
            Diccionario con información del modelo
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "latent_dim": self.latent_dim,
            "input_dim": self.input_dim,
            "hidden_size": self.hidden_size,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture": "Linear -> ReLU -> Linear -> Tanh",
        }


class Discriminator(nn.Module):
    """
    Discriminador del GAN.

    Arquitectura:
    - Capa de entrada: input_dim -> hidden_size
    - Activación: ReLU
    - Capa de salida: hidden_size -> 1
    - Activación: Sigmoid (para probabilidad de ser real)
    """

    def __init__(self, input_dim: int, hidden_size: int = 128):
        """
        Inicializa el discriminador.

        Args:
            input_dim: Dimensión de entrada (número de características)
            hidden_size: Tamaño de la capa oculta
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del discriminador.

        Args:
            x: Tensor de datos de entrada

        Returns:
            Tensor con probabilidad de ser real
        """
        return self.model(x)

    def get_model_info(self) -> dict:
        """
        Obtiene información del modelo.

        Returns:
            Diccionario con información del modelo
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "input_dim": self.input_dim,
            "hidden_size": self.hidden_size,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture": "Linear -> ReLU -> Linear -> Sigmoid",
        }


def create_gan_models(
    input_dim: int,
    latent_dim: int = 100,
    hidden_size: int = 128,
    device: Optional[str] = None,
) -> tuple[Generator, Discriminator]:
    """
    Crea los modelos Generator y Discriminator.

    Args:
        input_dim: Dimensión de entrada
        latent_dim: Dimensión del espacio latente
        hidden_size: Tamaño de la capa oculta
        device: Dispositivo donde colocar los modelos

    Returns:
        Tupla con (Generator, Discriminator)
    """
    generator = Generator(latent_dim, input_dim, hidden_size)
    discriminator = Discriminator(input_dim, hidden_size)

    if device:
        generator = generator.to(device)
        discriminator = discriminator.to(device)

    return generator, discriminator


def get_models_summary(generator: Generator, discriminator: Discriminator) -> dict:
    """
    Obtiene un resumen de ambos modelos.

    Args:
        generator: Modelo generador
        discriminator: Modelo discriminador

    Returns:
        Diccionario con resumen de ambos modelos
    """
    return {
        "generator": generator.get_model_info(),
        "discriminator": discriminator.get_model_info(),
        "total_parameters": (
            generator.get_model_info()["total_parameters"]
            + discriminator.get_model_info()["total_parameters"]
        ),
    }


__all__ = ["Generator", "Discriminator", "create_gan_models", "get_models_summary"]
