"""
Módulo de entrenamiento para GAN.

Este módulo contiene la lógica de entrenamiento del GAN, incluyendo
el bucle de entrenamiento, cálculo de pérdidas y optimización.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from typing import Optional, Dict, Tuple

from src.config import config
from src.models.gan_models import Generator, Discriminator
from src.models.data_processing import create_latent_noise, create_labels


class GANTrainer:
    """
    Entrenador del GAN con funcionalidades mejoradas.

    Maneja el entrenamiento completo del GAN incluyendo:
    - Configuración de modelos y optimizadores
    - Bucle de entrenamiento
    - Cálculo de pérdidas
    - Registro de estadísticas
    """

    def __init__(
        self,
        latent_dim: int = 100,
        hidden_size: int = 128,
        learning_rate: float = 0.0002,
        device: Optional[str] = None,
    ):
        """
        Inicializa el entrenador GAN.

        Args:
            latent_dim: Dimensión del espacio latente
            hidden_size: Tamaño de la capa oculta
            learning_rate: Tasa de aprendizaje
            device: Dispositivo para entrenamiento
        """
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger("gan_training")

        # Inicializar modelos (se configurarán en setup_models)
        self.generator: Optional[Generator] = None
        self.discriminator: Optional[Discriminator] = None
        self.optimizer_G: Optional[optim.Adam] = None
        self.optimizer_D: Optional[optim.Adam] = None
        self.loss_fn = nn.BCELoss()

        # Estadísticas de entrenamiento
        self.training_history: Dict[str, list] = {
            "generator_loss": [],
            "discriminator_loss": [],
        }

    def setup_models(self, input_dim: int) -> None:
        """
        Configura los modelos del GAN.

        Args:
            input_dim: Dimensión de entrada
        """
        from src.models.gan_models import create_gan_models

        self.generator, self.discriminator = create_gan_models(
            input_dim=input_dim,
            latent_dim=self.latent_dim,
            hidden_size=self.hidden_size,
            device=self.device,
        )

        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.999),  # Parámetros recomendados para GAN
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999)
        )

        self.logger.info(f"Modelos configurados en dispositivo: {self.device}")

    def train_epoch(
        self, dataloader: DataLoader, epoch: int, log_prefix: str = ""
    ) -> Tuple[float, float]:
        """
        Entrena una época completa.

        Args:
            dataloader: DataLoader con los datos
            epoch: Número de época
            log_prefix: Prefijo para logs

        Returns:
            Tupla con (pérdida_promedio_generador, pérdida_promedio_discriminador)
        """
        if not self._validate_models():
            raise RuntimeError("Modelos no inicializados correctamente")

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0

        for (real_batch,) in dataloader:
            real_batch = real_batch.to(self.device)
            batch_size = real_batch.size(0)

            # Crear etiquetas
            real_labels = create_labels(batch_size, real=True, device=self.device)
            fake_labels = create_labels(batch_size, real=False, device=self.device)

            # Entrenar discriminador
            d_loss = self._train_discriminator(real_batch, real_labels, fake_labels)

            # Entrenar generador
            g_loss = self._train_generator(real_labels)

            epoch_d_loss += d_loss
            epoch_g_loss += g_loss
            num_batches += 1

        # Calcular promedios
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches

        # Registrar estadísticas
        self.training_history["discriminator_loss"].append(avg_d_loss)
        self.training_history["generator_loss"].append(avg_g_loss)

        # Log periódico
        if epoch % config.gan.LOG_INTERVAL == 0:
            self.logger.info(
                f"{log_prefix} Epoch {epoch} | Loss D: {avg_d_loss:.4f} | Loss G: {avg_g_loss:.4f}"
            )

        return avg_g_loss, avg_d_loss

    def _train_discriminator(
        self,
        real_batch: torch.Tensor,
        real_labels: torch.Tensor,
        fake_labels: torch.Tensor,
    ) -> float:
        """
        Entrena el discriminador.

        Args:
            real_batch: Batch de datos reales
            real_labels: Etiquetas para datos reales
            fake_labels: Etiquetas para datos falsos

        Returns:
            Pérdida del discriminador
        """
        if not self._validate_models():
            raise RuntimeError("Modelos no inicializados correctamente")

        # Type assertions para el linter
        assert self.generator is not None
        assert self.discriminator is not None
        assert self.optimizer_D is not None

        batch_size = real_batch.size(0)

        # Generar datos falsos
        z = create_latent_noise(batch_size, self.latent_dim, self.device)
        fake_data = self.generator(z)

        # Predicciones del discriminador
        real_preds = self.discriminator(real_batch)
        fake_preds = self.discriminator(fake_data.detach())

        # Pérdida del discriminador
        loss_D = self.loss_fn(real_preds, real_labels) + self.loss_fn(
            fake_preds, fake_labels
        )

        # Backpropagation
        self.optimizer_D.zero_grad()
        loss_D.backward()
        self.optimizer_D.step()

        return loss_D.item()

    def _train_generator(self, real_labels: torch.Tensor) -> float:
        """
        Entrena el generador.

        Args:
            real_labels: Etiquetas para datos reales

        Returns:
            Pérdida del generador
        """
        if not self._validate_models():
            raise RuntimeError("Modelos no inicializados correctamente")

        # Type assertions para el linter
        assert self.generator is not None
        assert self.discriminator is not None
        assert self.optimizer_G is not None

        batch_size = real_labels.size(0)

        # Generar datos falsos
        z = create_latent_noise(batch_size, self.latent_dim, self.device)
        fake_data = self.generator(z)
        fake_preds = self.discriminator(fake_data)

        # Pérdida del generador (quiere engañar al discriminador)
        loss_G = self.loss_fn(fake_preds, real_labels)

        # Backpropagation
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

        return loss_G.item()

    def _validate_models(self) -> bool:
        """
        Valida que los modelos estén inicializados.

        Returns:
            True si todos los modelos están inicializados
        """
        return (
            self.generator is not None
            and self.discriminator is not None
            and self.optimizer_G is not None
            and self.optimizer_D is not None
        )

    def get_training_summary(self) -> Dict:
        """
        Obtiene un resumen del entrenamiento.

        Returns:
            Diccionario con resumen del entrenamiento
        """
        if not self.training_history["generator_loss"]:
            return {"status": "No training completed"}

        final_g_loss = self.training_history["generator_loss"][-1]
        final_d_loss = self.training_history["discriminator_loss"][-1]

        return {
            "total_epochs": len(self.training_history["generator_loss"]),
            "final_generator_loss": final_g_loss,
            "final_discriminator_loss": final_d_loss,
            "min_generator_loss": min(self.training_history["generator_loss"]),
            "min_discriminator_loss": min(self.training_history["discriminator_loss"]),
            "convergence_ratio": final_d_loss / (final_g_loss + 1e-8),
        }

    def save_training_history(self, filepath: str) -> None:
        """
        Guarda el historial de entrenamiento.

        Args:
            filepath: Ruta donde guardar el historial
        """
        import json

        with open(filepath, "w") as f:
            json.dump(self.training_history, f, indent=2)

        self.logger.info(f"Historial de entrenamiento guardado en: {filepath}")


__all__ = ["GANTrainer"]
