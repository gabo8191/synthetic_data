import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging


def _train_gan_on_dataframe(
    df: pd.DataFrame,
    cantidad: int,
    *,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    log_prefix: str,
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger("gan")

    data = df.values.astype(np.float32)

    # Normalización [-1, 1] para Tanh
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    data = 2 * (data - min_vals) / (max_vals - min_vals + 1e-8) - 1

    dataset = TensorDataset(torch.tensor(data))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_dim = data.shape[1]

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim),
                nn.Tanh(),
            )

        def forward(self, z):
            return self.model(z)

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()
            )

        def forward(self, x):
            return self.model(x)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    loss_fn = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(epochs):
        for (real_batch,) in dataloader:
            real_batch = real_batch.to(device)

            # Discriminador
            real_labels = torch.ones(real_batch.size(0), 1).to(device)
            fake_labels = torch.zeros(real_batch.size(0), 1).to(device)

            z = torch.randn(real_batch.size(0), latent_dim).to(device)
            fake_data = generator(z)

            real_preds = discriminator(real_batch)
            fake_preds = discriminator(fake_data.detach())
            loss_D = loss_fn(real_preds, real_labels) + loss_fn(fake_preds, fake_labels)

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # Generador
            z = torch.randn(real_batch.size(0), latent_dim).to(device)
            fake_data = generator(z)
            fake_preds = discriminator(fake_data)
            loss_G = loss_fn(fake_preds, real_labels)

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

        if epoch % 100 == 0:
            logger.info(
                f"{log_prefix} Epoch {epoch} | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}"
            )

    # Generación
    generator.eval()
    with torch.no_grad():
        z = torch.randn(cantidad, latent_dim).to(device)
        synthetic_data = generator(z).cpu().numpy()

    # Desnormaliza
    synthetic_data = (synthetic_data + 1) / 2
    synthetic_data = synthetic_data * (max_vals - min_vals + 1e-8) + min_vals

    df_sintetico = pd.DataFrame(synthetic_data, columns=df.columns)
    return df_sintetico


def simple_gan_generator(
    df: pd.DataFrame,
    cantidad: int,
    epochs: int = 1200,
    batch_size: int = 64,
    latent_dim: int = 100,
    *,
    labels: pd.Series | None = None,
    stratified: bool = False,
) -> pd.DataFrame:
    """Genera datos sintéticos con un GAN sencillo.

    - Si stratified=True y labels se proporcionan, entrena un GAN por clase
      y concatena las muestras generadas en proporciones iguales.
    """
    logger = logging.getLogger("gan")

    if stratified and labels is not None:
        # Entrena por clase
        unique_classes = sorted(pd.Series(labels).unique())
        num_classes = len(unique_classes)
        per_class = max(1, cantidad // max(1, num_classes))
        generated_parts: list[pd.DataFrame] = []
        for cls in unique_classes:
            mask = labels == cls
            df_cls = df.loc[mask]
            logger.info(f"Entrenando GAN por clase={cls} con {len(df_cls)} filas...")
            part = _train_gan_on_dataframe(
                df_cls,
                cantidad=per_class,
                epochs=epochs,
                batch_size=batch_size,
                latent_dim=latent_dim,
                log_prefix=f"[cls={cls}]",
            )
            generated_parts.append(part)

        df_sintetico = pd.concat(generated_parts, axis=0, ignore_index=True)
        # Si faltan muestras por división entera, genera el remanente en la primera clase
        if len(df_sintetico) < cantidad:
            remaining = cantidad - len(df_sintetico)
            first_cls = unique_classes[0]
            mask = labels == first_cls
            df_cls = df.loc[mask]
            extra = _train_gan_on_dataframe(
                df_cls,
                cantidad=remaining,
                epochs=epochs,
                batch_size=batch_size,
                latent_dim=latent_dim,
                log_prefix=f"[cls={first_cls}]",
            )
            df_sintetico = pd.concat([df_sintetico, extra], axis=0, ignore_index=True)
        return df_sintetico

    # Estrategia estándar: un solo GAN
    return _train_gan_on_dataframe(
        df,
        cantidad=cantidad,
        epochs=epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        log_prefix="",
    )


__all__ = ["simple_gan_generator"]
