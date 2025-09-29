## Algoritmo y pipeline de generación de datos sintéticos

Este documento describe, en tercera persona, el flujo end‑to‑end, las decisiones de preprocesamiento y el funcionamiento del GAN (Generative Adversarial Network = red generativa adversaria) utilizado para generar datos sintéticos del Titanic. Se incluyen referencias directas al código fuente y a los artefactos generados en `results/`.

### 1) Carga y limpieza del dataset

La carga y limpieza residen en `src/data/loader.py`. Se estandarizan nombres a snake_case (snake_case = minúsculas con guiones bajos), se convierten tipos numéricos y se imputan nulos con mediana/moda. También se eliminan duplicados y se registra un reporte completo.

```1:18:src/data/loader.py
import pandas as pd
import numpy as np
import os
import re


def load_and_clean_data(
    file_path: str, target_column: str
) -> tuple[pd.DataFrame, dict]:
    """Carga y limpia datos, normaliza nombres y tipos, y rellena nulos.

    - Normaliza nombres de columnas a snake_case lowercase
    - Convierte strings vacíos y marcadores a NaN y los imputa
    - Convierte columnas numéricas a float64
    """
    report: dict[str, object] = {}
```

La normalización de columnas y la imputación se realizan aquí:

```39:67:src/data/loader.py
# Normalizar nombres de columnas
df.columns = [
    re.sub(r"\W+", "_", str(col).strip().lower()) for col in df.columns
]
...
# Convertir numéricos a float y manejar nulos con mediana/moda
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
...
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype in ["float64", "int64"]:
            fill_val = df[col].median()
        else:
            fill_val = df[col].mode()[0]
        df[col] = df[col].fillna(fill_val)
```

El resultado queda documentado en `results/reports/cleaning_report.json` y el dataset limpio se guarda en `results/data/cleaned_dataset.csv` desde el pipeline:

```64:80:src/pipeline/main.py
df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_dataset.csv"), index=False)
with open(os.path.join(REPORTS_DIR, "cleaning_report.json"), "w") as f:
    json.dump(clean_report, f, indent=4, default=str)
...
balanced_df.to_csv(os.path.join(OUTPUT_DIR, "balanced_data.csv"), index=False)
with open(os.path.join(REPORTS_DIR, "analysis_report.json"), "w") as f:
    json.dump(analysis_report, f, indent=4, default=str)
```

### 2) Análisis de clases y balanceo

El análisis de la clase objetivo y el balanceo por submuestreo están encapsulados en `analyze_data`.

```21:55:src/analysis/analyzer.py
# Distribución de clases original y cálculo de minoría/ratio de desbalance
class_dist = df[target_column].value_counts().to_dict()
...
undersampler = RandomUnderSampler(sampling_strategy="all", random_state=42)
X_res, y_res = undersampler.fit_resample(X, y)
balanced_df = pd.DataFrame(X_res, columns=X.columns)
balanced_df[target_column] = y_res
```

Además, se generan los gráficos `results/graphics/original_dist.png` y `results/graphics/balanced_dist.png` con `plot_class_distribution` (countplot = barras de conteo por clase).

### 3) Preparación de variables para el generador

En el pipeline, antes de sintetizar, se eliminan columnas tipo ID, se aplica One‑Hot (codificación binaria por categoría) a categóricas y se escalan sólo columnas continuas usando `StandardScaler` (estandarización a media 0 y desviación 1).

```92:129:src/pipeline/main.py
# Quitar identificadores
id_like_candidates = ["passengerid", "name", "ticket", "cabin"]
...
# OneHotEncoding de categóricas y selección de continuas
categorical_cols = list(features_df.select_dtypes(include="object").columns)
if len(categorical_cols) > 0:
    features_df = pd.get_dummies(features_df, columns=categorical_cols)
...
scaler = StandardScaler()
scaled_features_df[continuous_cols] = scaler.fit_transform(features_df[continuous_cols])
```

Posteriormente, tras generar, se aplica la inversa del escalado:

```140:145:src/pipeline/main.py
synthetic_df = synthetic_scaled_df.copy()
if scaler is not None and len(continuous_cols) > 0:
    synthetic_df[continuous_cols] = scaler.inverse_transform(
        synthetic_scaled_df[continuous_cols]
    )
```

### 4) Generación con GAN sencillo

El generador/discriminador son redes densas (fully‑connected) con activaciones `ReLU` y salida `Tanh` para el generador. Los datos se normalizan a [-1, 1] antes de entrenar para que `Tanh` opere en su rango efectivo.

```24:41:src/models/gan.py
# Normalización a [-1, 1] para que Tanh funcione adecuadamente
min_vals = data.min(axis=0)
max_vals = data.max(axis=0)
data = 2 * (data - min_vals) / (max_vals - min_vals + 1e-8) - 1
...
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Tanh(),
        )
```

El bucle de entrenamiento alterna actualizaciones de D (discriminador) y G (generador) con `BCELoss` (Binary Cross‑Entropy = entropía cruzada binaria) y `Adam` (optimizador de gradiente adaptativo):

```62:89:src/models/gan.py
for epoch in range(epochs):
    for (real_batch,) in dataloader:
        # Discriminador
        real_preds = discriminator(real_batch)
        fake_preds = discriminator(fake_data.detach())
        loss_D = loss_fn(real_preds, real_labels) + loss_fn(fake_preds, fake_labels)
        ...
        # Generador
        fake_preds = discriminator(fake_data)
        loss_G = loss_fn(fake_preds, real_labels)
        ...
```

El entrenamiento puede ser estratificado por clase del objetivo (`stratified=True`), entrenando un GAN por clase y concatenando resultados. Estratificar (stratify) significa entrenar modelos condicionados a cada clase del objetivo para preservar dependencias condicionales:

```127:147:src/models/gan.py
if stratified and labels is not None:
    unique_classes = sorted(pd.Series(labels).unique())
    per_class = max(1, cantidad // max(1, num_classes))
    for cls in unique_classes:
        df_cls = df.loc[mask]
        part = _train_gan_on_dataframe(df_cls, cantidad=per_class, epochs=epochs, ...)
        generated_parts.append(part)
df_sintetico = pd.concat(generated_parts, axis=0, ignore_index=True)
```

La generación final desnormaliza (inversa de la normalización) de vuelta al dominio original:

```96:107:src/models/gan.py
with torch.no_grad():
    z = torch.randn(cantidad, latent_dim).to(device)
    synthetic_data = generator(z).cpu().numpy()
# Desnormaliza a rango original por columna
synthetic_data = (synthetic_data + 1) / 2
synthetic_data = synthetic_data * (max_vals - min_vals + 1e-8) + min_vals
df_sintetico = pd.DataFrame(synthetic_data, columns=df.columns)
```

El CSV resultante se guarda en `results/data/synthetic_data.csv`.

### 5) Comparación y generación de gráficos unificada

Las comparaciones se realizan en `compare_real_vs_synthetic`. Para barras lado a lado se utiliza una única utilidad para garantizar estilo consistente y evitar discrepancias entre figuras:

```72:120:src/analysis/analyzer.py
def _plot_side_by_side_bars(...):
    COLOR_REAL = "#4e79a7"
    COLOR_SYNTHETIC = "#e15759"
    COLOR_BALANCED = "#59a14f"
    ...
    plt.bar([i - 0.2 for i in x], left_values, width=0.4, label=left_label, color=COLOR_REAL)
    plt.bar([i + 0.2 for i in x], right_values, width=0.4, label=right_label, color=right_color)
```

Para numéricas, se calculan histogramas con mismos bordes de bin (bin = intervalo) para una comparación justa y se guardan en `results/graphics/compare/compare_<col>.png`:

```169:205:src/analysis/analyzer.py
real_counts, edges = np.histogram(real_vals, bins=bins, range=(vmin, vmax))
syn_counts, _ = np.histogram(syn_vals, bins=bins, range=(vmin, vmax))
real_props = real_counts / max(1, len(real_vals))
syn_props = syn_counts / max(1, len(syn_vals))
...
plt.savefig(os.path.join(compare_dir, f"compare_{col}.png"))
```

Para categóricas, si los sintéticos están one-hot, se agregan con la media por prefijo, alineando categorías y usando la utilidad de barras:

```236:281:src/analysis/analyzer.py
prefix = f"{col}_"
oh_cols = [c for c in synthetic_df.columns if c.startswith(prefix)]
...
syn_counts = (synthetic_df[oh_cols].mean().rename(lambda x: x.replace(prefix, "")))
...
_plot_side_by_side_bars(all_cats, real_heights, syn_heights, left_label="Real", right_label="Synthetic", ...)
```

La distribución del objetivo Real vs Balanced también usa la misma utilidad, guardando `results/graphics/compare/compare_survived.png`:

```300:317:src/analysis/analyzer.py
_plot_side_by_side_bars(
    all_t,
    [float(v) for v in real_t.values],
    [float(v) for v in bal_t.values],
    left_label="Real",
    right_label="Balanced",
    title=f"Distribución objetivo: {target_column} (real vs balanced)",
    filepath=os.path.join(compare_dir, f"compare_{target_column}.png"),
)
```

Los resultados cuantitativos de la comparación se serializan a `results/reports/comparison_report.json` para su análisis detallado.
