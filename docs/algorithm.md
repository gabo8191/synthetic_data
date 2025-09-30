# Algoritmo y pipeline de generación de datos sintéticos

Este documento describe el flujo end‑to‑end, las decisiones de preprocesamiento y el funcionamiento del GAN (Generative Adversarial Network = red generativa adversaria) utilizado para generar datos sintéticos del Titanic. Incluyendo referencias directas al código fuente y a los artefactos generados en `results/`.

## 1) Carga y limpieza del dataset

La carga y limpieza residen en `src/data/loader.py`. Se convierten tipos numéricos y se imputan nulos con mediana/moda. También se eliminan duplicados y se registra un reporte completo.

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

### 4) División de datos y entrenamiento

**División 80-20**: El proyecto implementa una división estándar de datos:

- **Datos de entrenamiento**: 80% (712 muestras)
- **Datos de prueba**: 20% (179 muestras)
- **División estratificada**: Mantiene la proporción de clases en ambos conjuntos

**Proceso de división:**

```87:105:src/pipeline/main.py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Distribución final:**

- **Datos de entrenamiento**: 712 muestras (80% del dataset original)
- **Datos de prueba**: 179 muestras (20% del dataset original)
- **Datos balanceados**: 546 muestras (273 por clase, solo datos de entrenamiento)
- **Datos sintéticos generados**: 1,000 muestras (entrenados con datos balanceados)

Esta división permite evaluar el rendimiento del generador en datos no vistos durante el entrenamiento.

### 5) Arquitectura del GAN y algoritmos de entrenamiento

#### 5.1) Arquitectura de las redes

**Generador (Generator)**:

- **Entrada**: Vector de ruido aleatorio (latent_dim=100)
- **Arquitectura**: Red densa (fully-connected) con 2 capas
- **Capas ocultas**: 128 neuronas con activación ReLU
- **Capa de salida**: input_dim neuronas con activación Tanh
- **Propósito**: Transformar ruido aleatorio en datos sintéticos

```24:41:src/models/gan.py
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),    # 100 → 128
            nn.ReLU(),                     # Activación no lineal
            nn.Linear(128, input_dim),     # 128 → dimensiones de datos
            nn.Tanh(),                     # Salida en [-1, 1]
        )
```

**Discriminador (Discriminator)**:

- **Entrada**: Datos reales o sintéticos
- **Arquitectura**: Red densa con 2 capas
- **Capas ocultas**: 128 neuronas con activación ReLU
- **Capa de salida**: 1 neurona con activación Sigmoid
- **Propósito**: Clasificar si los datos son reales (1) o sintéticos (0)

```42:55:src/models/gan.py
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),     # Datos → 128
            nn.ReLU(),                     # Activación no lineal
            nn.Linear(128, 1),             # 128 → 1
            nn.Sigmoid(),                  # Probabilidad [0, 1]
        )
```

#### 5.2) Normalización de datos

Los datos se normalizan a [-1, 1] antes del entrenamiento para que la función Tanh opere en su rango efectivo:

```24:41:src/models/gan.py
# Normalización a [-1, 1] para que Tanh funcione adecuadamente
min_vals = data.min(axis=0)
max_vals = data.max(axis=0)
data = 2 * (data - min_vals) / (max_vals - min_vals + 1e-8) - 1
```

**Proceso de normalización**:

1. **Cálculo de rangos**: min_vals y max_vals por columna
2. **Escalado lineal**: Mapea [min, max] → [0, 1]
3. **Centrado**: Mapea [0, 1] → [-1, 1]
4. **Epsilon**: Evita división por cero (1e-8)

#### 5.3) Algoritmo de entrenamiento adversarial

El entrenamiento sigue el algoritmo minimax de GANs con actualizaciones alternadas:

**Paso 1: Entrenamiento del Discriminador**

```python
# Generar datos sintéticos
z = torch.randn(batch_size, latent_dim)
fake_data = generator(z)

# Calcular pérdida del discriminador
real_preds = discriminator(real_batch)      # Predicciones para datos reales
fake_preds = discriminator(fake_data.detach())  # Predicciones para datos sintéticos

# Pérdida: BCE para datos reales (target=1) + BCE para datos sintéticos (target=0)
loss_D = BCELoss(real_preds, ones) + BCELoss(fake_preds, zeros)

# Actualizar discriminador
discriminator_optimizer.zero_grad()
loss_D.backward()
discriminator_optimizer.step()
```

**Paso 2: Entrenamiento del Generador**

```python
# Generar nuevos datos sintéticos
z = torch.randn(batch_size, latent_dim)
fake_data = generator(z)

# Calcular pérdida del generador
fake_preds = discriminator(fake_data)

# Pérdida: BCE para datos sintéticos (target=1) - quiere engañar al discriminador
loss_G = BCELoss(fake_preds, ones)

# Actualizar generador
generator_optimizer.zero_grad()
loss_G.backward()
generator_optimizer.step()
```

#### 5.4) Entrenamiento estratificado

**Configuración actual**: El proyecto utiliza entrenamiento estratificado por defecto (`STRATIFIED_TRAINING: True`), entrenando un GAN separado por cada clase del objetivo:

```python
# Configuración en src/config.py
STRATIFIED_TRAINING: bool = True
EPOCHS: int = 1400  # Épocas por clase
SYNTHETIC_SAMPLES: int = 1000  # Total (500 por clase)
```

**Proceso de entrenamiento estratificado**:

1. **Identificación de clases**: Se identifican las clases únicas en el dataset balanceado
2. **Entrenamiento por clase**: Se entrena un GAN separado para cada clase
3. **Generación balanceada**: Se generan 500 muestras por clase
4. **Concatenación**: Se combinan los resultados en un dataset final

```python
# Implementación en src/models/generation.py
def generate_stratified_data(df, cantidad, labels, epochs, batch_size, latent_dim):
    unique_classes = sorted(pd.Series(labels).unique())
    per_class = cantidad // len(unique_classes)

    for cls in unique_classes:
        # Entrenar GAN específico para esta clase
        class_data = df[labels == cls]
        synthetic_class = train_gan_for_class(class_data, per_class, epochs, ...)
        generated_parts.append(synthetic_class)

    return pd.concat(generated_parts, axis=0, ignore_index=True)
```

**Ventajas del entrenamiento estratificado**:

- **Preservación de dependencias**: Cada GAN aprende la distribución específica de su clase
- **Balanceo automático**: Genera exactamente 500 muestras por clase (50%-50%)
- **Mejor calidad**: Evita que una clase domine el entrenamiento
- **Especialización**: Cada GAN se especializa en patrones específicos de supervivencia

**Configuración de épocas**:

- **Épocas por clase**: 1400 épocas
- **Total de épocas**: 2800 épocas (1400 × 2 clases)
- **Tiempo de entrenamiento**: Aproximadamente 2-3 minutos por clase
- **Logging**: Se registra el progreso cada 100 épocas por clase

#### 5.5) Generación y desnormalización

La generación final desnormaliza los datos de vuelta al dominio original:

```96:107:src/models/gan.py
with torch.no_grad():
    z = torch.randn(cantidad, latent_dim).to(device)
    synthetic_data = generator(z).cpu().numpy()

# Desnormaliza a rango original por columna
synthetic_data = (synthetic_data + 1) / 2  # [-1, 1] → [0, 1]
synthetic_data = synthetic_data * (max_vals - min_vals + 1e-8) + min_vals  # [0, 1] → [min, max]
df_sintetico = pd.DataFrame(synthetic_data, columns=df.columns)
```

**Proceso de desnormalización**:

1. **Generación**: Ruido → Datos sintéticos en [-1, 1]
2. **Escalado**: [-1, 1] → [0, 1]
3. **Mapeo**: [0, 1] → [min_original, max_original]
4. **DataFrame**: Conversión a formato tabular

El CSV resultante se guarda en `results/data/synthetic_data.csv`.

### 6) Volumen de datos sintéticos generados

El pipeline está configurado para generar **1,000 datos sintéticos**, superando significativamente los **891 datos originales** del dataset Titanic. Esta configuración se establece en el pipeline principal:

```163:177:src/pipeline/main.py
synthetic_scaled_df = simple_gan_generator(
    features_df,
    cantidad=config.gan.SYNTHETIC_SAMPLES,  # 1000 datos sintéticos
    epochs=config.gan.EPOCHS,              # 1400 épocas por clase
    batch_size=config.gan.BATCH_SIZE,      # 64
    stratified=config.gan.STRATIFIED_TRAINING,  # True
    labels=balanced_df[config.data.TARGET_COLUMN],
)
```

**Comparación de volúmenes:**

- **Datos originales**: 891 registros
- **Datos limpios**: 891 registros (sin pérdida por limpieza)
- **Datos de entrenamiento**: 712 registros (80%)
- **Datos de prueba**: 179 registros (20%)
- **Datos balanceados**: 546 registros (273 por clase, solo entrenamiento)
- **Datos sintéticos**: 1,000 registros (+12.2% vs originales)

**Distribución de datos sintéticos:**

- **Clase 0 (No sobrevivientes)**: 500 muestras (50%)
- **Clase 1 (Sobrevivientes)**: 500 muestras (50%)
- **Balanceo perfecto**: Mantiene la distribución 50%-50% del dataset balanceado

Esta estrategia de generación superior al volumen original permite:

- Compensar la pérdida de datos por balanceo (166 muestras removidas)
- Proporcionar mayor variabilidad para entrenar modelos downstream
- Mejorar la representatividad estadística del dataset sintético
- Mantener balanceo perfecto entre clases

### 7) Arquitectura modular y orquestadores

**Refactorización implementada**: El proyecto ha sido refactorizado en una arquitectura modular con orquestadores:

#### 7.1) Orquestador de Análisis (`src/analysis/analyzer.py`)

Coordina funciones especializadas de análisis:

```python
def analyze_data(df: pd.DataFrame, target_column: str, graphics_dir: str) -> Dict:
    """Orquestador principal de análisis de datos"""
    # 1. Análisis de distribución de clases
    class_analysis = analyze_class_distribution(df, target_column)

    # 2. Balanceo de datos
    balanced_df, balancing_info = balance_dataset(df, target_column)

    # 3. Generación de gráficos
    plot_class_distribution(df[target_column], f"{graphics_dir}/original_dist.png")
    plot_class_distribution(balanced_df[target_column], f"{graphics_dir}/balanced_dist.png")

    return {
        "class_analysis": class_analysis,
        "balancing": balancing_info,
        "validation": validate_balancing_result(df, balanced_df, target_column)
    }
```

#### 7.2) Orquestador de GAN (`src/models/gan.py`)

Coordina la generación de datos sintéticos:

```python
def simple_gan_generator(df, cantidad, epochs, batch_size, latent_dim, *, labels=None, stratified=False):
    """Orquestador principal de generación GAN"""
    if stratified and labels is not None:
        # Entrenamiento estratificado por clase
        return generate_stratified_data(df, cantidad, labels, epochs, batch_size, latent_dim)
    else:
        # Entrenamiento estándar
        return generate_standard_data(df, cantidad, epochs, batch_size, latent_dim)
```

#### 7.3) Módulos especializados

- **`plotting.py`**: Funciones de visualización centralizadas
- **`balancing.py`**: Lógica de balanceo de clases
- **`comparison.py`**: Comparación real vs sintético
- **`metrics.py`**: Cálculo de métricas de evaluación
- **`gan_models.py`**: Arquitecturas de redes neuronales
- **`training.py`**: Lógica de entrenamiento
- **`data_processing.py`**: Preprocesamiento específico para GAN
- **`generation.py`**: Estrategias de generación

### 8) Análisis de correlación y matriz de correlación

El proyecto incluye un análisis completo de correlaciones entre variables, implementado en `src/analysis/metrics.py`:

```python
def generate_correlation_analysis(df: pd.DataFrame, graphics_dir: str) -> Dict:
    """Genera análisis completo de correlación"""
    # 1. Generar reporte de correlación
    correlation_report = generate_correlation_report(df, graphics_dir)

    # 2. Calcular matriz de correlación
    correlation_matrix, _ = calculate_correlation_matrix(df)

    # 3. Generar gráficos
    plot_correlation_matrix(correlation_matrix, f"{graphics_dir}/correlation_matrix.png")
    plot_survived_correlation(survived_corr, f"{graphics_dir}/survived_correlation.png")

    return correlation_report
```

**Gráficos generados**:

- `results/graphics/correlation_matrix.png`: Matriz de correlación completa con heatmap
- `results/graphics/survived_correlation.png`: Correlación específica de variables con supervivencia

**Análisis de patrones**:

- Identificación de correlaciones fuertes (|r| > 0.5)
- Análisis de variables más correlacionadas con supervivencia
- Mapeo de encoders para variables categóricas

### 9) Comparación y generación de gráficos unificada

Las comparaciones se realizan en `src/analysis/comparison.py`. Para barras lado a lado se utiliza una única utilidad para garantizar estilo consistente y evitar discrepancias entre figuras:

```python
def plot_side_by_side_bars(categories, left_values, right_values, left_label, right_label, title, filepath):
    """Función unificada para gráficos de barras lado a lado"""
    COLOR_REAL = "#4e79a7"
    COLOR_SYNTHETIC = "#e15759"
    COLOR_BALANCED = "#59a14f"

    x = range(len(categories))
    plt.bar([i - 0.2 for i in x], left_values, width=0.4, label=left_label, color=COLOR_REAL)
    plt.bar([i + 0.2 for i in x], right_values, width=0.4, label=right_label, color=COLOR_SYNTHETIC)
    plt.xticks(x, categories)
    plt.title(title)
    plt.legend()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
```

**Gráficos de comparación generados**:

#### Variables numéricas

- `results/graphics/compare/compare_age.png`: Comparación de distribuciones de edad
- `results/graphics/compare/compare_fare.png`: Comparación de tarifas de pasaje
- `results/graphics/compare/compare_parch.png`: Comparación de padres/hijos a bordo
- `results/graphics/compare/compare_pclass.png`: Comparación de clase de pasaje
- `results/graphics/compare/compare_sibsp.png`: Comparación de hermanos/cónyuges a bordo

Para numéricas, se calculan histogramas con mismos bordes de bin (bin = intervalo) para una comparación justa:

```169:205:src/analysis/analyzer.py
real_counts, edges = np.histogram(real_vals, bins=bins, range=(vmin, vmax))
syn_counts, _ = np.histogram(syn_vals, bins=bins, range=(vmin, vmax))
real_props = real_counts / max(1, len(real_vals))
syn_props = syn_counts / max(1, len(syn_vals))
...
plt.savefig(os.path.join(compare_dir, f"compare_{col}.png"))
```

#### Variables categóricas

- `results/graphics/compare/compare_sex.png`: Comparación de género
- `results/graphics/compare/compare_embarked.png`: Comparación de puerto de embarque
- `results/graphics/compare/compare_survived.png`: Comparación de supervivencia (Real vs Synthetic)

Para categóricas, si los sintéticos están one-hot, se agregan con la media por prefijo, alineando categorías y usando la utilidad de barras:

```236:281:src/analysis/analyzer.py
prefix = f"{col}_"
oh_cols = [c for c in synthetic_df[oh_cols].mean().rename(lambda x: x.replace(prefix, "")))
...
_plot_side_by_side_bars(all_cats, real_heights, syn_heights, left_label="Real", right_label="Synthetic", ...)
```

### 10) Métricas de evaluación y porcentaje discriminador

El sistema calcula múltiples métricas para evaluar la calidad de los datos sintéticos:

**Métricas por variable**

- `rel_mean_diff`: Diferencia relativa de la media
- `std_rel_diff`: Diferencia relativa de la desviación estándar
- `drift`: Detección de deriva estadística

**Porcentaje discriminador general**

```python
# Implementación en src/analysis/metrics.py
def calculate_discrimination_percentage(comparison_report: Dict) -> Dict:
    """Calcula el porcentaje discriminador general"""
    total_variables = len([k for k in comparison_report.keys() if not k.startswith("_")])
    high_drift_variables = 0

    for var_name, var_data in comparison_report.items():
        if var_name.startswith("_"):
            continue

        # Validar que var_data sea un diccionario
        if not isinstance(var_data, dict):
            continue

        if var_data.get("type") == "numeric":
            rel_mean_diff = var_data.get("rel_mean_diff", 0)
            if rel_mean_diff > config.analysis.DRIFT_HIGH_THRESHOLD:  # 0.3
                high_drift_variables += 1

    discrimination_percentage = ((total_variables - high_drift_variables) / total_variables) * 100

    return {
        "total_variables": total_variables,
        "high_drift_variables": high_drift_variables,
        "discrimination_percentage": discrimination_percentage
    }
```

**Resultados actuales (última ejecución)**

- **Total de variables**: 9
- **Variables con drift alto**: 2 (sibsp, parch)
- **Porcentaje discriminador**: 77.78%

**Variables con problemas identificados**:

- **`sibsp`**: rel_mean_diff = 0.75 (75% de diferencia)
- **`parch`**: rel_mean_diff = 0.32 (32% de diferencia)

Los resultados cuantitativos de la comparación se serializan a `results/reports/comparison_report.json` para su análisis detallado.
