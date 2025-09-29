import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import os
import logging
from typing import Dict, List, Optional


def analyze_data(
    df: pd.DataFrame, target_column: str, graphics_dir: str = "results/graphics/"
) -> tuple[pd.DataFrame, dict]:
    """Realiza análisis básico y balanceo por undersampling.
    Guarda gráﬁcos en outputs/graphics.
    """
    report: dict[str, object] = {}
    os.makedirs(graphics_dir, exist_ok=True)

    try:
        # Análisis inicial
        class_dist = df[target_column].value_counts().to_dict()
        minority_class = min(class_dist.items(), key=lambda kv: kv[1])[0]
        imbalance_ratio = float(min(class_dist.values()) / max(class_dist.values()))
        report["class_analysis"] = {
            "original_distribution": class_dist,
            "minority_class": minority_class,
            "imbalance_ratio": imbalance_ratio,
        }

        # Gráfico inicial
        plot_class_distribution(
            df[target_column], os.path.join(graphics_dir, "original_dist.png")
        )

        # Balancear datos
        X = df.drop(columns=[target_column])
        y = df[target_column]

        undersampler = RandomUnderSampler(sampling_strategy="all", random_state=42)
        X_res, y_res = undersampler.fit_resample(X, y)
        balanced_df = pd.DataFrame(X_res, columns=X.columns)
        balanced_df[target_column] = y_res

        # Reporte de balanceo
        report["balancing"] = {
            "method": "RandomUnderSampler",
            "new_distribution": Counter(y_res),
            "samples_removed": int(len(y) - len(y_res)),
        }

        # Gráfico balanceado
        plot_class_distribution(y_res, os.path.join(graphics_dir, "balanced_dist.png"))

        return balanced_df, dict(report)

    except Exception as e:
        report["error"] = str(e)
        raise RuntimeError(f"Error en analyzer: {str(e)}")


def plot_class_distribution(series: pd.Series, filepath: str) -> None:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=series)
    plt.title("Distribución de Clases")
    plt.xticks(rotation=45)
    plt.savefig(filepath)
    plt.close()


def _plot_side_by_side_bars(
    labels: List[str],
    left_values: List[float],
    right_values: List[float],
    *,
    left_label: str,
    right_label: str,
    title: str,
    filepath: str,
    rotation: int = 45,
    ha: str = "right",
) -> None:
    """Utilidad estándar para gráficas comparativas de barras.

    Asegura colores, layout y leyendas uniformes para cualquier comparación
    binaria (p. ej., Real vs Synthetic o Real vs Balanced).
    """
    # Paleta unificada
    COLOR_REAL = "#4e79a7"
    COLOR_SYNTHETIC = "#e15759"
    COLOR_BALANCED = "#59a14f"

    right_label_lower = right_label.lower()
    if right_label_lower == "synthetic":
        right_color = COLOR_SYNTHETIC
    elif right_label_lower == "balanced":
        right_color = COLOR_BALANCED
    else:
        right_color = "#9c755f"  # color neutro consistente para terceros

    plt.figure(figsize=(10, 5))
    x = list(range(len(labels)))
    plt.bar(
        [i - 0.2 for i in x], left_values, width=0.4, label=left_label, color=COLOR_REAL
    )
    plt.bar(
        [i + 0.2 for i in x],
        right_values,
        width=0.4,
        label=right_label,
        color=right_color,
    )
    plt.xticks(x, labels, rotation=rotation, ha=ha)
    plt.ylabel("Proporción")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def compare_real_vs_synthetic(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    graphics_dir: str = "results/graphics/",
    *,
    target_column: Optional[str] = None,
    real_target: Optional[pd.Series] = None,
    balanced_target: Optional[pd.Series] = None,
    max_categories: int = 30,
    skip_id_like: bool = True,
    drift_warn_threshold: float = 0.3,
) -> dict:
    """Compara datos reales y sintéticos.

    - Guarda gráficos en subcarpeta graphics/compare
    - Numéricas: histogramas con KDE y estadísticas (media, std)
    - Categóricas: barras por frecuencia. Si en sintético vienen en one-hot, se
      agregan por prefijo "col_".
    """
    compare_dir = os.path.join(graphics_dir, "compare")
    os.makedirs(compare_dir, exist_ok=True)

    logger = logging.getLogger("analysis")
    comparison_report: Dict[str, dict] = {}

    # 1) Variables numéricas comunes
    common_numeric: List[str] = []
    for col in real_df.columns:
        if col not in synthetic_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(real_df[col]):
            continue
        if skip_id_like:
            name_l = str(col).lower()
            id_name_patterns = ["id", "ticket", "name", "cabin", "passengerid"]
            if any(p in name_l for p in id_name_patterns):
                logger.info(f"Omitiendo columna numérica tipo ID: {col}")
                continue
            nunique_ratio = float(real_df[col].nunique()) / max(
                1.0, float(len(real_df))
            )
            if nunique_ratio > 0.9:
                logger.info(f"Omitiendo columna numérica de alta unicidad: {col}")
                continue
        common_numeric.append(col)

    for col in common_numeric:
        plt.figure(figsize=(10, 5))
        # Bins comunes en rango combinado para comparación justa
        real_vals = real_df[col].astype(float)
        syn_vals = synthetic_df[col].astype(float)
        vmin = float(min(real_vals.min(), syn_vals.min()))
        vmax = float(max(real_vals.max(), syn_vals.max()))
        bins = 30

        real_counts, edges = np.histogram(real_vals, bins=bins, range=(vmin, vmax))
        syn_counts, _ = np.histogram(syn_vals, bins=bins, range=(vmin, vmax))
        real_props = real_counts / max(1, len(real_vals))
        syn_props = syn_counts / max(1, len(syn_vals))
        centers = (edges[:-1] + edges[1:]) / 2.0
        width = (edges[1] - edges[0]) * 0.9

        plt.bar(
            centers - width * 0.25,
            real_props,
            width=width * 0.5,
            label="Real",
            color="#4e79a7",
        )
        plt.bar(
            centers + width * 0.25,
            syn_props,
            width=width * 0.5,
            label="Synthetic",
            color="#e15759",
        )
        plt.title(f"Proporción por bin: {col}")
        plt.ylabel("Proporción")
        plt.xlabel(col)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(compare_dir, f"compare_{col}.png"))
        plt.close()

        comp_entry = {
            "type": "numeric",
            "real_mean": float(real_df[col].mean()),
            "synthetic_mean": float(synthetic_df[col].mean()),
            "real_std": float(real_df[col].std()),
            "synthetic_std": float(synthetic_df[col].std()),
        }
        # Validación simple de drift
        real_mean = comp_entry["real_mean"]
        syn_mean = comp_entry["synthetic_mean"]
        real_std = comp_entry["real_std"] or 1e-8
        syn_std = comp_entry["synthetic_std"] or 1e-8
        rel_mean_diff = abs(syn_mean - real_mean) / (abs(real_mean) + 1e-8)
        std_ratio = abs(syn_std - real_std) / (real_std)
        comp_entry["rel_mean_diff"] = float(rel_mean_diff)
        comp_entry["std_rel_diff"] = float(std_ratio)
        if rel_mean_diff > drift_warn_threshold or std_ratio > drift_warn_threshold:
            logger.warning(
                f"Alto drift en '{col}': rel_mean_diff={rel_mean_diff:.2f}, std_rel_diff={std_ratio:.2f}"
            )

        comparison_report[col] = comp_entry

    # 2) Variables categóricas: buscar columnas object en real_df
    categorical_cols: List[str] = list(real_df.select_dtypes(include="object").columns)

    for col in categorical_cols:
        real_counts = real_df[col].astype(str).value_counts(normalize=True)

        # Identificar si en sintético existe one-hot de este col (prefijo "col_")
        prefix = f"{col}_"
        oh_cols = [c for c in synthetic_df.columns if c.startswith(prefix)]

        if len(oh_cols) == 0:
            # Si no hay one-hot, intentar que exista la misma columna categórica
            if col in synthetic_df.columns:
                syn_counts = synthetic_df[col].astype(str).value_counts(normalize=True)
            else:
                # No hay forma de comparar; continuar
                logger.info(
                    f"Omitiendo categórica sin correspondencia en sintético: {col}"
                )
                continue
        else:
            # Mapear one-hot a categorías
            # Para probabilidad por categoría, tomar media de la columna binaria
            syn_counts = (
                synthetic_df[oh_cols].mean().rename(lambda x: x.replace(prefix, ""))
            )

        # Alinear índices
        all_cats = sorted(
            set(real_counts.index.tolist()) | set(syn_counts.index.tolist())
        )
        if len(all_cats) > max_categories:
            logger.info(
                f"Omitiendo categórica con muchas categorías ({len(all_cats)}): {col}"
            )
            continue
        real_aligned = real_counts.reindex(all_cats, fill_value=0.0)
        syn_aligned = syn_counts.reindex(all_cats, fill_value=0.0)

        # Gráfico de barras comparativo (unificado)
        real_heights = [float(v) for v in real_aligned.values]
        syn_heights = [float(v) for v in syn_aligned.values]
        _plot_side_by_side_bars(
            all_cats,
            real_heights,
            syn_heights,
            left_label="Real",
            right_label="Synthetic",
            title=f"Proporción por categoría: {col}",
            filepath=os.path.join(compare_dir, f"compare_{col}.png"),
            rotation=45,
            ha="right",
        )

        comparison_report[col] = {
            "type": "categorical",
            "categories": all_cats,
            "real_freq": {k: float(v) for k, v in zip(all_cats, real_aligned.values)},
            "synthetic_freq": {
                k: float(v) for k, v in zip(all_cats, syn_aligned.values)
            },
        }

    # 3) Comparación del target (real vs balanced) si se provee
    if (
        target_column is not None
        and real_target is not None
        and balanced_target is not None
    ):
        try:
            real_t = real_target.astype(str).value_counts(normalize=True)
            bal_t = balanced_target.astype(str).value_counts(normalize=True)
            all_t = sorted(set(real_t.index.tolist()) | set(bal_t.index.tolist()))
            real_t = real_t.reindex(all_t, fill_value=0.0)
            bal_t = bal_t.reindex(all_t, fill_value=0.0)

            # Gráfico objetivo (unificado con la misma utilidad)
            _plot_side_by_side_bars(
                all_t,
                [float(v) for v in real_t.values],
                [float(v) for v in bal_t.values],
                left_label="Real",
                right_label="Balanced",
                title=f"Distribución objetivo: {target_column} (real vs balanced)",
                filepath=os.path.join(compare_dir, f"compare_{target_column}.png"),
                rotation=0,
                ha="center",
            )

            comparison_report[target_column] = {
                "type": "target",
                "categories": all_t,
                "real_freq": {k: float(v) for k, v in zip(all_t, real_t.values)},
                "balanced_freq": {k: float(v) for k, v in zip(all_t, bal_t.values)},
            }
        except Exception as e:
            logger.warning(f"No se pudo comparar target '{target_column}': {e}")

    return comparison_report


__all__ = [
    "analyze_data",
    "compare_real_vs_synthetic",
    "plot_class_distribution",
]
