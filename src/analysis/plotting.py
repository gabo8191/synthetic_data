"""
Módulo de visualización para análisis de datos.

Este módulo contiene todas las funciones relacionadas con la generación
de gráficos y visualizaciones para el análisis de datos.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List


def plot_class_distribution(series: pd.Series, filepath: str) -> None:
    """
    Genera un gráfico de distribución de clases.

    Args:
        series: Serie con las clases a visualizar
        filepath: Ruta donde guardar el gráfico
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=series)
    plt.title("Distribución de Clases")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_side_by_side_bars(
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
    """
    Utilidad estándar para gráficas comparativas de barras.

    Asegura colores, layout y leyendas uniformes para cualquier comparación
    binaria (p. ej., Real vs Synthetic o Real vs Balanced).

    Args:
        labels: Lista de etiquetas para el eje X
        left_values: Valores para las barras izquierdas
        right_values: Valores para las barras derechas
        left_label: Etiqueta para las barras izquierdas
        right_label: Etiqueta para las barras derechas
        title: Título del gráfico
        filepath: Ruta donde guardar el gráfico
        rotation: Rotación de las etiquetas del eje X
        ha: Alineación horizontal de las etiquetas
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
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_numeric_comparison(
    real_values: pd.Series,
    synthetic_values: pd.Series,
    column_name: str,
    filepath: str,
    bins: int = 30,
) -> None:
    """
    Genera un histograma comparativo para variables numéricas.

    Args:
        real_values: Valores reales
        synthetic_values: Valores sintéticos
        column_name: Nombre de la columna
        filepath: Ruta donde guardar el gráfico
        bins: Número de bins para el histograma
    """
    plt.figure(figsize=(10, 5))

    # Bins comunes en rango combinado para comparación justa
    vmin = float(min(real_values.min(), synthetic_values.min()))
    vmax = float(max(real_values.max(), synthetic_values.max()))

    real_counts, edges = np.histogram(real_values, bins=bins, range=(vmin, vmax))
    syn_counts, _ = np.histogram(synthetic_values, bins=bins, range=(vmin, vmax))
    real_props = real_counts / max(1, len(real_values))
    syn_props = syn_counts / max(1, len(synthetic_values))
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
    plt.title(f"Proporción por bin: {column_name}")
    plt.ylabel("Proporción")
    plt.xlabel(column_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_categorical_comparison(
    real_frequencies: pd.Series,
    synthetic_frequencies: pd.Series,
    column_name: str,
    filepath: str,
) -> None:
    """
    Genera un gráfico de barras comparativo para variables categóricas.

    Args:
        real_frequencies: Frecuencias reales
        synthetic_frequencies: Frecuencias sintéticas
        column_name: Nombre de la columna
        filepath: Ruta donde guardar el gráfico
    """
    # Alinear índices
    all_cats = sorted(
        set(real_frequencies.index.tolist()) | set(synthetic_frequencies.index.tolist())
    )
    real_aligned = real_frequencies.reindex(all_cats, fill_value=0.0)
    syn_aligned = synthetic_frequencies.reindex(all_cats, fill_value=0.0)

    # Gráfico de barras comparativo
    real_heights = [float(v) for v in real_aligned.values]
    syn_heights = [float(v) for v in syn_aligned.values]

    plot_side_by_side_bars(
        all_cats,
        real_heights,
        syn_heights,
        left_label="Real",
        right_label="Synthetic",
        title=f"Proporción por categoría: {column_name}",
        filepath=filepath,
        rotation=45,
        ha="right",
    )


def plot_target_comparison(
    real_target: pd.Series,
    synthetic_target: pd.Series,
    target_column: str,
    filepath: str,
) -> None:
    """
    Genera un gráfico comparativo para la variable objetivo.

    Args:
        real_target: Valores reales del target
        synthetic_target: Valores sintéticos del target
        target_column: Nombre de la columna objetivo
        filepath: Ruta donde guardar el gráfico
    """
    real_t = real_target.astype(str).value_counts(normalize=True)
    syn_t = synthetic_target.astype(str).value_counts(normalize=True)
    all_t = sorted(set(real_t.index.tolist()) | set(syn_t.index.tolist()))
    real_t = real_t.reindex(all_t, fill_value=0.0)
    syn_t = syn_t.reindex(all_t, fill_value=0.0)

    # Gráfico objetivo: Real vs Synthetic
    plot_side_by_side_bars(
        all_t,
        [float(v) for v in real_t.values],
        [float(v) for v in syn_t.values],
        left_label="Real",
        right_label="Synthetic",
        title=f"Distribución objetivo: {target_column} (real vs synthetic)",
        filepath=filepath,
        rotation=0,
        ha="center",
    )


def plot_correlation_matrix(
    correlation_matrix: pd.DataFrame,
    filepath: str,
    title: str = "Matriz de Correlación",
) -> None:
    """
    Genera un gráfico de matriz de correlación.

    Args:
        correlation_matrix: Matriz de correlación
        filepath: Ruta donde guardar el gráfico
        title: Título del gráfico
    """
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
    )
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_survived_correlation(
    survived_corr: pd.Series,
    filepath: str,
    title: str = "Correlación de Variables con la Supervivencia",
) -> None:
    """
    Genera un gráfico de barras con correlaciones de supervivencia.

    Args:
        survived_corr: Serie con correlaciones de supervivencia
        filepath: Ruta donde guardar el gráfico
        title: Título del gráfico
    """
    plt.figure(figsize=(10, 6))
    colors = ["red" if x < 0 else "blue" for x in survived_corr.values]
    bars = plt.bar(
        range(len(survived_corr)),
        survived_corr.values.astype(float),
        color=colors,
        alpha=0.7,
    )
    plt.xticks(range(len(survived_corr)), survived_corr.index, rotation=45, ha="right")
    plt.ylabel("Correlación con Survived")
    plt.title(title, fontsize=14, pad=20)
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    plt.grid(axis="y", alpha=0.3)

    # Añadir valores en las barras
    for i, (bar, value) in enumerate(zip(bars, survived_corr.values)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.01 if value > 0 else -0.03),
            f"{value:.3f}",
            ha="center",
            va="bottom" if value > 0 else "top",
        )

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


__all__ = [
    "plot_class_distribution",
    "plot_side_by_side_bars",
    "plot_numeric_comparison",
    "plot_categorical_comparison",
    "plot_target_comparison",
    "plot_correlation_matrix",
    "plot_survived_correlation",
]
