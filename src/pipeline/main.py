import pandas as pd
import numpy as np
import json
import os
import logging
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_and_clean_data
from src.models.gan import simple_gan_generator
from src.analysis.analyzer import analyze_data, compare_real_vs_synthetic


def _configure_logging(reports_dir: str) -> None:
    os.makedirs(reports_dir, exist_ok=True)
    log_path = os.path.join(reports_dir, "pipeline.log")

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


def main() -> None:
    # Configuración de rutas
    INPUT_FILE = "data/Titanic-Dataset.csv"
    OUTPUT_DIR = "results/data/"
    GRAPHICS_DIR = "results/graphics/"
    REPORTS_DIR = "results/reports/"
    TARGET_COL = "survived"  # objetivo en Titanic

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(GRAPHICS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    _configure_logging(REPORTS_DIR)
    logger = logging.getLogger("pipeline")

    try:
        # 1. Carga y limpieza
        logger.info("Cargando datos...")
        df, clean_report = load_and_clean_data(INPUT_FILE, TARGET_COL)

        # Reporte en JSON serializable
        clean_report = {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in clean_report.items()
        }

        df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_dataset.csv"), index=False)
        with open(os.path.join(REPORTS_DIR, "cleaning_report.json"), "w") as f:
            json.dump(clean_report, f, indent=4, default=str)

        # 2. Análisis y balanceo
        logger.info("Analizando datos...")
        balanced_df, analysis_report = analyze_data(df, TARGET_COL, GRAPHICS_DIR)

        # Reporte en JSON serializable
        analysis_report = {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in analysis_report.items()
        }

        balanced_df.to_csv(os.path.join(OUTPUT_DIR, "balanced_data.csv"), index=False)
        with open(os.path.join(REPORTS_DIR, "analysis_report.json"), "w") as f:
            json.dump(analysis_report, f, indent=4, default=str)

        # 3. Resumen
        logger.info("=== RESUMEN FINAL ===")
        logger.info(
            f"Clases originales: {analysis_report['class_analysis']['original_distribution']}"
        )
        logger.info(f"Método balanceo: {analysis_report['balancing']['method']}")
        logger.info(
            f"Nuevo distribución: {analysis_report['balancing']['new_distribution']}"
        )

        logger.info("Generando datos sintéticos con GAN (estratificado por clase)...")

        features_df = balanced_df.drop(columns=[TARGET_COL]).copy()

        # Remover columnas tipo ID antes de generar
        id_like_candidates = ["passengerid", "name", "ticket", "cabin"]
        dropped_id_like = [c for c in id_like_candidates if c in features_df.columns]
        if len(dropped_id_like) > 0:
            logger.info(
                f"Removiendo columnas tipo ID antes de generar: {dropped_id_like}"
            )
            features_df.drop(columns=dropped_id_like, inplace=True)

        # OneHotEncoding para categóricas
        categorical_cols = list(features_df.select_dtypes(include="object").columns)
        if len(categorical_cols) > 0:
            features_df = pd.get_dummies(features_df, columns=categorical_cols)

        # Escalar numéricas continuas (no escalar dummies 0/1)
        numeric_cols = list(features_df.select_dtypes(include=np.number).columns)
        continuous_cols = []
        for col in numeric_cols:
            uniques = set(pd.Series(features_df[col].dropna().unique()).tolist())
            if uniques.issubset({0, 1}):
                continue
            continuous_cols.append(col)

        scaler = None
        scaled_features_df = features_df.copy()
        if len(continuous_cols) > 0:
            scaler = StandardScaler()
            scaled_features_df[continuous_cols] = scaler.fit_transform(
                features_df[continuous_cols]
            )
            logger.info(
                f"Columnas continuas escaladas: {len(continuous_cols)} ({continuous_cols[:8]}{'...' if len(continuous_cols) > 8 else ''})"
            )

        # Generación
        synthetic_scaled_df = simple_gan_generator(
            scaled_features_df,
            cantidad=1000,
            epochs=1400,
            batch_size=64,
            stratified=True,
            labels=balanced_df[TARGET_COL],
        )

        # Inversa del escalado para columnas continuas
        synthetic_df = synthetic_scaled_df.copy()
        if scaler is not None and len(continuous_cols) > 0:
            synthetic_df[continuous_cols] = scaler.inverse_transform(
                synthetic_scaled_df[continuous_cols]
            )

        # Guardar datos sintéticos
        synthetic_df.to_csv(os.path.join(OUTPUT_DIR, "synthetic_data.csv"), index=False)
        logger.info("Datos sintéticos generados y guardados.")

        # Comparar datos reales vs sintéticos (numéricas y categóricas) + target
        logger.info("Comparando datos reales vs. sintéticos...")
        comparison_report = compare_real_vs_synthetic(
            df,
            synthetic_df,
            GRAPHICS_DIR,
            target_column=TARGET_COL,
            real_target=df[TARGET_COL],
            balanced_target=balanced_df[TARGET_COL],
            max_categories=30,
            skip_id_like=True,
            drift_warn_threshold=0.3,
        )

        # Guardar reporte de comparación
        with open(
            os.path.join(REPORTS_DIR, "comparison_report.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(comparison_report, f, indent=4, ensure_ascii=False)
        logger.info("Reporte de comparación guardado en reports/comparison_report.json")

    except Exception as e:
        logger.exception(f"ERROR en pipeline: {str(e)}")


__all__ = ["main"]
