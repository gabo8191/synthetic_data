import pandas as pd
import logging
from typing import Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.config import config
from src.utils import (
    setup_logging,
    save_json_report,
    save_dataframe,
    identify_continuous_columns,
    identify_id_like_columns,
    log_dataframe_info,
)
from src.validation import (
    validate_configuration,
    validate_data_quality,
    ValidationError,
)
from src.data.loader import load_and_clean_data
from src.models.gan import simple_gan_generator
from src.analysis.analyzer import (
    analyze_data,
    compare_real_vs_synthetic,
    generate_correlation_analysis,
)


def _load_and_clean_data(logger: logging.Logger) -> tuple[pd.DataFrame, dict]:
    """Carga y limpia los datos del archivo de entrada."""
    logger.info("Cargando datos...")

    df, clean_report = load_and_clean_data(
        config.paths.INPUT_FILE, config.data.TARGET_COLUMN
    )

    # Validar calidad de datos
    quality_report = validate_data_quality(df, "Dataset limpio")
    clean_report["quality_report"] = quality_report

    # Guardar datos limpios
    save_dataframe(df, f"{config.paths.OUTPUT_DIR}/cleaned_dataset.csv")
    save_json_report(clean_report, f"{config.paths.REPORTS_DIR}/cleaning_report.json")

    log_dataframe_info(df, "Dataset limpio", logger)
    return df, clean_report


def _generate_correlation_analysis(logger: logging.Logger) -> dict:
    """Genera análisis de correlación."""
    logger.info("Generando análisis de correlación...")

    # Cargar datos limpios
    df = pd.read_csv(f"{config.paths.OUTPUT_DIR}/cleaned_dataset.csv")

    # Generar análisis completo de correlación
    correlation_report = generate_correlation_analysis(df, config.paths.GRAPHICS_DIR)

    # Guardar reporte
    save_json_report(
        correlation_report, f"{config.paths.REPORTS_DIR}/correlation_report.json"
    )

    return correlation_report


def _split_data(
    df: pd.DataFrame, logger: logging.Logger
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide los datos en entrenamiento y prueba."""
    logger.info("Dividiendo datos en entrenamiento (80%) y prueba (20%)...")

    X = df.drop(columns=[config.data.TARGET_COLUMN])
    y = df[config.data.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.data.TEST_SIZE,
        random_state=config.data.RANDOM_STATE,
        stratify=y,
    )

    # Reconstruir DataFrames
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    logger.info(f"Datos de entrenamiento: {len(train_df)} muestras")
    logger.info(f"Datos de prueba: {len(test_df)} muestras")

    # Guardar datos divididos
    save_dataframe(train_df, f"{config.paths.OUTPUT_DIR}/train_data.csv")
    save_dataframe(test_df, f"{config.paths.OUTPUT_DIR}/test_data.csv")

    return train_df, test_df


def _analyze_and_balance_data(
    train_df: pd.DataFrame, logger: logging.Logger
) -> tuple[pd.DataFrame, dict]:
    """Analiza y balancea los datos de entrenamiento."""
    logger.info("Analizando datos de entrenamiento...")

    balanced_df, analysis_report = analyze_data(
        train_df, config.data.TARGET_COLUMN, config.paths.GRAPHICS_DIR
    )

    # Guardar datos balanceados y reporte
    save_dataframe(balanced_df, f"{config.paths.OUTPUT_DIR}/balanced_data.csv")
    save_json_report(
        analysis_report, f"{config.paths.REPORTS_DIR}/analysis_report.json"
    )

    # Log resumen
    logger.info("=== RESUMEN FINAL ===")
    logger.info(
        f"Clases originales: {analysis_report['class_analysis']['original_distribution']}"
    )
    logger.info(f"Método balanceo: {analysis_report['balancing']['method']}")
    logger.info(
        f"Nueva distribución: {analysis_report['balancing']['new_distribution']}"
    )

    return balanced_df, analysis_report


def _prepare_features_for_gan(
    balanced_df: pd.DataFrame, logger: logging.Logger
) -> tuple[pd.DataFrame, Optional[StandardScaler], list]:
    """Prepara las características para el GAN."""
    logger.info("Preparando características para generación...")

    features_df = balanced_df.drop(columns=[config.data.TARGET_COLUMN]).copy()

    # Remover columnas tipo ID
    id_like_cols = identify_id_like_columns(features_df)
    if id_like_cols:
        logger.info(f"Removiendo columnas tipo ID: {id_like_cols}")
        features_df.drop(columns=id_like_cols, inplace=True)

    # One-Hot Encoding para categóricas
    categorical_cols = list(features_df.select_dtypes(include="object").columns)
    if categorical_cols:
        features_df = pd.get_dummies(features_df, columns=categorical_cols)

    # Escalar columnas continuas
    continuous_cols = identify_continuous_columns(features_df)
    scaler = None

    if continuous_cols:
        scaler = StandardScaler()
        features_df[continuous_cols] = scaler.fit_transform(
            features_df[continuous_cols]
        )
        logger.info(
            f"Columnas continuas escaladas: {len(continuous_cols)} ({continuous_cols[:8]}{'...' if len(continuous_cols) > 8 else ''})"
        )

    return features_df, scaler, continuous_cols


def _generate_synthetic_data(
    features_df: pd.DataFrame, balanced_df: pd.DataFrame, logger: logging.Logger
) -> pd.DataFrame:
    """Genera datos sintéticos usando GAN."""
    logger.info("Generando datos sintéticos con GAN (estratificado por clase)...")

    synthetic_scaled_df = simple_gan_generator(
        features_df,
        cantidad=config.gan.SYNTHETIC_SAMPLES,
        epochs=config.gan.EPOCHS,
        batch_size=config.gan.BATCH_SIZE,
        stratified=config.gan.STRATIFIED_TRAINING,
        labels=balanced_df[config.data.TARGET_COLUMN],
    )

    return synthetic_scaled_df


def _postprocess_synthetic_data(
    synthetic_scaled_df: pd.DataFrame,
    scaler: Optional[StandardScaler],
    continuous_cols: list,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Post-procesa los datos sintéticos."""
    synthetic_df = synthetic_scaled_df.copy()

    # Inversa del escalado para columnas continuas
    if scaler is not None and continuous_cols:
        synthetic_df[continuous_cols] = scaler.inverse_transform(
            synthetic_scaled_df[continuous_cols]
        )

    # Guardar datos sintéticos
    save_dataframe(synthetic_df, f"{config.paths.OUTPUT_DIR}/synthetic_data.csv")
    logger.info("Datos sintéticos generados y guardados.")

    return synthetic_df


def _compare_real_vs_synthetic(
    train_df: pd.DataFrame, synthetic_df: pd.DataFrame, logger: logging.Logger
) -> dict:
    """Compara datos reales vs sintéticos."""
    logger.info("Comparando datos reales vs. sintéticos...")

    # Generar target sintético basado en la distribución balanceada
    synthetic_target = train_df[config.data.TARGET_COLUMN].sample(
        n=len(synthetic_df), replace=True, random_state=config.data.RANDOM_STATE
    )

    comparison_report = compare_real_vs_synthetic(
        train_df,
        synthetic_df,
        config.paths.GRAPHICS_DIR,
        target_column=config.data.TARGET_COLUMN,
        real_target=train_df[config.data.TARGET_COLUMN],
        synthetic_target=synthetic_target,
        max_categories=config.analysis.MAX_CATEGORIES,
        skip_id_like=True,
        drift_warn_threshold=config.analysis.DRIFT_WARN_THRESHOLD,
    )

    # Guardar reporte de comparación
    save_json_report(
        comparison_report, f"{config.paths.REPORTS_DIR}/comparison_report.json"
    )
    logger.info("Reporte de comparación guardado en reports/comparison_report.json")

    return comparison_report


def main() -> None:
    """
    Función principal del pipeline de generación de datos sintéticos.

    Ejecuta el pipeline completo:
    1. Carga y limpieza de datos
    2. Análisis de correlación
    3. División de datos
    4. Análisis y balanceo
    5. Generación de datos sintéticos
    6. Comparación y evaluación
    """
    try:
        # Validar configuración
        validate_configuration()

        # Configurar logging
        logger = setup_logging(config.paths.REPORTS_DIR)
        logger.info("=== INICIANDO PIPELINE DE GENERACIÓN DE DATOS SINTÉTICOS ===")

        # 1. Carga y limpieza de datos
        df, clean_report = _load_and_clean_data(logger)

        # 2. Análisis de correlación
        correlation_report = _generate_correlation_analysis(logger)

        # 3. División de datos
        train_df, test_df = _split_data(df, logger)

        # 4. Análisis y balanceo
        balanced_df, analysis_report = _analyze_and_balance_data(train_df, logger)

        # 5. Preparación de características para GAN
        features_df, scaler, continuous_cols = _prepare_features_for_gan(
            balanced_df, logger
        )

        # 6. Generación de datos sintéticos
        synthetic_scaled_df = _generate_synthetic_data(features_df, balanced_df, logger)

        # 7. Post-procesamiento de datos sintéticos
        synthetic_df = _postprocess_synthetic_data(
            synthetic_scaled_df, scaler, continuous_cols, logger
        )

        # 8. Comparación y evaluación
        comparison_report = _compare_real_vs_synthetic(train_df, synthetic_df, logger)

        # Resumen final
        logger.info("=== PIPELINE COMPLETADO EXITOSAMENTE ===")
        logger.info(
            f"Porcentaje discriminador: {comparison_report['_metadata']['discrimination_percentage']}%"
        )
        logger.info(f"Datos sintéticos generados: {len(synthetic_df)} muestras")

    except ValidationError as e:
        logging.error(f"Error de validación: {e}")
        raise
    except Exception as e:
        logging.exception(f"ERROR en pipeline: {str(e)}")
        raise


__all__ = ["main"]
