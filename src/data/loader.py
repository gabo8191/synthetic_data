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

    try:
        # Cargar datos
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        elif file_path.endswith(".json"):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Formato de archivo no soportado")

        report["load_info"] = {
            "file": os.path.basename(file_path),
            "original_shape": df.shape,
            "original_columns": df.columns.tolist(),
        }

        # Limpieza básica
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df.replace(["?", "", "NA", "NaN"], np.nan, inplace=True)

        # Normalizar nombres de columnas
        df.columns = [
            re.sub(r"\W+", "_", str(col).strip().lower()) for col in df.columns
        ]

        # Corregir columna objetivo si existe en otra forma de mayúsculas/minúsculas
        if target_column not in df.columns:
            # Intentar variantes comunes
            candidates = {c.lower(): c for c in df.columns}
            if target_column.lower() in candidates:
                df.rename(
                    columns={candidates[target_column.lower()]: target_column},
                    inplace=True,
                )

        # Convertir numéricos a float
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        # Manejar valores nulos
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ["float64", "int64"]:
                    fill_val = df[col].median()
                else:
                    fill_val = df[col].mode()[0]
                df[col] = df[col].fillna(fill_val)

        # Eliminar duplicados
        dups_before = int(df.duplicated().sum())
        df.drop_duplicates(inplace=True)
        dups_after = int(df.duplicated().sum())
        report["duplicates"] = {
            "before": dups_before,
            "removed": dups_before - dups_after,
        }

        report["final_info"] = {
            "final_shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict(),
        }

        return df, dict(report)

    except Exception as e:
        report["error"] = str(e)
        raise RuntimeError(f"Error en loader: {str(e)}")


__all__ = ["load_and_clean_data"]
