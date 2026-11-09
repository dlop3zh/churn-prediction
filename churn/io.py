"""
Entrada/salida: lectura de sesiones, artefactos del modelo y escritura del scoring.

Dos backends:
  - local     -> ficheros parquet/csv y carpeta `artifacts/`
  - snowflake -> tabla/vista origen, stage interno y tabla de scoring (Snowpark)
"""
from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from .config import Config


# =============================================================================
# Sesión Snowflake (ejecución fuera de Snowflake)
# =============================================================================
def create_snowflake_session(cfg: Config):
    """
    Crea una sesión Snowpark usando una conexión con nombre de
    ~/.snowflake/connections.toml (o la conexión por defecto si no se indica).
    Las credenciales NUNCA van en el repositorio.
    """
    from snowflake.snowpark import Session

    builder = Session.builder
    if cfg.snowflake_connection_name:
        builder = builder.config("connection_name", cfg.snowflake_connection_name)
    session = builder.create()
    session.use_database(cfg.database)
    session.use_schema(cfg.schema)
    return session


# =============================================================================
# Datos de sesiones
# =============================================================================
def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memoria: float64 -> float32, int64 -> menor entero posible."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def load_sessions(cfg: Config, session=None, from_month: Optional[int] = None) -> pd.DataFrame:
    """Carga las sesiones (opcionalmente desde `from_month` AAAAMM)."""
    month_col = cfg.source_columns["anio_mes"]

    if cfg.mode == "local":
        path = cfg.local_data_path
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        if from_month is not None:
            df = df[df[month_col] >= from_month]
    else:
        table = session.table(cfg.fq_source_table)
        if from_month is not None:
            table = table.filter(f"{month_col} >= {int(from_month)}")
        # Descarga por lotes: evita el pico de memoria de un to_pandas() de golpe.
        batches = list(table.to_pandas_batches())
        df = pd.concat(batches, ignore_index=True, copy=False)
        del batches

    df = _downcast(df.reset_index(drop=True))
    print(f"[OK] Sesiones cargadas: {len(df):,} filas  |  "
          f"{df[month_col].min()} -> {df[month_col].max()}")
    return df


# =============================================================================
# Artefactos
# =============================================================================
def save_artifacts(bundle: Dict[str, Any], cfg: Config, session=None) -> str:
    if cfg.mode == "local":
        os.makedirs(cfg.local_artifacts_dir, exist_ok=True)
        path = os.path.join(cfg.local_artifacts_dir, cfg.artifact_name)
        joblib.dump(bundle, path)
        print(f"[OK] Artefactos guardados en {path}")
        return path

    session.sql(f"CREATE STAGE IF NOT EXISTS {cfg.stage_uri[1:]}").collect()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, cfg.artifact_name)
        joblib.dump(bundle, path)
        session.file.put(path, cfg.stage_uri, auto_compress=False, overwrite=True)
    print(f"[OK] Artefactos subidos a {cfg.stage_uri}/{cfg.artifact_name}")
    return f"{cfg.stage_uri}/{cfg.artifact_name}"


def load_artifacts(cfg: Config, session=None) -> Dict[str, Any]:
    if cfg.mode == "local":
        path = os.path.join(cfg.local_artifacts_dir, cfg.artifact_name)
        return joblib.load(path)

    tmp = tempfile.mkdtemp()
    session.file.get(f"{cfg.stage_uri}/{cfg.artifact_name}", tmp)
    return joblib.load(os.path.join(tmp, cfg.artifact_name))


# =============================================================================
# Escritura del scoring
# =============================================================================
SCORING_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    CLIENTE_ID                VARCHAR,
    PAIS                      VARCHAR,
    IDCENTRO_PRINCIPAL        VARCHAR,
    CENTRO_PRINCIPAL          VARCHAR,
    N_CENTROS_DISTINTOS       INTEGER,
    MONETARY                  VARCHAR,
    ANIO_MES                  INTEGER,
    FECHA_ULTIMA_VISITA       DATE,
    PROB_CHURN_HARD           FLOAT,
    PRED_CHURN_HARD           INTEGER,
    SEGMENTO_RIESGO           VARCHAR,
    MODELO                    VARCHAR,
    CLIENTE_NUEVO             INTEGER,
    N_SESIONES_MES            INTEGER,
    RTP_MEDIO                 FLOAT,
    COIN_IN_TOTAL             FLOAT,
    RATIO_ADVANTAGE           FLOAT,
    RACHA_MAX_SESIONES_DURAS  INTEGER,
    DIAS_ENTRE_SESIONES_MEDIO FLOAT,
    FECHA_SCORING             INTEGER
)
"""


def write_scoring(df_out: pd.DataFrame, month: int, cfg: Config, session=None) -> str:
    """Escritura idempotente: borra el mes y lo vuelve a insertar."""
    if cfg.mode == "local":
        path = cfg.local_output_path
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            prev = pd.read_csv(path)
            prev = prev[prev["ANIO_MES"] != month]
            df_out = pd.concat([prev, df_out], ignore_index=True)
        df_out.to_csv(path, index=False)
        return path

    table = cfg.fq_scoring_table
    session.sql(SCORING_DDL.format(table=table)).collect()
    session.sql(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS MODELO VARCHAR").collect()
    session.sql(f"DELETE FROM {table} WHERE ANIO_MES = {int(month)}").collect()

    sp_df = session.create_dataframe(df_out)
    if "FECHA_ULTIMA_VISITA" in sp_df.columns:
        sp_df = sp_df.with_column("FECHA_ULTIMA_VISITA", sp_df["FECHA_ULTIMA_VISITA"].cast("date"))
    sp_df.write.mode("append").save_as_table(table)
    return table


def stored_procedure_import_dir() -> Optional[str]:
    """Directorio de IMPORTS cuando el código corre dentro de un Stored Procedure."""
    return sys._xoptions.get("snowflake_import_directory")  # type: ignore[attr-defined]
