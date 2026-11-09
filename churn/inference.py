"""
Inferencia mensual: puntúa el último mes cerrado y escribe el resultado.

Local / CLI:
    python -m churn.inference --config config/config.yaml [--month 202610]

Snowflake Stored Procedure:
    HANDLER = 'churn.inference.sp_handler'   (ver sql/02_stored_procedure.sql)
"""
from __future__ import annotations

import argparse
import os
from datetime import date
from typing import Optional

import pandas as pd

from . import io
from .config import Config, load_config
from .features import build_client_month_df, previous_month, shift_month
from .scoring import score

OUTPUT_COLUMNS = [
    "cliente_id", "pais", "idcentro_principal", "centro_principal", "n_centros_distintos",
    "monetary", "anio_mes", "fecha_ultima_visita",
    "prob_churn_hard", "pred_churn_hard", "segmento_riesgo", "modelo", "cliente_nuevo",
    "n_sesiones_mes", "rtp_medio", "coin_in_total",
    "ratio_advantage", "racha_max_sesiones_duras", "dias_entre_sesiones_medio",
    "fecha_scoring",
]


def _format_output(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"fecha_ultima_sesion_mes": "fecha_ultima_visita"})
    today = date.today()
    df["fecha_scoring"] = today.year * 100 + today.month
    out = df[[c for c in OUTPUT_COLUMNS if c in df.columns]].copy()

    for col in ["cliente_id", "pais", "centro_principal", "monetary", "segmento_riesgo", "modelo"]:
        if col in out.columns:
            out[col] = out[col].astype(str)
    if "idcentro_principal" in out.columns:
        out["idcentro_principal"] = out["idcentro_principal"].astype(str).str.replace(r"\.0$", "", regex=True)
    for col in ["n_centros_distintos", "cliente_nuevo", "anio_mes", "pred_churn_hard",
                "fecha_scoring", "n_sesiones_mes", "racha_max_sesiones_duras"]:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)
    if "fecha_ultima_visita" in out.columns:
        out["fecha_ultima_visita"] = pd.to_datetime(out["fecha_ultima_visita"])
    float_cols = out.select_dtypes("float").columns
    out[float_cols] = out[float_cols].fillna(-1.0)

    out.columns = [c.upper() for c in out.columns]
    return out


def run_inference(cfg: Config, session=None, target_month: Optional[int] = None,
                  df_sessions: Optional[pd.DataFrame] = None) -> str:
    target_month = target_month or previous_month()

    # Solo se descargan los meses necesarios para los rolling (no todo el histórico)
    if df_sessions is None:
        from_month = shift_month(target_month, -cfg.inference_lookback_months)
        df_sessions = io.load_sessions(cfg, session, from_month=from_month)

    bundle = io.load_artifacts(cfg, session)
    print(f"[OK] Artefactos: general={bundle['general']['name']} | "
          f"vip={bundle['vip']['name'] if bundle.get('vip') else '—'} | "
          f"entrenado {bundle.get('trained_at')} | umbrales {bundle['thresholds']}")

    df_inf = build_client_month_df(df_sessions, cfg.source_columns,
                                   inference=True, target_month=target_month)
    df_inf = score(df_inf, bundle)

    dist = df_inf["segmento_riesgo"].value_counts()
    for seg in ["Alto", "Medio", "Bajo"]:
        n = int(dist.get(seg, 0))
        print(f"     {seg:<6}: {n:,} ({n / max(len(df_inf), 1):.1%})")

    df_out = _format_output(df_inf)
    dest = io.write_scoring(df_out, target_month, cfg, session)
    msg = f"[OK] {len(df_out):,} registros del mes {target_month} guardados en {dest}"
    print(msg)
    return msg


def sp_handler(session, target_month: int = 0) -> str:
    """Punto de entrada del Stored Procedure (Snowflake inyecta la sesión)."""
    import_dir = io.stored_procedure_import_dir()
    cfg_path = os.path.join(import_dir, "config.yaml") if import_dir else None
    cfg = load_config(cfg_path if cfg_path and os.path.exists(cfg_path) else None,
                      overrides={"mode": "snowflake"})
    session.use_database(cfg.database)
    session.use_schema(cfg.schema)
    return run_inference(cfg, session, target_month=target_month or None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia mensual de churn")
    parser.add_argument("--config", default=None, help="Ruta al YAML de configuración")
    parser.add_argument("--month", type=int, default=None,
                        help="Mes a puntuar AAAAMM (por defecto, el último mes cerrado)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    session = io.create_snowflake_session(cfg) if cfg.mode == "snowflake" else None
    try:
        run_inference(cfg, session, target_month=args.month)
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    main()
