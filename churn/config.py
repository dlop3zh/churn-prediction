"""
Configuración del pipeline.

Todos los valores tienen un valor por defecto genérico. Los valores reales de
tu entorno (base de datos, esquema, vista origen, nombres de columnas, etc.)
van en un YAML local (`config/config.yaml`) que está en `.gitignore`.
Parte de `config/config.example.yaml`.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

import yaml

# Nombre interno -> nombre de la columna en la fuente de datos.
# Sobrescríbelo en el YAML (`source_columns`) para adaptarlo a tu vista/tabla.
DEFAULT_SOURCE_COLUMNS: Dict[str, str] = {
    "cliente_id":               "CUSTOMER_ID",
    "monetary":                 "SEGMENT",
    "pais":                     "COUNTRY",
    "idcentro":                 "VENUE_ID",
    "centro":                   "VENUE_NAME",
    "anio_mes":                 "YEAR_MONTH",
    "fecha_sesion":             "SESSION_DATE",
    "meses_juego_total":        "MONTHS_ACTIVE",
    "visitas_total_historico":  "TOTAL_VISITS",
    "vis_x_mes_historico":      "VISITS_PER_MONTH",
    "primera_visita":           "FIRST_VISIT",
    "dias_desde_ultima_visita": "DAYS_SINCE_LAST_VISIT",
    "dias_sin_visita_fin_mes":  "DAYS_WITHOUT_VISIT_EOM",
    "rtp_ultimas3_sesiones":    "RTP_LAST3_SESSIONS",
    "rtp_sesion_anterior":      "RTP_PREV_SESSION",
    "win_teorico":              "THEORETICAL_WIN",
    "win_real":                 "ACTUAL_WIN",
    "coin_in":                  "COIN_IN",
    "rtp":                      "RTP",
}


@dataclass
class Config:
    # ── Origen / destino ─────────────────────────────────────────────────────
    mode: str = "local"                       # "local" | "snowflake"
    local_data_path: str = "data/sessions.parquet"
    local_artifacts_dir: str = "artifacts"
    local_output_path: str = "data/scoring.csv"
    reports_dir: str = "reports"

    snowflake_connection_name: Optional[str] = None   # ~/.snowflake/connections.toml
    database: str = "MY_DB"
    schema: str = "MY_SCHEMA"
    source_table: str = "SESSIONS_VIEW"
    scoring_table: str = "CHURN_SCORING_MENSUAL"
    stage: str = "CHURN_ARTIFACTS"
    artifact_name: str = "churn_artifacts.joblib"

    source_columns: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_SOURCE_COLUMNS)
    )

    # ── Definición de churn ──────────────────────────────────────────────────
    churn_hard_days: int = 30
    churn_soft_drop: float = 0.50
    churn_valor_drop: float = 0.50

    # ── Entrenamiento ────────────────────────────────────────────────────────
    test_months: int = 3          # últimos N meses -> validación (no vistos)
    calib_months: int = 1         # meses previos a validación -> calibración VIP
    seed: int = 42
    use_xgboost: bool = True

    # ── Segmentación VIP ─────────────────────────────────────────────────────
    vip_segments: List[str] = field(default_factory=lambda: ["Platinum", "Diamond"])
    vip_excluded_features: List[str] = field(
        default_factory=lambda: ["n_sesiones_mes", "densidad_sesiones"]
    )
    vip_min_calib_rows: int = 30
    vip_fbeta: float = 0.5                    # <1 prioriza precisión ("francotirador")
    vip_threshold: Optional[float] = None     # None -> umbral automático (F-beta)
    vip_threshold_default: float = 0.90       # fallback si no hay datos suficientes

    # ── Umbrales de riesgo (resto de clientes) ───────────────────────────────
    std_threshold_high: float = 0.77
    threshold_medium: float = 0.45

    # ── Inferencia ───────────────────────────────────────────────────────────
    inference_lookback_months: int = 5        # meses previos al mes puntuado

    @property
    def fq_source_table(self) -> str:
        return f"{self.database}.{self.schema}.{self.source_table}"

    @property
    def fq_scoring_table(self) -> str:
        return f"{self.database}.{self.schema}.{self.scoring_table}"

    @property
    def stage_uri(self) -> str:
        return f"@{self.database}.{self.schema}.{self.stage}"


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Carga la configuración: defaults <- YAML <- overrides."""
    cfg = Config()
    data: Dict[str, Any] = {}

    if path is None:
        path = os.environ.get("CHURN_CONFIG")
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    if overrides:
        data.update(overrides)

    valid = {f.name for f in fields(Config)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Claves de configuración desconocidas: {sorted(unknown)}")

    for key, value in data.items():
        if key == "source_columns":
            merged = copy.deepcopy(cfg.source_columns)
            merged.update(value or {})
            value = merged
        setattr(cfg, key, value)
    return cfg
