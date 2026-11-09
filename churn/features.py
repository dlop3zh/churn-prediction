"""
Feature engineering: sesiones de juego -> dataset cliente-mes.

Una única función (`build_client_month_df`) se usa tanto en entrenamiento como
en inferencia, para garantizar que las features se calculan exactamente igual
en ambos modos (evita *training-serving skew*).
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

KEYS = ["cliente_id", "anio_mes"]

# Features numéricas base. Las dummies del segmento (`monetary_*`) se añaden
# dinámicamente en entrenamiento según los valores reales del campo.
BASE_FEATURES: List[str] = [
    "rtp_medio", "rtp_min", "rtp_max", "rtp_std",
    "rtp_medio_rolling3m", "rtp_vs_mes_anterior",
    "n_sesiones_mes", "densidad_sesiones",
    "coin_in_total", "coin_in_medio",
    "win_real_total", "margen_casa_total",
    "ratio_advantage", "ratio_sesiones_duras",
    "racha_max_sesiones_duras",
    "dias_entre_sesiones_medio", "dias_entre_sesiones_max",
    "n_sesiones_rolling3m", "coin_in_total_rolling3m",
    "win_real_total_rolling3m",
    "meses_juego_total",
    "visita_unica",
    "dias_sin_visita_fin_mes",
    "rtp_ultimas3_sesiones",
    "ratio_visitas_vs_historico",
    "meses_consecutivos_perdiendo",
    "ratio_coinIn_vs_historico",
    "rtp_vs_rolling3m",
    "tendencia_visitas_3m",
    "ratio_sesiones_duras_rolling3m",
]

SEGMENT_PREFIX = "monetary_"


# =============================================================================
# Utilidades
# =============================================================================
def previous_month(as_of: Optional[date] = None) -> int:
    """Último mes cerrado (AAAAMM) respecto a `as_of` (por defecto hoy)."""
    d = as_of or date.today()
    return (d.year - 1) * 100 + 12 if d.month == 1 else d.year * 100 + d.month - 1


def shift_month(yyyymm: int, months: int) -> int:
    """Desplaza un AAAAMM `months` meses (negativo = hacia atrás)."""
    y, m = divmod(int(yyyymm), 100)
    total = y * 12 + (m - 1) + months
    return (total // 12) * 100 + total % 12 + 1


def _parse_numeric(s: pd.Series, percent: bool = False) -> pd.Series:
    """Convierte strings con formato europeo ('1.234,5' / '95,3%') a float."""
    if s.dtype != object:
        return s
    out = s.astype(str).str.strip()
    if percent:
        out = out.str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
        return pd.to_numeric(out, errors="coerce") / 100
    out = out.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(out, errors="coerce")


def _rolling_prev(agg: pd.DataFrame, col: str, window: int = 3) -> pd.Series:
    """Media de los `window` meses ANTERIORES (excluye el mes actual)."""
    return agg.groupby("cliente_id")[col].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )


# =============================================================================
# Pipeline
# =============================================================================
def prepare_sessions(df: pd.DataFrame, source_columns: Dict[str, str]) -> pd.DataFrame:
    """Renombra columnas fuente -> internas y tipa."""
    df.columns = df.columns.str.strip()
    rename = {src: internal for internal, src in source_columns.items() if src in df.columns}
    df = df.rename(columns=rename)

    df["cliente_id"] = df["cliente_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["fecha_sesion"] = pd.to_datetime(df["fecha_sesion"], errors="coerce")
    if "primera_visita" in df.columns:
        df["primera_visita"] = pd.to_datetime(df["primera_visita"], errors="coerce")

    for col in ["rtp", "rtp_sesion_anterior", "rtp_ultimas3_sesiones"]:
        if col in df.columns:
            df[col] = _parse_numeric(df[col], percent=True)
    for col in ["win_teorico", "win_real", "coin_in", "dias_desde_ultima_visita",
                "dias_sin_visita_fin_mes", "vis_x_mes_historico",
                "meses_juego_total", "visitas_total_historico"]:
        if col in df.columns:
            df[col] = _parse_numeric(df[col])

    df = df[df["anio_mes"].notna()].copy()
    df["anio_mes"] = df["anio_mes"].astype(float).astype(int)
    if "pais" not in df.columns:
        df["pais"] = "NA"
    df = df.sort_values(["cliente_id", "fecha_sesion"]).reset_index(drop=True)
    df["rtp"] = df["rtp"].clip(upper=5.0)
    return df


def _session_features(df: pd.DataFrame) -> pd.DataFrame:
    df["es_advantage"] = (df["rtp"] > 1.0).astype(np.int8)
    df["es_dura"] = (df["rtp"] < 0.90).astype(np.int8)
    df["margen_casa"] = df["win_teorico"] - df["win_real"]
    return df


def _max_hard_streak(df: pd.DataFrame) -> pd.DataFrame:
    """Racha máxima de sesiones 'duras' (RTP < 0.90) consecutivas por cliente-mes."""
    prev = df.groupby(KEYS)["es_dura"].shift()
    run_id = (df["es_dura"] != prev).astype(int).groupby([df[k] for k in KEYS]).cumsum()
    hard = df[df["es_dura"] == 1]
    return (
        hard.groupby([hard["cliente_id"], hard["anio_mes"], run_id[hard.index]])
        .size()
        .groupby(level=[0, 1]).max()
        .rename("racha_max_sesiones_duras")
        .reset_index()
    )


def _mode_per_group(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Valor más frecuente de `col` por cliente-mes (vectorizado)."""
    counts = df.groupby(KEYS + [col]).size().reset_index(name="_n")
    counts = counts.sort_values(KEYS + ["_n"], ascending=[True, True, False])
    return counts.drop_duplicates(KEYS)[KEYS + [col]]


def _main_venue(df: pd.DataFrame) -> pd.DataFrame:
    """Centro principal del mes (más sesiones; empate -> más coin-in). Informativo."""
    stats = (
        df.groupby(KEYS + ["idcentro", "centro"], dropna=False)
        .agg(_n=("fecha_sesion", "count"), _ci=("coin_in", "sum"))
        .reset_index()
        .sort_values(KEYS + ["_n", "_ci"], ascending=[True, True, False, False])
    )
    main = (
        stats.drop_duplicates(KEYS)[KEYS + ["idcentro", "centro"]]
        .rename(columns={"idcentro": "idcentro_principal", "centro": "centro_principal"})
    )
    n_venues = df.groupby(KEYS)["idcentro"].nunique().rename("n_centros_distintos").reset_index()
    return main.merge(n_venues, on=KEYS, how="left")


def _consecutive_losing_months(agg: pd.DataFrame) -> pd.Series:
    """Meses consecutivos con win_real_total negativo (o sin dato)."""
    losing = agg["win_real_total"].isna() | (agg["win_real_total"] < 0)
    reset_id = (~losing).groupby(agg["cliente_id"]).cumsum()
    return losing.astype(int).groupby([agg["cliente_id"], reset_id]).cumsum()


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["cliente_id", "pais", "anio_mes"]).agg(
        fecha_primera_sesion_mes=("fecha_sesion", "min"),
        fecha_ultima_sesion_mes=("fecha_sesion", "max"),
        n_sesiones_mes=("fecha_sesion", "count"),
        coin_in_total=("coin_in", "sum"),
        coin_in_medio=("coin_in", "mean"),
        coin_in_max=("coin_in", "max"),
        rtp_medio=("rtp", "mean"),
        rtp_min=("rtp", "min"),
        rtp_max=("rtp", "max"),
        rtp_std=("rtp", "std"),
        win_real_total=("win_real", "sum"),
        win_real_medio=("win_real", "mean"),
        win_teorico_total=("win_teorico", "sum"),
        margen_casa_total=("margen_casa", "sum"),
        n_sesiones_advantage=("es_advantage", "sum"),
        n_sesiones_duras=("es_dura", "sum"),
        dias_entre_sesiones_medio=("dias_desde_ultima_visita", "mean"),
        dias_entre_sesiones_max=("dias_desde_ultima_visita", "max"),
        meses_juego_total=("meses_juego_total", "first"),
        visitas_total_historico=("visitas_total_historico", "first"),
        vis_x_mes_historico=("vis_x_mes_historico", "first"),
        primera_visita=("primera_visita", "first"),
        dias_sin_visita_fin_mes=("dias_sin_visita_fin_mes", "max"),
        rtp_ultimas3_sesiones=("rtp_ultimas3_sesiones", "mean"),
    ).reset_index()

    agg = agg.merge(_max_hard_streak(df), on=KEYS, how="left")
    agg["racha_max_sesiones_duras"] = agg["racha_max_sesiones_duras"].fillna(0).astype(int)

    if "monetary" in df.columns:
        agg = agg.merge(_mode_per_group(df, "monetary"), on=KEYS, how="left")
        dummies = pd.get_dummies(agg["monetary"], prefix="monetary", dtype=np.int8)
        agg = pd.concat([agg, dummies], axis=1)

    if "idcentro" in df.columns and "centro" in df.columns:
        agg = agg.merge(_main_venue(df), on=KEYS, how="left")

    return agg


def _month_features(agg: pd.DataFrame) -> pd.DataFrame:
    n = agg["n_sesiones_mes"]
    agg["ratio_advantage"] = agg["n_sesiones_advantage"] / n
    agg["ratio_sesiones_duras"] = agg["n_sesiones_duras"] / n
    agg["ratio_visitas_vs_historico"] = n / agg["vis_x_mes_historico"].replace(0, np.nan)
    agg["win_real_vs_teorico"] = agg["win_real_total"] / agg["win_teorico_total"].replace(0, np.nan)
    agg["duracion_mes_dias"] = (
        agg["fecha_ultima_sesion_mes"] - agg["fecha_primera_sesion_mes"]
    ).dt.days + 1
    agg["densidad_sesiones"] = n / agg["duracion_mes_dias"].replace(0, np.nan)
    agg["visita_unica"] = (n == 1).astype(int)
    return agg


def _inter_month_features(agg: pd.DataFrame) -> pd.DataFrame:
    agg = agg.sort_values(KEYS).reset_index(drop=True)
    for col, alias in [("rtp_medio", "rtp_medio"),
                       ("win_real_total", "win_real_total"),
                       ("coin_in_total", "coin_in_total"),
                       ("n_sesiones_mes", "n_sesiones")]:
        agg[f"{alias}_rolling3m"] = _rolling_prev(agg, col)

    agg["meses_consecutivos_perdiendo"] = _consecutive_losing_months(agg)
    agg["rtp_vs_mes_anterior"] = agg.groupby("cliente_id")["rtp_medio"].diff()
    agg["ratio_coinIn_vs_historico"] = (
        agg["coin_in_total"] / agg["coin_in_total_rolling3m"].replace(0, np.nan)
    )
    agg["rtp_vs_rolling3m"] = agg["rtp_medio"] - agg["rtp_medio_rolling3m"]

    # Pendiente media de visitas en los 3 meses previos (negativo = señal de riesgo)
    agg["tendencia_visitas_3m"] = agg.groupby("cliente_id")["n_sesiones_mes"].transform(
        lambda x: x.diff().shift(1).rolling(3, min_periods=1).mean()
    )
    agg["ratio_sesiones_duras_rolling3m"] = _rolling_prev(agg, "ratio_sesiones_duras")
    return agg


def _add_targets(agg: pd.DataFrame, df: pd.DataFrame, churn_hard_days: int,
                 churn_soft_drop: float, churn_valor_drop: float) -> pd.DataFrame:
    # Churn hard: sin visita en los N días siguientes a la última sesión del mes.
    first_by_month = (
        df.groupby(KEYS)["fecha_sesion"].min().reset_index()
        .rename(columns={"anio_mes": "anio_mes_futuro", "fecha_sesion": "primera_sesion_mes"})
        .sort_values(["cliente_id", "anio_mes_futuro"])
    )
    # La primera sesión del mes siguiente con actividad = próxima visita.
    first_by_month["primera_sesion_futura"] = (
        first_by_month.groupby("cliente_id")["primera_sesion_mes"].shift(-1)
    )
    nxt = first_by_month.rename(columns={"anio_mes_futuro": "anio_mes"})[
        KEYS + ["primera_sesion_futura"]
    ]
    agg = agg.merge(nxt, on=KEYS, how="left")

    dias = (agg["primera_sesion_futura"] - agg["fecha_ultima_sesion_mes"]).dt.days
    churn_col = f"churn_hard_{churn_hard_days}d"
    agg[churn_col] = np.where(dias.isna(), 1, (dias > churn_hard_days).astype(int))
    agg = agg.drop(columns=["primera_sesion_futura"])

    def _drop(actual, ref, thr):
        valid = ref.notna() & (ref != 0)
        return np.where(valid, (((ref - actual) / ref) > thr).astype(float), np.nan)

    agg["churn_soft"] = _drop(agg["n_sesiones_mes"], agg["n_sesiones_rolling3m"], churn_soft_drop)
    agg["churn_valor"] = _drop(agg["coin_in_total"], agg["coin_in_total_rolling3m"], churn_valor_drop)
    return agg


def build_client_month_df(
    df: pd.DataFrame,
    source_columns: Dict[str, str],
    churn_hard_days: int = 30,
    churn_soft_drop: float = 0.50,
    churn_valor_drop: float = 0.50,
    inference: bool = False,
    target_month: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Transforma sesiones brutas en un DataFrame cliente-mes.

    inference=False -> modo entrenamiento: calcula targets y excluye meses sin
                       ventana futura suficiente para observar el churn.
    inference=True  -> devuelve SOLO `target_month` (por defecto, el último mes
                       cerrado respecto a hoy), sin targets.
    """
    df = prepare_sessions(df, source_columns)
    df = _session_features(df)
    agg = _aggregate(df)
    agg = _month_features(agg)
    agg = _inter_month_features(agg)

    if inference:
        target_month = target_month or previous_month()
        if target_month not in set(agg["anio_mes"]):
            raise ValueError(
                f"El mes {target_month} no está en los datos. "
                f"Meses disponibles: {sorted(agg['anio_mes'].unique())}"
            )
        n_months = agg.groupby("cliente_id")["anio_mes"].transform("count")
        agg["cliente_nuevo"] = (n_months < 3).astype(int)
        out = agg[agg["anio_mes"] == target_month].copy()
        if verbose:
            print(f"[INF] Mes a puntuar: {target_month}  |  clientes: {len(out):,}")
        return out

    # ── Entrenamiento ────────────────────────────────────────────────────────
    agg = _add_targets(agg, df, churn_hard_days, churn_soft_drop, churn_valor_drop)

    # Se excluye el último mes global (posible mes abierto) y los 2 últimos
    # meses de cada cliente: en los meses más recientes aún no ha pasado
    # tiempo suficiente para confirmar si el cliente volvió.
    agg = agg[agg["anio_mes"] != agg["anio_mes"].max()].copy()
    for _ in range(2):
        last = agg.groupby("cliente_id")["anio_mes"].transform("max")
        agg = agg[agg["anio_mes"] != last].copy()

    n_months = agg.groupby("cliente_id")["anio_mes"].transform("count")
    agg["cliente_nuevo"] = (n_months < 3).astype(int)

    if verbose:
        churn_col = f"churn_hard_{churn_hard_days}d"
        print("[OK] DataFrame cliente-mes (train)")
        print(f"     Filas: {len(agg):,}  |  clientes: {agg['cliente_id'].nunique():,}"
              f"  |  meses: {agg['anio_mes'].nunique()}")
        print(f"     Churn hard: {agg[churn_col].mean():.1%}  |  soft: "
              f"{agg['churn_soft'].mean():.1%}  |  valor: {agg['churn_valor'].mean():.1%}")
    return agg.reset_index(drop=True)


def segment_dummy_columns(df: pd.DataFrame) -> List[str]:
    return sorted(c for c in df.columns if c.startswith(SEGMENT_PREFIX))


def make_matrix(df: pd.DataFrame, features: List[str], medians: pd.Series) -> pd.DataFrame:
    """
    Matriz de features alineada con el entrenamiento: columnas que falten se
    crean (dummies de segmento ausentes -> 0, resto -> mediana del train).
    """
    X = df.reindex(columns=features)
    dummy_cols = [c for c in features if c.startswith(SEGMENT_PREFIX)]
    if dummy_cols:
        X[dummy_cols] = X[dummy_cols].fillna(0)
    num_cols = [c for c in features if c not in dummy_cols]
    X[num_cols] = X[num_cols].astype(float).fillna(medians.reindex(num_cols))
    return X.astype(float)
