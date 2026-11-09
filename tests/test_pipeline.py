import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from churn.config import DEFAULT_SOURCE_COLUMNS, load_config  # noqa: E402
from churn.features import (BASE_FEATURES, build_client_month_df,  # noqa: E402
                            previous_month, shift_month)
from churn.scoring import assign_risk  # noqa: E402
from churn.synthetic import generate_sessions  # noqa: E402

END_MONTH = 202606


@pytest.fixture(scope="module")
def sessions():
    return generate_sessions(n_customers=1500, n_months=16, end_month=END_MONTH, seed=7)


# ── Utilidades de fechas ─────────────────────────────────────────────────────
def test_month_helpers():
    from datetime import date
    assert previous_month(date(2026, 1, 15)) == 202512
    assert previous_month(date(2026, 11, 9)) == 202610
    assert shift_month(202601, -1) == 202512
    assert shift_month(202611, 3) == 202702
    assert shift_month(202606, -18) == 202412


# ── Feature engineering ──────────────────────────────────────────────────────
def test_hard_streak_and_targets():
    c = DEFAULT_SOURCE_COLUMNS
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-09", "2026-01-12",
                            "2026-01-20", "2026-03-15"])
    df = pd.DataFrame({
        c["cliente_id"]: "1", c["monetary"]: "Gold", c["pais"]: "ES",
        c["anio_mes"]: [202601] * 5 + [202603],
        c["fecha_sesion"]: dates,
        c["rtp"]: [0.5, 0.6, 1.2, 0.7, 0.8, 0.95],
        c["coin_in"]: 100.0, c["win_real"]: -10.0, c["win_teorico"]: -8.0,
        c["dias_desde_ultima_visita"]: 3.0, c["meses_juego_total"]: 1,
        c["visitas_total_historico"]: 5, c["vis_x_mes_historico"]: 5.0,
        c["primera_visita"]: dates.min(), c["dias_sin_visita_fin_mes"]: 11,
        c["rtp_ultimas3_sesiones"]: 0.8,
    })
    out = build_client_month_df(df, DEFAULT_SOURCE_COLUMNS, inference=True,
                                target_month=202601, verbose=False)
    row = out.iloc[0]
    assert row["racha_max_sesiones_duras"] == 2          # 0.5, 0.6 | 1.2 | 0.7, 0.8
    assert row["n_sesiones_mes"] == 5
    assert row["ratio_advantage"] == pytest.approx(0.2)

    # Target: próxima visita 54 días después -> churn hard 30d = 1
    from churn.features import _add_targets, _aggregate, _inter_month_features, \
        _month_features, _session_features, prepare_sessions
    s = _session_features(prepare_sessions(df.copy(), DEFAULT_SOURCE_COLUMNS))
    agg = _inter_month_features(_month_features(_aggregate(s)))
    agg = _add_targets(agg, s, 30, 0.5, 0.5)
    assert agg.loc[agg["anio_mes"] == 202601, "churn_hard_30d"].item() == 1
    assert agg.loc[agg["anio_mes"] == 202603, "churn_hard_30d"].item() == 1  # sin visita futura


def test_train_inference_feature_parity(sessions):
    """
    Las features de un mes calculadas en modo inferencia (ventana corta) deben
    coincidir con las del modo entrenamiento (histórico completo) para clientes
    con actividad continua en la ventana.
    """
    month = shift_month(END_MONTH, -4)
    lookback = 5
    month_col = DEFAULT_SOURCE_COLUMNS["anio_mes"]
    cid_col = DEFAULT_SOURCE_COLUMNS["cliente_id"]

    full = build_client_month_df(sessions.copy(), DEFAULT_SOURCE_COLUMNS, verbose=False)
    window = sessions[(sessions[month_col] >= shift_month(month, -lookback))
                      & (sessions[month_col] <= month)].copy()
    inf = build_client_month_df(window, DEFAULT_SOURCE_COLUMNS, inference=True,
                                target_month=month, verbose=False)

    # Clientes activos en todos los meses de la ventana
    n_months = window.groupby(cid_col)[month_col].nunique()
    continuous = set(n_months[n_months == lookback + 1].index.astype(str))
    assert len(continuous) > 20

    feats = [f for f in BASE_FEATURES if f not in {
        "meses_consecutivos_perdiendo",   # depende de todo el histórico (ver README)
    }]
    a = full[(full["anio_mes"] == month) & full["cliente_id"].isin(continuous)] \
        .set_index("cliente_id")[feats].sort_index()
    b = inf[inf["cliente_id"].isin(continuous)].set_index("cliente_id")[feats].sort_index()
    common = a.index.intersection(b.index)
    assert len(common) > 20
    pd.testing.assert_frame_equal(a.loc[common], b.loc[common], check_dtype=False, rtol=1e-4)


# ── Scoring ──────────────────────────────────────────────────────────────────
def test_assign_risk():
    thr = {"medium": 0.45, "vip_high": 0.60, "std_high": 0.77}
    prob = np.array([0.30, 0.50, 0.65, 0.65, 0.80])
    vip = np.array([True, True, True, False, False])
    pred, seg = assign_risk(prob, vip, thr)
    assert pred.tolist() == [0, 0, 1, 0, 1]
    assert seg.tolist() == ["Bajo", "Medio", "Alto", "Medio", "Alto"]


# ── End to end ───────────────────────────────────────────────────────────────
def test_end_to_end(sessions, tmp_path):
    from churn.inference import run_inference
    from churn.train import run_training

    data_path = tmp_path / "sessions.parquet"
    sessions.to_parquet(data_path, index=False)
    cfg = load_config(overrides={
        "mode": "local",
        "local_data_path": str(data_path),
        "local_artifacts_dir": str(tmp_path / "artifacts"),
        "local_output_path": str(tmp_path / "scoring.csv"),
        "reports_dir": str(tmp_path / "reports"),
        "use_xgboost": False,
    })
    bundle = run_training(cfg)
    assert bundle["general"]["model"] is not None
    assert 0 < bundle["thresholds"]["vip_high"] <= 1

    run_inference(cfg, target_month=END_MONTH)
    run_inference(cfg, target_month=END_MONTH)          # idempotente
    out = pd.read_csv(cfg.local_output_path)
    assert out["ANIO_MES"].unique().tolist() == [END_MONTH]
    assert out["CLIENTE_ID"].is_unique
    assert out["PROB_CHURN_HARD"].between(0, 1).all()
    assert set(out["SEGMENTO_RIESGO"]) <= {"Bajo", "Medio", "Alto"}
    if bundle["vip"] is not None:
        vip_rows = out[out["MONETARY"].isin(cfg.vip_segments)]
        assert (vip_rows["MODELO"] == "vip").all()
