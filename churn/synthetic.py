"""
Generador de datos sintéticos de sesiones de juego, con el mismo esquema que
espera el pipeline. Permite ejecutar todo el proyecto sin acceso a datos reales.

    python -m churn.synthetic --customers 3000 --months 18 --out data/sessions.parquet

El comportamiento simulado incluye señales realistas de abandono: los clientes
que encadenan sesiones con RTP bajo y bajan su frecuencia tienen más riesgo de
dejar de venir.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_SOURCE_COLUMNS
from .features import previous_month, shift_month

SEGMENTS = ["Bronze", "Silver", "Gold", "Platinum", "Diamond"]
SEGMENT_P = [0.40, 0.27, 0.18, 0.10, 0.05]
SEGMENT_STAKE = {"Bronze": 40, "Silver": 90, "Gold": 200, "Platinum": 500, "Diamond": 1200}
SEGMENT_FREQ = {"Bronze": 3, "Silver": 4.5, "Gold": 6, "Platinum": 8, "Diamond": 10}
COUNTRIES = ["ES", "PT", "IT"]
VENUES = {"ES": [101, 102, 103, 104], "PT": [201, 202], "IT": [301, 302, 303]}


def generate_sessions(n_customers: int = 3000, n_months: int = 18,
                      end_month: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end_month = end_month or previous_month()
    months = [shift_month(end_month, -i) for i in range(n_months - 1, -1, -1)]

    rows = []
    for cid in range(1, n_customers + 1):
        seg = rng.choice(SEGMENTS, p=SEGMENT_P)
        country = rng.choice(COUNTRIES, p=[0.6, 0.15, 0.25])
        home = rng.choice(VENUES[country])
        freq = SEGMENT_FREQ[seg] * rng.lognormal(0, 0.35)
        stake = SEGMENT_STAKE[seg] * rng.lognormal(0, 0.5)
        start_idx = int(rng.integers(0, max(n_months - 3, 1)))
        base_hazard = 0.04 if seg in ("Platinum", "Diamond") else 0.07
        loss_streak = 0
        active = True

        for mi in range(start_idx, n_months):
            if not active:
                # Pequeña probabilidad de reactivación
                if rng.random() < 0.05:
                    active = True
                else:
                    continue
            ym = months[mi]
            y, m = divmod(ym, 100)
            days_in_month = pd.Period(f"{y}-{m:02d}").days_in_month

            # Riesgo de abandono crece con rachas de pérdidas
            hazard = base_hazard + 0.06 * min(loss_streak, 4)
            n_sess = rng.poisson(max(freq * (1 - 0.15 * min(loss_streak, 3)), 0.3))
            if n_sess == 0:
                n_sess = 1 if rng.random() < 0.5 else 0
            if n_sess == 0:
                continue

            days = np.sort(rng.choice(np.arange(1, days_in_month + 1),
                                      size=min(n_sess, days_in_month), replace=False))
            rtp = np.clip(rng.normal(0.93 - 0.02 * loss_streak, 0.12, size=len(days)), 0.2, 2.5)
            coin_in = stake * rng.lognormal(0, 0.6, size=len(days))
            venue = np.where(rng.random(len(days)) < 0.85, home, rng.choice(VENUES[country]))

            for d, r, ci, v in zip(days, rtp, coin_in, venue):
                rows.append((cid, seg, country, int(v), ym,
                             pd.Timestamp(y, m, int(d)), float(r), float(ci)))

            month_loss = float(np.sum(coin_in * (rtp - 1)))
            loss_streak = loss_streak + 1 if month_loss < 0 else 0
            if rng.random() < hazard:
                active = False

    df = pd.DataFrame(rows, columns=["cliente_id", "monetary", "pais", "idcentro",
                                     "anio_mes", "fecha_sesion", "rtp", "coin_in"])
    return _derive_session_columns(df)


def _derive_session_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["cliente_id", "fecha_sesion"]).reset_index(drop=True)
    g = df.groupby("cliente_id")

    df["centro"] = "Venue " + df["idcentro"].astype(str)
    df["win_real"] = df["coin_in"] * (df["rtp"] - 1)            # resultado del jugador
    df["win_teorico"] = df["coin_in"] * (0.92 - 1)              # esperado con RTP teórico
    df["rtp_sesion_anterior"] = g["rtp"].shift(1)
    df["rtp_ultimas3_sesiones"] = g["rtp"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["dias_desde_ultima_visita"] = g["fecha_sesion"].diff().dt.days
    df["primera_visita"] = g["fecha_sesion"].transform("min")
    df["visitas_total_historico"] = g.cumcount() + 1
    df["meses_juego_total"] = (
        df.groupby("cliente_id")["anio_mes"].transform(lambda s: (~s.duplicated()).cumsum())
    )
    df["vis_x_mes_historico"] = df["visitas_total_historico"] / df["meses_juego_total"]
    month_end = df["fecha_sesion"] + pd.offsets.MonthEnd(0)
    last_in_month = df.groupby(["cliente_id", "anio_mes"])["fecha_sesion"].transform("max")
    df["dias_sin_visita_fin_mes"] = (month_end - last_in_month).dt.days

    to_source = {k: v for k, v in DEFAULT_SOURCE_COLUMNS.items() if k in df.columns}
    return df.rename(columns=to_source)[list(to_source.values())]


def main() -> None:
    p = argparse.ArgumentParser(description="Genera sesiones sintéticas")
    p.add_argument("--customers", type=int, default=3000)
    p.add_argument("--months", type=int, default=18)
    p.add_argument("--end-month", type=int, default=None, help="AAAAMM (por defecto, último mes cerrado)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/sessions.parquet")
    a = p.parse_args()

    df = generate_sessions(a.customers, a.months, a.end_month, a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.out.endswith(".parquet"):
        df.to_parquet(a.out, index=False)
    else:
        df.to_csv(a.out, index=False)
    print(f"[OK] {len(df):,} sesiones sintéticas de {df['CUSTOMER_ID'].nunique():,} "
          f"clientes -> {a.out}")


if __name__ == "__main__":
    main()
