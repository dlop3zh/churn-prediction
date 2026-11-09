"""
Scoring: aplica el bundle de artefactos (modelo general + modelo VIP) y
asigna predicción y segmento de riesgo con umbrales diferenciados.
Se usa igual en la validación del entrenamiento y en la inferencia mensual.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .features import make_matrix


def assign_risk(prob: np.ndarray, is_vip: np.ndarray, thresholds: Dict[str, float]):
    """
    VIP   : Bajo < medium <= Medio < vip_high <= Alto
    Resto : Bajo < medium <= Medio <= std_high < Alto
    """
    medium, vip_high, std_high = thresholds["medium"], thresholds["vip_high"], thresholds["std_high"]
    pred = np.where(is_vip, prob >= vip_high, prob > std_high).astype(int)
    seg_vip = np.select([prob < medium, prob < vip_high], ["Bajo", "Medio"], default="Alto")
    seg_std = np.select([prob < medium, prob <= std_high], ["Bajo", "Medio"], default="Alto")
    return pred, np.where(is_vip, seg_vip, seg_std)


def score(df: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    gen = bundle["general"]
    vip = bundle.get("vip")
    segments = bundle.get("vip_segments", [])
    is_vip = (df["monetary"].isin(segments).values if "monetary" in df.columns
              else np.zeros(len(df), dtype=bool))

    prob = np.empty(len(df), dtype=float)
    model_used = np.full(len(df), "general", dtype=object)

    use_vip = vip is not None and is_vip.any()
    std_mask = ~is_vip if use_vip else np.ones(len(df), dtype=bool)

    if std_mask.any():
        X = make_matrix(df.loc[std_mask], gen["features"], gen["medians"])
        prob[std_mask] = gen["model"].predict_proba(X.values)[:, 1]
    if use_vip:
        Xv = make_matrix(df.loc[is_vip], vip["features"], vip["medians"])
        prob[is_vip] = vip["model"].predict_proba(Xv.values)[:, 1]
        model_used[is_vip] = "vip"

    df["prob_churn_hard"] = prob
    df["modelo"] = model_used
    df["pred_churn_hard"], df["segmento_riesgo"] = assign_risk(prob, is_vip, bundle["thresholds"])
    return df
