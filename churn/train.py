"""
Entrenamiento del modelo de churn.

    python -m churn.train --config config/config.yaml

Flujo:
  1. Carga de sesiones y construcción del dataset cliente-mes (con targets).
  2. Split temporal:  [ train ........ | calib | validación ]
  3. Modelo general (LR / RF / XGBoost) -> el mejor por AUC en validación.
  4. Modelo VIP específico, calibrado (isotonic) en el bloque `calib`.
  5. Umbral VIP elegido con F-beta sobre las probabilidades del modelo VIP
     en `calib`, y evaluado en validación (datos que no ha visto nadie).
  6. Se guarda un único bundle de artefactos (local o stage de Snowflake).
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             f1_score, precision_recall_curve, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import io
from .config import Config, load_config
from .features import (BASE_FEATURES, build_client_month_df, make_matrix,
                       segment_dummy_columns)
from .scoring import score


# =============================================================================
# Modelos
# =============================================================================
def candidate_models(y_train: np.ndarray, seed: int, use_xgboost: bool,
                     min_samples_leaf: int = 10) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                       random_state=seed)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", max_depth=6,
            min_samples_leaf=min_samples_leaf, random_state=seed, n_jobs=-1,
        ),
    }
    if use_xgboost:
        try:
            from xgboost import XGBClassifier
            ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            models["XGBoost"] = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                scale_pos_weight=ratio, eval_metric="logloss",
                random_state=seed, n_jobs=-1,
            )
        except ImportError:
            print("[INFO] XGBoost no instalado: se usan LR y RF.")
    return models


def fit_select(models: Dict[str, Any], X_tr, y_tr, X_ev, y_ev, label: str) -> Tuple[str, Any, Dict]:
    """Entrena todos los candidatos y devuelve el mejor por AUC en (X_ev, y_ev)."""
    results = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        p_tr = model.predict_proba(X_tr)[:, 1]
        p_ev = model.predict_proba(X_ev)[:, 1]
        results[name] = {
            "auc_train": roc_auc_score(y_tr, p_tr),
            "auc_eval": roc_auc_score(y_ev, p_ev),
            "ap_eval": average_precision_score(y_ev, p_ev),
        }
        r = results[name]
        flag = "  [!] posible overfitting" if r["auc_train"] - r["auc_eval"] > 0.10 else ""
        print(f"  {label} {name:<20} AUC train {r['auc_train']:.3f} | "
              f"AUC eval {r['auc_eval']:.3f} | AP {r['ap_eval']:.3f}{flag}")
    best = max(results, key=lambda k: results[k]["auc_eval"])
    return best, models[best], results


def calibrate_prefit(model, X, y):
    """Calibración isotónica de un modelo ya entrenado (compatible sklearn 1.4 -> 1.9+)."""
    try:
        from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
        cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    except ImportError:
        cal = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
    return cal.fit(X, y)


def feature_importance(model, features: List[str]) -> pd.Series:
    est = model
    if hasattr(est, "calibrated_classifiers_"):          # CalibratedClassifierCV
        est = est.calibrated_classifiers_[0].estimator
        est = getattr(est, "estimator", est)             # FrozenEstimator
    if hasattr(est, "named_steps"):
        est = est.named_steps["clf"]
    if hasattr(est, "feature_importances_"):
        return pd.Series(est.feature_importances_, index=features).sort_values(ascending=False)
    if hasattr(est, "coef_"):
        return pd.Series(np.abs(est.coef_[0]), index=features).sort_values(ascending=False)
    return pd.Series(dtype=float)


def print_importance(imp: pd.Series, title: str, top: int = 15) -> None:
    if imp.empty:
        return
    print(f"\n  Top {top} features — {title}")
    head = imp.head(top)
    for feat, v in head.items():
        print(f"    {feat:<34} {v:.4f}  {'█' * int(v / head.iloc[0] * 30)}")


# =============================================================================
# Umbral VIP
# =============================================================================
def choose_vip_threshold(y: np.ndarray, prob: np.ndarray, cfg: Config) -> Tuple[float, str]:
    """
    Umbral VIP: manual si `cfg.vip_threshold` está fijado; si no, el que
    maximiza F-beta (beta<1 => prioriza precisión, modo "francotirador").
    """
    if cfg.vip_threshold is not None:
        return float(cfg.vip_threshold), "manual"
    if len(y) < cfg.vip_min_calib_rows or y.sum() == 0:
        return float(cfg.vip_threshold_default), "default (pocos datos)"

    prec, rec, thr = precision_recall_curve(y, prob)
    b2 = cfg.vip_fbeta ** 2
    fbeta = (1 + b2) * prec[:-1] * rec[:-1] / (b2 * prec[:-1] + rec[:-1] + 1e-9)
    return float(thr[int(np.argmax(fbeta))]), f"auto F{cfg.vip_fbeta}"


def tradeoff_table(y: np.ndarray, prob: np.ndarray, chosen: float) -> pd.DataFrame:
    rows = []
    for u in np.arange(0.30, 0.96, 0.05):
        mask = prob >= u
        n = int(mask.sum())
        if n == 0:
            continue
        tp = int((mask & (y == 1)).sum())
        rows.append({"umbral": round(u, 2), "n_marcados": n,
                     "precision": tp / n, "recall": tp / max(int(y.sum()), 1),
                     "elegido": "<--" if abs(u - chosen) < 0.025 else ""})
    return pd.DataFrame(rows)


# =============================================================================
# Pipeline
# =============================================================================
def temporal_split(df: pd.DataFrame, test_months: int, calib_months: int):
    months = sorted(df["anio_mes"].unique())
    if len(months) < test_months + calib_months + 2:
        raise ValueError(f"Histórico insuficiente: {len(months)} meses.")
    val_m = months[-test_months:]
    cal_m = months[-(test_months + calib_months):-test_months]
    trn_m = months[:-(test_months + calib_months)]
    return trn_m, cal_m, val_m


def run_training(cfg: Config, session=None, df_sessions: pd.DataFrame = None) -> Dict[str, Any]:
    if df_sessions is None:
        df_sessions = io.load_sessions(cfg, session)

    df_cm = build_client_month_df(
        df_sessions, cfg.source_columns, cfg.churn_hard_days,
        cfg.churn_soft_drop, cfg.churn_valor_drop, inference=False,
    )
    target = f"churn_hard_{cfg.churn_hard_days}d"
    features = [f for f in BASE_FEATURES if f in df_cm.columns] + segment_dummy_columns(df_cm)

    # Solo clientes veteranos (>= 3 meses de histórico) y con target observado
    df_model = df_cm[(df_cm["cliente_nuevo"] == 0) & df_cm[target].notna()].copy()
    trn_m, cal_m, val_m = temporal_split(df_model, cfg.test_months, cfg.calib_months)
    print(f"\n[SPLIT] train {trn_m[0]}->{trn_m[-1]} | calib {cal_m[0]}->{cal_m[-1]} | "
          f"val {val_m[0]}->{val_m[-1]}")

    is_vip = df_model["monetary"].isin(cfg.vip_segments) if "monetary" in df_model else \
        pd.Series(False, index=df_model.index)
    d_trn = df_model[df_model["anio_mes"].isin(trn_m + cal_m)]   # modelo general
    d_val = df_model[df_model["anio_mes"].isin(val_m)]

    # ── Modelo general ───────────────────────────────────────────────────────
    print("\n" + "=" * 70 + "\n  MODELO GENERAL\n" + "=" * 70)
    medians = d_trn[features].median(numeric_only=True)
    X_trn, y_trn = make_matrix(d_trn, features, medians).values, d_trn[target].astype(int).values
    X_val, y_val = make_matrix(d_val, features, medians).values, d_val[target].astype(int).values
    best_name, best_model, res = fit_select(
        candidate_models(y_trn, cfg.seed, cfg.use_xgboost), X_trn, y_trn, X_val, y_val, "GEN"
    )
    print(f"  -> Mejor modelo general: {best_name}")
    print_importance(feature_importance(best_model, features), best_name)

    # ── Modelo VIP ───────────────────────────────────────────────────────────
    vip_bundle = None
    vip_threshold, vip_mode = float(cfg.vip_threshold_default), "default (sin modelo VIP)"
    v_trn = df_model[is_vip & df_model["anio_mes"].isin(trn_m)]
    v_cal = df_model[is_vip & df_model["anio_mes"].isin(cal_m)]
    v_val = df_model[is_vip & df_model["anio_mes"].isin(val_m)]

    if min(len(v_trn), len(v_cal), len(v_val)) >= cfg.vip_min_calib_rows and v_trn[target].nunique() == 2:
        print("\n" + "=" * 70 + f"\n  MODELO VIP  ({', '.join(cfg.vip_segments)})\n" + "=" * 70)
        print(f"  Filas VIP -> train {len(v_trn):,} | calib {len(v_cal):,} | val {len(v_val):,}")
        feats_vip = [f for f in features if f not in set(cfg.vip_excluded_features)]
        med_vip = v_trn[feats_vip].median(numeric_only=True)
        Xv_trn, yv_trn = make_matrix(v_trn, feats_vip, med_vip).values, v_trn[target].astype(int).values
        Xv_cal, yv_cal = make_matrix(v_cal, feats_vip, med_vip).values, v_cal[target].astype(int).values
        Xv_val, yv_val = make_matrix(v_val, feats_vip, med_vip).values, v_val[target].astype(int).values

        # Selección del modelo VIP en calib (validación queda intacta)
        vip_name, vip_raw, _ = fit_select(
            candidate_models(yv_trn, cfg.seed, cfg.use_xgboost, min_samples_leaf=5),
            Xv_trn, yv_trn, Xv_cal, yv_cal, "VIP",
        )
        vip_model = calibrate_prefit(vip_raw, Xv_cal, yv_cal)
        p_cal = vip_model.predict_proba(Xv_cal)[:, 1]
        p_val = vip_model.predict_proba(Xv_val)[:, 1]

        vip_threshold, vip_mode = choose_vip_threshold(yv_cal, p_cal, cfg)
        print(f"  -> Mejor modelo VIP: {vip_name} (+ calibración isotónica)")
        print(f"  -> AUC val VIP: {roc_auc_score(yv_val, p_val):.3f} | "
              f"AP: {average_precision_score(yv_val, p_val):.3f}")
        print(f"  -> Umbral VIP: {vip_threshold:.3f}  [{vip_mode}]")
        print(f"     F1 val con umbral VIP: "
              f"{f1_score(yv_val, (p_val >= vip_threshold).astype(int), zero_division=0):.3f}")
        print("\n  Trade-off en validación (precisión / recall / volumen):")
        print(tradeoff_table(yv_val, p_val, vip_threshold).to_string(index=False,
              float_format=lambda x: f"{x:.3f}"))
        print_importance(feature_importance(vip_model, feats_vip), f"{vip_name} VIP")

        vip_bundle = {"name": vip_name, "model": vip_model,
                      "features": feats_vip, "medians": med_vip}
    else:
        print("\n[AVISO] Datos VIP insuficientes: se usará el modelo general para todos.")

    # ── Reporte final en validación con la lógica real de producción ─────────
    print("\n" + "=" * 70 + "\n  VALIDACIÓN (lógica de producción)\n" + "=" * 70)
    thresholds = {"vip_high": vip_threshold, "std_high": cfg.std_threshold_high,
                  "medium": cfg.threshold_medium}
    bundle = {
        "general": {"name": best_name, "model": best_model,
                    "features": features, "medians": medians},
        "vip": vip_bundle,
        "vip_segments": list(cfg.vip_segments),
        "thresholds": thresholds,
        "vip_threshold_mode": vip_mode,
        "target": target,
        "metrics_general": res,
        "train_months": {"train": trn_m, "calib": cal_m, "val": val_m},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    scored = score(d_val.copy(), bundle)
    for seg_name, mask in [("VIP", scored["monetary"].isin(cfg.vip_segments)),
                           ("Resto", ~scored["monetary"].isin(cfg.vip_segments))]:
        if mask.sum() == 0:
            continue
        print(f"\n  -- {seg_name} (n={mask.sum():,}) --")
        print(classification_report(scored.loc[mask, target].astype(int),
                                    scored.loc[mask, "pred_churn_hard"],
                                    target_names=["No churn", "Churn"], digits=3,
                                    zero_division=0))
    print("  Tasa de churn real por segmento de riesgo (validación):")
    print(scored.groupby("segmento_riesgo")[target].agg(total="count", tasa_real="mean").round(3))

    _save_plots(cfg, y_val, scored["prob_churn_hard"].values, target)
    io.save_artifacts(bundle, cfg, session)
    return bundle


def _save_plots(cfg: Config, y: np.ndarray, prob: np.ndarray, target: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
    except ImportError:
        return
    os.makedirs(cfg.reports_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    RocCurveDisplay.from_predictions(y, prob, ax=axes[0])
    axes[0].set_title("ROC (validación)")
    PrecisionRecallDisplay.from_predictions(y, prob, ax=axes[1])
    axes[1].set_title("Precisión-Recall (validación)")
    for val, label, color in [(0, "No churn", "#1D9E75"), (1, "Churn", "#D85A30")]:
        axes[2].hist(prob[y == val], bins=25, alpha=0.6, label=label, color=color)
    axes[2].axvline(cfg.std_threshold_high, color="gray", ls="--", lw=1, label="Umbral alto")
    axes[2].set_title("Distribución de probabilidades")
    axes[2].legend()
    fig.suptitle(f"Validación — {target}")
    fig.tight_layout()
    path = os.path.join(cfg.reports_dir, "validacion.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] Gráfico de validación: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo de churn")
    parser.add_argument("--config", default=None, help="Ruta al YAML de configuración")
    args = parser.parse_args()

    cfg = load_config(args.config)
    session = io.create_snowflake_session(cfg) if cfg.mode == "snowflake" else None
    try:
        run_training(cfg, session)
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    main()
