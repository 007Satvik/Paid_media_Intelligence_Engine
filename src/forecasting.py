"""
forecasting.py — M2 Hill spend-response / saturation engine + mROAS.

Not a full time-series forecaster (Prophet/XGBoost deferred).
See workflow.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .ingestion import IngestionResult


@dataclass
class CurveFit:
    slice_id: str
    platform: str
    funnel_stage: str
    beta: float
    k: float
    gamma: float
    r2: float
    n_points: int
    current_spend: float
    predicted_revenue: float
    mroas_current: float
    aroas_current: float
    ci_low: float
    ci_high: float
    confidence: float
    residual_std: float
    notes: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key, val in list(d.items()):
            if isinstance(val, float):
                d[key] = round(val, 6) if abs(val) < 1e3 else round(val, 2)
        return d


def hill(s: np.ndarray | float, beta: float, k: float, gamma: float) -> np.ndarray | float:
    s = np.asarray(s, dtype=float)
    s = np.maximum(s, 0.0)
    return beta * np.power(s, gamma) / (np.power(k, gamma) + np.power(s, gamma))


def hill_mroas(s: float, beta: float, k: float, gamma: float) -> float:
    """Analytic derivative dR/dS of the Hill function."""
    if s <= 0:
        # limit behavior: use small epsilon
        s = 1e-6
    k_g = k**gamma
    s_g = s**gamma
    denom = k_g + s_g
    # R = beta * s^g / denom
    # R' = beta * g * s^(g-1) * k^g / denom^2
    return float(beta * gamma * (s ** (gamma - 1.0)) * k_g / (denom**2))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def fit_hill(spend: np.ndarray, revenue: np.ndarray) -> tuple[float, float, float, float, float]:
    """
    Returns beta, k, gamma, r2, residual_std.
    """
    spend = np.asarray(spend, dtype=float)
    revenue = np.asarray(revenue, dtype=float)
    mask = np.isfinite(spend) & np.isfinite(revenue) & (spend >= 0) & (revenue >= 0)
    spend = spend[mask]
    revenue = revenue[mask]
    if len(spend) < 4:
        raise ValueError("need at least 4 points to fit Hill curve")

    beta0 = max(float(np.percentile(revenue, 90)), float(revenue.max()), 1.0)
    k0 = max(float(np.median(spend)), 1.0)
    gamma0 = 1.4
    bounds = ([1e-3, 1e-3, 0.3], [beta0 * 8 + 1.0, max(spend) * 8 + 1.0, 4.0])

    params, _ = curve_fit(
        hill,
        spend,
        revenue,
        p0=[beta0, k0, gamma0],
        bounds=bounds,
        maxfev=20000,
    )
    beta, k, gamma = (float(params[0]), float(params[1]), float(params[2]))
    pred = np.asarray(hill(spend, beta, k, gamma), dtype=float)
    resid = revenue - pred
    return beta, k, gamma, _r2(revenue, pred), float(np.std(resid))


def _slice_confidence(
    r2: float,
    n_points: int,
    residual_std: float,
    mean_revenue: float,
    upstream_flag_rate: float,
) -> float:
    """Heuristic confidence in [0, 1] for M3 shift caps."""
    r2_term = clamp(r2, 0.0, 1.0)
    n_term = clamp(n_points / 21.0, 0.0, 1.0)
    cv = residual_std / max(mean_revenue, 1.0)
    noise_term = clamp(1.0 - cv, 0.0, 1.0)
    flag_term = clamp(1.0 - upstream_flag_rate, 0.0, 1.0)
    return float(0.4 * r2_term + 0.2 * n_term + 0.2 * noise_term + 0.2 * flag_term)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def aggregate_slice_daily(unified: pd.DataFrame) -> pd.DataFrame:
    return (
        unified.groupby(["date", "slice_id", "platform", "funnel_stage"], as_index=False)
        .agg(
            spend_usd=("spend_usd", "sum"),
            reconciled_revenue=("reconciled_revenue", "sum"),
            platform_revenue=("platform_revenue", "sum"),
            n_flagged=(
                "flags",
                lambda s: sum(1 for flags in s if flags),
            ),
            n_rows=("campaign_name", "count"),
        )
    )


def fit_slice(slice_df: pd.DataFrame, slice_id: str, platform: str, funnel: str) -> CurveFit:
    spend = slice_df["spend_usd"].to_numpy(dtype=float)
    rev = slice_df["reconciled_revenue"].to_numpy(dtype=float)
    flag_rate = float(slice_df["n_flagged"].sum()) / max(float(slice_df["n_rows"].sum()), 1.0)

    notes = ""
    try:
        beta, k, gamma, r2, resid_std = fit_hill(spend, rev)
    except Exception as exc:  # noqa: BLE001 — prototype: fall back to linear-ish Hill
        notes = f"fit_fallback: {exc}"
        beta = max(float(rev.max()) * 1.2, 1.0)
        k = max(float(np.median(spend)), 1.0)
        gamma = 1.2
        pred = np.asarray(hill(spend, beta, k, gamma), dtype=float)
        r2 = _r2(rev, pred)
        resid_std = float(np.std(rev - pred))

    # Current spend = mean of last 3 days (stable demo point)
    tail = slice_df.sort_values("date").tail(3)
    current_spend = float(tail["spend_usd"].mean())
    pred_rev = float(hill(current_spend, beta, k, gamma))
    mroas = hill_mroas(current_spend, beta, k, gamma)
    aroas = pred_rev / current_spend if current_spend > 0 else 0.0
    z = 1.645  # ~90%
    ci_low = max(0.0, pred_rev - z * resid_std)
    ci_high = pred_rev + z * resid_std
    conf = _slice_confidence(r2, len(slice_df), resid_std, float(np.mean(rev)), flag_rate)
    if notes:
        conf *= 0.7

    return CurveFit(
        slice_id=slice_id,
        platform=platform,
        funnel_stage=funnel,
        beta=beta,
        k=k,
        gamma=gamma,
        r2=r2,
        n_points=len(slice_df),
        current_spend=current_spend,
        predicted_revenue=pred_rev,
        mroas_current=mroas,
        aroas_current=aroas,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=conf,
        residual_std=resid_std,
        notes=notes or "hill_curve_fit",
    )


def fit_all_slices(unified: pd.DataFrame) -> list[CurveFit]:
    daily = aggregate_slice_daily(unified)
    curves: list[CurveFit] = []
    for slice_id, grp in daily.groupby("slice_id"):
        platform = str(grp["platform"].iloc[0])
        funnel = str(grp["funnel_stage"].iloc[0])
        if funnel == "unknown" and float(grp["spend_usd"].sum()) < 1:
            continue
        curves.append(fit_slice(grp, str(slice_id), platform, funnel))
    curves.sort(key=lambda c: c.slice_id)
    return curves


def fit_from_ingestion(ingestion: IngestionResult) -> list[CurveFit]:
    return fit_all_slices(ingestion.unified)


def predict_revenue(curve: CurveFit, spend: float) -> float:
    return float(hill(spend, curve.beta, curve.k, curve.gamma))


def predict_mroas(curve: CurveFit, spend: float) -> float:
    return hill_mroas(spend, curve.beta, curve.k, curve.gamma)


if __name__ == "__main__":
    from .ingestion import run_ingestion

    ingestion = run_ingestion()
    curves = fit_all_slices(ingestion.unified)
    print(f"Fitted {len(curves)} slice curves")
    for c in curves:
        print(
            f"  {c.slice_id:22} spend=${c.current_spend:,.0f}  "
            f"R=${c.predicted_revenue:,.0f}  mROAS={c.mroas_current:.2f}  "
            f"aROAS={c.aroas_current:.2f}  R²={c.r2:.3f}  conf={c.confidence:.2f}"
        )
