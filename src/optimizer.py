"""
optimizer.py — M3 channel×funnel budget allocation (SLSQP) + campaign waste flags.

Maximizes forecasted reconciled revenue under guardrails — not ROAS alone.
See workflow.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .forecasting import CurveFit, predict_mroas, predict_revenue
from .ingestion import IngestionResult

DEFAULT_TARGET_ROAS = 4.0
DEFAULT_TARGET_NC_CPA = 50.0
BASE_SHIFT_LO = 0.80
BASE_SHIFT_HI = 1.30
LOW_CONF_SHIFT_LO = 0.90
LOW_CONF_SHIFT_HI = 1.10
LOW_CONF_THRESHOLD = 0.55
# Waste thresholds are relative to reconciled economics (post-haircut aROAS often ~1–2x).
AROAS_BURN_ABS = 0.85          # clearly destroying cash after haircut
AROAS_BURN_REL = 0.65          # below 65% of peer-slice median aROAS
MROAS_HUNGRY_REL = 1.15        # above slice mROAS peer / own slice
AOV_DEFAULT = 72.0
ROAS_PENALTY_WEIGHT = 50_000.0  # soft ROAS penalty scale
NC_CPA_PENALTY_WEIGHT = 5_000.0


@dataclass
class SliceState:
    slice_id: str
    platform: str
    funnel_stage: str
    current_spend: float
    curve: CurveFit
    is_prospecting_like: bool
    shift_lo: float
    shift_hi: float


@dataclass
class WasteFlag:
    flag_type: str  # capped_hungry | cash_burner
    platform: str
    platform_campaign_id: str
    campaign_name: str
    slice_id: str
    spend_usd: float
    reconciled_revenue: float
    aroas: float
    mroas_slice: float
    reason: str
    suggested_action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AllocationPlan:
    feasible: bool
    objective_revenue: float
    slices: list[dict[str, Any]]
    metrics_before: dict[str, float]
    metrics_after: dict[str, float]
    relaxed_constraints: list[str] = field(default_factory=list)
    waste_flags: list[WasteFlag] = field(default_factory=list)
    mutate_payloads: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "objective_revenue": round(self.objective_revenue, 2),
            "slices": self.slices,
            "metrics_before": {k: round(v, 4) for k, v in self.metrics_before.items()},
            "metrics_after": {k: round(v, 4) for k, v in self.metrics_after.items()},
            "relaxed_constraints": self.relaxed_constraints,
            "waste_flags": [w.to_dict() for w in self.waste_flags],
            "mutate_payloads": self.mutate_payloads,
            "message": self.message,
        }


def _shift_bounds(confidence: float) -> tuple[float, float]:
    if confidence < LOW_CONF_THRESHOLD:
        return LOW_CONF_SHIFT_LO, LOW_CONF_SHIFT_HI
    return BASE_SHIFT_LO, BASE_SHIFT_HI


def build_slice_state(curves: list[CurveFit]) -> list[SliceState]:
    states: list[SliceState] = []
    for c in curves:
        lo, hi = _shift_bounds(c.confidence)
        # unknown funnel: freeze tighter regardless
        if c.funnel_stage == "unknown":
            lo, hi = 0.95, 1.05
        states.append(
            SliceState(
                slice_id=c.slice_id,
                platform=c.platform,
                funnel_stage=c.funnel_stage,
                current_spend=c.current_spend,
                curve=c,
                is_prospecting_like=c.funnel_stage in {"prospecting", "non_brand"},
                shift_lo=lo,
                shift_hi=hi,
            )
        )
    return states


def _metrics(
    spends: np.ndarray,
    states: list[SliceState],
    new_customer_rate: float,
    aov: float,
) -> dict[str, float]:
    revenues = np.array([predict_revenue(s.curve, float(sp)) for s, sp in zip(states, spends)])
    total_spend = float(spends.sum())
    total_rev = float(revenues.sum())
    blended_roas = total_rev / total_spend if total_spend > 0 else 0.0
    prospect_spend = float(
        sum(sp for s, sp in zip(states, spends) if s.is_prospecting_like)
    )
    # Forecast new customers from prospecting-like revenue share
    prospect_rev = float(
        sum(
            predict_revenue(s.curve, float(sp))
            for s, sp in zip(states, spends)
            if s.is_prospecting_like
        )
    )
    forecast_new = max((prospect_rev / max(aov, 1.0)) * new_customer_rate, 1e-6)
    nc_cpa = prospect_spend / forecast_new
    return {
        "total_spend": total_spend,
        "total_revenue": total_rev,
        "blended_roas": blended_roas,
        "nc_cpa": nc_cpa,
        "prospect_spend": prospect_spend,
        "forecast_new_customers": forecast_new,
    }


def _estimate_new_customer_rate(ingestion: IngestionResult) -> float:
    shop = ingestion.shopify
    if len(shop) == 0:
        return 0.55
    return float(shop["is_new_customer"].mean())


def optimize_budget(
    states: list[SliceState],
    *,
    target_roas: float = DEFAULT_TARGET_ROAS,
    target_nc_cpa: float = DEFAULT_TARGET_NC_CPA,
    new_customer_rate: float = 0.55,
    aov: float = AOV_DEFAULT,
    soft_roas: bool = True,
    soft_nc_cpa: bool = True,
) -> AllocationPlan:
    if not states:
        return AllocationPlan(
            feasible=False,
            objective_revenue=0.0,
            slices=[],
            metrics_before={},
            metrics_after={},
            message="No slices to optimize",
        )

    x0 = np.array([s.current_spend for s in states], dtype=float)
    total_budget = float(x0.sum())
    bounds = [
        (s.current_spend * s.shift_lo, s.current_spend * s.shift_hi) for s in states
    ]
    before = _metrics(x0, states, new_customer_rate, aov)

    def objective(x: np.ndarray) -> float:
        revenues = np.array([predict_revenue(s.curve, float(sp)) for s, sp in zip(states, x)])
        loss = -float(revenues.sum())
        total_spend = float(x.sum())
        roas = float(revenues.sum()) / total_spend if total_spend > 0 else 0.0
        if soft_roas and roas < target_roas:
            loss += ROAS_PENALTY_WEIGHT * (target_roas - roas) ** 2
        m = _metrics(x, states, new_customer_rate, aov)
        if soft_nc_cpa and m["nc_cpa"] > target_nc_cpa:
            loss += NC_CPA_PENALTY_WEIGHT * ((m["nc_cpa"] - target_nc_cpa) / target_nc_cpa) ** 2
        return loss

    constraints = [
        {
            "type": "eq",
            "fun": lambda x, b=total_budget: float(np.sum(x) - b),
        }
    ]
    if not soft_roas:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x: (
                    (
                        sum(predict_revenue(s.curve, float(sp)) for s, sp in zip(states, x))
                        / max(float(np.sum(x)), 1e-9)
                    )
                    - target_roas
                ),
            }
        )
    if not soft_nc_cpa:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x: target_nc_cpa
                - _metrics(x, states, new_customer_rate, aov)["nc_cpa"],
            }
        )

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    relaxed: list[str] = []
    feasible = bool(result.success)
    x_opt = np.asarray(result.x if result.success else x0, dtype=float)

    # Re-check soft metrics; if still badly off ROAS with soft penalties, note it
    after = _metrics(x_opt, states, new_customer_rate, aov)
    if soft_roas and after["blended_roas"] < target_roas:
        relaxed.append(
            f"blended_roas soft: achieved {after['blended_roas']:.2f}x vs target {target_roas:.1f}x"
        )
    if soft_nc_cpa and after["nc_cpa"] > target_nc_cpa:
        relaxed.append(
            f"nc_cpa soft: achieved ${after['nc_cpa']:.2f} vs target ${target_nc_cpa:.2f}"
        )
    if not result.success:
        feasible = False
        relaxed.append("slsqp_did_not_converge — showing current allocation")
        x_opt = x0
        after = before

    slices_out: list[dict[str, Any]] = []
    for s, sp in zip(states, x_opt):
        net = float(sp - s.current_spend)
        mroas = predict_mroas(s.curve, float(sp))
        if net > 1:
            reason = f"Raise: mROAS {mroas:.2f}x attractive under caps"
        elif net < -1:
            reason = f"Cut: reallocate away (mROAS {mroas:.2f}x / efficiency)"
        else:
            reason = "Hold near current"
        if s.curve.confidence < LOW_CONF_THRESHOLD:
            reason += " [tight caps: low confidence]"
        slices_out.append(
            {
                "slice_id": s.slice_id,
                "platform": s.platform,
                "funnel_stage": s.funnel_stage,
                "current_spend": round(s.current_spend, 2),
                "proposed_spend": round(float(sp), 2),
                "net_shift": round(net, 2),
                "mroas": round(mroas, 3),
                "confidence": round(s.curve.confidence, 3),
                "reason": reason,
            }
        )

    return AllocationPlan(
        feasible=feasible or soft_roas,  # soft mode still yields a reviewable plan
        objective_revenue=after["total_revenue"],
        slices=slices_out,
        metrics_before=before,
        metrics_after=after,
        relaxed_constraints=relaxed,
        message=(
            "Optimized with soft ROAS/NC-CPA penalties + hard budget/shift caps"
            if result.success
            else f"Solver issue: {result.message}"
        ),
    )


def flag_waste(
    ingestion: IngestionResult,
    curves: list[CurveFit],
) -> list[WasteFlag]:
    """Campaign-level capped & hungry / cash burner flags (reconciled economics)."""
    curve_by_slice = {c.slice_id: c for c in curves}
    df = ingestion.unified.copy()
    df["aroas"] = np.where(
        df["spend_usd"] > 0,
        df["reconciled_revenue"] / df["spend_usd"],
        0.0,
    )

    # Trailing 7-day campaign efficiency
    df = df.sort_values("date")
    trail = (
        df.groupby(["platform", "platform_campaign_id", "campaign_name", "slice_id"], as_index=False)
        .tail(7)
        .groupby(["platform", "platform_campaign_id", "campaign_name", "slice_id"], as_index=False)
        .agg(
            spend_usd=("spend_usd", "mean"),
            reconciled_revenue=("reconciled_revenue", "mean"),
            aroas=("aroas", "mean"),
        )
    )

    slice_median_aroas = trail.groupby("slice_id")["aroas"].median().to_dict()
    slice_mean_spend = trail.groupby("slice_id")["spend_usd"].mean().to_dict()

    flags: list[WasteFlag] = []
    for _, row in trail.iterrows():
        slice_id = row["slice_id"]
        curve = curve_by_slice.get(slice_id)
        if curve is None:
            continue
        spend = float(row["spend_usd"])
        rev = float(row["reconciled_revenue"])
        aroas = float(row["aroas"])
        mroas = curve.mroas_current
        name = str(row["campaign_name"])
        peer_aroas = float(slice_median_aroas.get(slice_id, aroas) or aroas)
        peer_spend = float(slice_mean_spend.get(slice_id, spend) or spend)
        name_l = name.lower()

        # Explicit cash-burner naming OR weak absolute/relative efficiency
        is_burn = (
            "cashburn" in name_l
            or "cash_burn" in name_l
            or aroas < AROAS_BURN_ABS
            or (peer_aroas > 0 and aroas < AROAS_BURN_REL * peer_aroas)
        )
        if spend > 0 and is_burn:
            flags.append(
                WasteFlag(
                    flag_type="cash_burner",
                    platform=row["platform"],
                    platform_campaign_id=str(row["platform_campaign_id"]),
                    campaign_name=name,
                    slice_id=slice_id,
                    spend_usd=spend,
                    reconciled_revenue=rev,
                    aroas=aroas,
                    mroas_slice=mroas,
                    reason=(
                        f"aROAS={aroas:.2f}x vs slice median {peer_aroas:.2f}x; "
                        f"slice mROAS={mroas:.2f}x"
                    ),
                    suggested_action="Cap daily budget or pause; fund higher-mROAS slices",
                )
            )
            continue

        # Capped & hungry: strong relative efficiency + constrained spend / hungry name
        hungry_name = "hungry" in name_l or "capped" in name_l
        strong = aroas >= max(peer_aroas * MROAS_HUNGRY_REL, peer_aroas + 0.15)
        constrained = hungry_name or spend < 0.8 * peer_spend
        if strong and constrained:
            flags.append(
                WasteFlag(
                    flag_type="capped_hungry",
                    platform=row["platform"],
                    platform_campaign_id=str(row["platform_campaign_id"]),
                    campaign_name=name,
                    slice_id=slice_id,
                    spend_usd=spend,
                    reconciled_revenue=rev,
                    aroas=aroas,
                    mroas_slice=mroas,
                    reason=(
                        f"aROAS={aroas:.2f}x above peers ({peer_aroas:.2f}x) with "
                        f"constrained spend ${spend:,.0f} (peer ${peer_spend:,.0f})"
                    ),
                    suggested_action="Increase budget funded by cash-burner / low-efficiency slices",
                )
            )
    return flags


def build_mutate_payloads(plan: AllocationPlan, states: list[SliceState]) -> list[dict[str, Any]]:
    """
    Stub platform mutate bodies proportional to slice shifts.
    Campaign-level split is equal-weight within slice for the prototype.
    """
    payloads: list[dict[str, Any]] = []
    state_by_id = {s.slice_id: s for s in states}
    for row in plan.slices:
        state = state_by_id.get(row["slice_id"])
        if state is None:
            continue
        proposed = float(row["proposed_spend"])
        if state.platform == "meta":
            payloads.append(
                {
                    "platform": "meta",
                    "endpoint": "POST /v18.0/{campaign_or_adset_id}",
                    "slice_id": row["slice_id"],
                    "body": {
                        "daily_budget": int(round(proposed * 100)),  # cents
                        "daily_budget_usd": round(proposed, 2),
                    },
                }
            )
        else:
            payloads.append(
                {
                    "platform": "google",
                    "endpoint": "mutateCampaignBudgets",
                    "slice_id": row["slice_id"],
                    "body": {
                        "amount_micros": int(round(proposed * 1_000_000)),
                        "daily_budget_usd": round(proposed, 2),
                    },
                }
            )
    return payloads


def run_optimizer(
    ingestion: IngestionResult,
    curves: list[CurveFit],
    *,
    target_roas: float = DEFAULT_TARGET_ROAS,
    target_nc_cpa: float = DEFAULT_TARGET_NC_CPA,
    soft_roas: bool = True,
    soft_nc_cpa: bool = True,
) -> AllocationPlan:
    states = build_slice_state(curves)
    nc_rate = _estimate_new_customer_rate(ingestion)
    plan = optimize_budget(
        states,
        target_roas=target_roas,
        target_nc_cpa=target_nc_cpa,
        new_customer_rate=nc_rate,
        soft_roas=soft_roas,
        soft_nc_cpa=soft_nc_cpa,
    )
    plan.waste_flags = flag_waste(ingestion, curves)
    plan.mutate_payloads = build_mutate_payloads(plan, states)
    return plan


if __name__ == "__main__":
    from .forecasting import fit_all_slices
    from .ingestion import run_ingestion

    ingestion = run_ingestion()
    curves = fit_all_slices(ingestion.unified)
    plan = run_optimizer(ingestion, curves)
    print("Optimizer plan")
    print(f"  feasible: {plan.feasible}")
    print(f"  message: {plan.message}")
    print(f"  before: { {k: round(v, 3) for k, v in plan.metrics_before.items()} }")
    print(f"  after:  { {k: round(v, 3) for k, v in plan.metrics_after.items()} }")
    if plan.relaxed_constraints:
        print(f"  relaxed: {plan.relaxed_constraints}")
    print("  shifts:")
    for s in plan.slices:
        print(
            f"    {s['slice_id']:22} {s['current_spend']:8.0f} → {s['proposed_spend']:8.0f} "
            f"({s['net_shift']:+.0f})  mROAS={s['mroas']:.2f}  {s['reason']}"
        )
    print(f"  waste flags: {len(plan.waste_flags)}")
    for w in plan.waste_flags:
        print(f"    [{w.flag_type}] {w.campaign_name}: {w.reason}")
    print(f"  mutate payloads: {len(plan.mutate_payloads)}")
