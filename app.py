"""
True Classic Media Control Room — Streamlit operator UI.

Run from repo root:
  .venv/bin/streamlit run app.py

Walks M1 → M2 → M3 live: ingestion (+ LLM agent reconcile), Hill curves,
budget plan, waste flags, human Approve/Reject with stubbed mutate payloads.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.agent import agent_available
from src.forecasting import CurveFit, fit_all_slices, hill
from src.ingestion import IngestionResult, run_ingestion
from src.optimizer import AllocationPlan, run_optimizer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
load_dotenv(ROOT / ".env")

st.set_page_config(
    page_title="True Classic Media Control Room",
    page_icon="TC",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal operator-console styling (avoid generic purple/AI chrome)
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; max-width: 1200px; }
      div[data-testid="stMetricValue"] { font-size: 1.35rem; }
      .tc-banner {
        background: linear-gradient(120deg, #0f1c2e 0%, #1a334d 55%, #243b55 100%);
        color: #e8eef5;
        padding: 1.1rem 1.35rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        border-left: 4px solid #c4a35a;
      }
      .tc-banner h1 {
        font-size: 1.45rem; margin: 0 0 0.25rem 0;
        font-weight: 650; letter-spacing: 0.02em;
      }
      .tc-banner p { margin: 0; opacity: 0.85; font-size: 0.92rem; }
      .tc-assume {
        background: #f4f1ea; border: 1px solid #d9d2c5; color: #3d3428;
        padding: 0.65rem 0.9rem; border-radius: 4px; font-size: 0.88rem;
        margin: 0.5rem 0 1rem 0;
      }
      .ok { color: #1b7a3d; font-weight: 600; }
      .warn { color: #a15c00; font-weight: 600; }
      .bad { color: #a11d1d; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def _fmt_roas(x: float) -> str:
    return f"{x:.2f}x"


def _status_class(ok: bool) -> str:
    return "ok" if ok else "warn"


@st.cache_data(show_spinner=False)
def load_pipeline(agent_enabled_flag: str, data_mtime: float) -> dict:
    """
    Run M1→M2→M3 once per cache key.
    data_mtime busts cache when fixtures change; agent flag when .env toggle changes.
    """
    _ = data_mtime
    os.environ["AGENT_ENABLED"] = agent_enabled_flag
    ingestion = run_ingestion(DATA_DIR)
    curves = fit_all_slices(ingestion.unified)
    # Default targets; UI can re-optimize without re-ingesting
    plan = run_optimizer(ingestion, curves, target_roas=4.0, target_nc_cpa=50.0)
    return {
        "ingestion": ingestion,
        "curves": curves,
        "plan": plan,
    }


def _data_mtime() -> float:
    files = [
        DATA_DIR / "raw_meta_export.json",
        DATA_DIR / "raw_google_export.json",
        DATA_DIR / "shopify_orders.csv",
        DATA_DIR / "id_map.csv",
    ]
    return max((f.stat().st_mtime for f in files if f.exists()), default=0.0)


def render_header(target_roas: float, target_nc_cpa: float) -> None:
    st.markdown(
        f"""
        <div class="tc-banner">
          <h1>TRUE CLASSIC — MEDIA CONTROL ROOM</h1>
          <p>Operator console · Meta + Google · Target ROAS {target_roas:.1f}x · NC-CPA ${target_nc_cpa:.0f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_assumption_badge(ingestion: IngestionResult, plan: AllocationPlan) -> None:
    hair = "; ".join(h.before_after_note for h in ingestion.haircut_reports)
    st.markdown(
        f"""
        <div class="tc-assume">
          <strong>Recommendation under stated assumptions</strong> —
          Shopify haircut applied ({hair}).
          Hill spend-response curves + soft ROAS/NC-CPA penalties + hard shift caps.
          Agent: {ingestion.agent_status}.
          {plan.message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(plan: AllocationPlan, target_roas: float, target_nc_cpa: float) -> None:
    before = plan.metrics_before
    after = plan.metrics_after
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Current")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Daily budget", _fmt_money(before.get("total_spend", 0)))
        m2.metric("Blended ROAS", _fmt_roas(before.get("blended_roas", 0)))
        m3.metric("NC-CPA", f"${before.get('nc_cpa', 0):.2f}")
        m4.metric("Revenue", _fmt_money(before.get("total_revenue", 0)))
        roas_ok = before.get("blended_roas", 0) >= target_roas
        cpa_ok = before.get("nc_cpa", 999) <= target_nc_cpa
        st.markdown(
            f"ROAS target: <span class='{_status_class(roas_ok)}'>"
            f"{'MET' if roas_ok else 'BELOW'}</span> · "
            f"NC-CPA: <span class='{_status_class(cpa_ok)}'>"
            f"{'MET' if cpa_ok else 'ABOVE'}</span>",
            unsafe_allow_html=True,
        )
    with c2:
        st.subheader("After reallocation (forecast)")
        delta_rev = after.get("total_revenue", 0) - before.get("total_revenue", 0)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Daily budget", _fmt_money(after.get("total_spend", 0)))
        m2.metric(
            "Blended ROAS",
            _fmt_roas(after.get("blended_roas", 0)),
            delta=f"{after.get('blended_roas', 0) - before.get('blended_roas', 0):+.2f}x",
        )
        m3.metric(
            "NC-CPA",
            f"${after.get('nc_cpa', 0):.2f}",
            delta=f"{after.get('nc_cpa', 0) - before.get('nc_cpa', 0):+.2f}",
        )
        m4.metric("Revenue", _fmt_money(after.get("total_revenue", 0)), delta=_fmt_money(delta_rev))
        roas_ok = after.get("blended_roas", 0) >= target_roas
        cpa_ok = after.get("nc_cpa", 999) <= target_nc_cpa
        st.markdown(
            f"ROAS target: <span class='{_status_class(roas_ok)}'>"
            f"{'MET' if roas_ok else 'BELOW (soft)'}</span> · "
            f"NC-CPA: <span class='{_status_class(cpa_ok)}'>"
            f"{'MET' if cpa_ok else 'ABOVE (soft)'}</span>",
            unsafe_allow_html=True,
        )


def render_shifts(plan: AllocationPlan) -> None:
    st.subheader("Recommended shifts (channel × funnel)")
    if not plan.slices:
        st.info("No slice recommendations.")
        return
    df = pd.DataFrame(plan.slices)
    show = df[
        [
            "slice_id",
            "current_spend",
            "proposed_spend",
            "net_shift",
            "mroas",
            "confidence",
            "reason",
        ]
    ].rename(
        columns={
            "slice_id": "Channel × funnel",
            "current_spend": "Current $",
            "proposed_spend": "Proposed $",
            "net_shift": "Net shift $",
            "mroas": "mROAS",
            "confidence": "Conf.",
            "reason": "Reason",
        }
    )
    st.dataframe(
        show.style.format(
            {
                "Current $": "${:,.0f}",
                "Proposed $": "${:,.0f}",
                "Net shift $": "${:+,.0f}",
                "mROAS": "{:.2f}x",
                "Conf.": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    if plan.relaxed_constraints:
        st.warning("Soft / relaxed constraints:\n- " + "\n- ".join(plan.relaxed_constraints))


def render_curves(curves: list[CurveFit]) -> None:
    st.subheader("M2 — Spend-response curves (Hill)")
    if not curves:
        st.info("No curves fitted.")
        return
    fig = go.Figure()
    colors = ["#1a334d", "#c4a35a", "#3d7a6a", "#8b4513", "#5c6b7a", "#a15c00"]
    for i, c in enumerate(curves):
        s_max = max(c.current_spend * 2.2, c.k * 2.5, 500.0)
        xs = np.linspace(0, s_max, 80)
        ys = hill(xs, c.beta, c.k, c.gamma)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=f"{c.slice_id} (mROAS {c.mroas_current:.2f})",
                line=dict(color=colors[i % len(colors)], width=2.4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[c.current_spend],
                y=[c.predicted_revenue],
                mode="markers",
                name=f"{c.slice_id} now",
                marker=dict(size=10, color=colors[i % len(colors)], symbol="diamond"),
                showlegend=False,
                hovertext=f"CI [{c.ci_low:,.0f}–{c.ci_high:,.0f}] R²={c.r2:.2f}",
            )
        )
    fig.update_layout(
        height=380,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Daily spend ($)",
        yaxis_title="Reconciled revenue ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="#faf9f7",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    cdf = pd.DataFrame(
        [
            {
                "Slice": c.slice_id,
                "Current spend": c.current_spend,
                "Pred. revenue": c.predicted_revenue,
                "mROAS": c.mroas_current,
                "aROAS": c.aroas_current,
                "R²": c.r2,
                "Conf.": c.confidence,
                "CI low": c.ci_low,
                "CI high": c.ci_high,
            }
            for c in curves
        ]
    )
    st.dataframe(
        cdf.style.format(
            {
                "Current spend": "${:,.0f}",
                "Pred. revenue": "${:,.0f}",
                "mROAS": "{:.2f}x",
                "aROAS": "{:.2f}x",
                "R²": "{:.3f}",
                "Conf.": "{:.2f}",
                "CI low": "${:,.0f}",
                "CI high": "${:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_ingestion(ingestion: IngestionResult) -> None:
    st.subheader("M1 — Ingestion, haircut & agent reconciliation")
    a, b, c, d = st.columns(4)
    a.metric("Unified rows", f"{len(ingestion.unified):,}")
    b.metric("Total spend", _fmt_money(ingestion.total_spend))
    c.metric("Platform revenue", _fmt_money(float(ingestion.unified["platform_revenue"].sum())))
    d.metric("Reconciled (=Shopify scale)", _fmt_money(ingestion.total_reconciled_revenue))

    st.markdown("**Haircut (before → after)**")
    for h in ingestion.haircut_reports:
        st.write(f"- {h.before_after_note}")

    st.markdown("**Data-quality flags**")
    if ingestion.flags_summary:
        st.json(ingestion.flags_summary)
    else:
        st.write("None")

    st.markdown("**LLM agent status**")
    st.code(ingestion.agent_status or "(none)")
    if ingestion.agent_matches:
        rows = []
        for m in ingestion.agent_matches:
            rows.append(
                {
                    "type": m.get("query_type"),
                    "query": m.get("query"),
                    "matched_id": m.get("matched_id"),
                    "confidence": m.get("confidence"),
                    "applied": m.get("applied"),
                    "provider": f"{m.get('provider')}/{m.get('model')}" if m.get("provider") else "",
                    "rationale": (m.get("rationale") or m.get("error") or "")[:120],
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption(
            "No agent match attempts in this run "
            "(set OPENAI_API_KEY or Ollama + AGENT_ENABLED=1, then Refresh pipeline)."
        )

    with st.expander("Taxonomy tags"):
        tax = pd.DataFrame([t.to_dict() for t in ingestion.taxonomy_results])
        if not tax.empty:
            st.dataframe(tax, use_container_width=True, hide_index=True)

    with st.expander("Unified sample (latest day per campaign)"):
        u = ingestion.unified.sort_values("date").groupby(
            ["platform", "platform_campaign_id", "campaign_name"], as_index=False
        ).tail(1)
        cols = [
            c
            for c in [
                "date",
                "platform",
                "campaign_name",
                "map_status",
                "unified_campaign_id",
                "funnel_stage",
                "spend_usd",
                "platform_revenue",
                "reconciled_revenue",
                "taxonomy_source",
            ]
            if c in u.columns
        ]
        st.dataframe(u[cols], use_container_width=True, hide_index=True)


def render_waste(plan: AllocationPlan) -> None:
    st.subheader("Waste & underspend flags (campaign-level)")
    if not plan.waste_flags:
        st.success("No waste flags on latest window.")
        return
    df = pd.DataFrame([w.to_dict() for w in plan.waste_flags])
    st.dataframe(
        df[
            [
                "flag_type",
                "platform",
                "campaign_name",
                "slice_id",
                "spend_usd",
                "aroas",
                "mroas_slice",
                "reason",
                "suggested_action",
            ]
        ].rename(
            columns={
                "flag_type": "Flag",
                "campaign_name": "Campaign",
                "slice_id": "Slice",
                "spend_usd": "Spend $",
                "aroas": "aROAS",
                "mroas_slice": "Slice mROAS",
                "reason": "Reason",
                "suggested_action": "Action",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_approval(plan: AllocationPlan) -> None:
    st.subheader("Human gate — Approve / Reject")
    st.caption("No auto-push. Approve only materializes stubbed platform mutate payloads.")

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        approve = st.button("✅ Approve & show mutate payloads", type="primary", use_container_width=True)
    with col_b:
        reject = st.button("❌ Reject plan", use_container_width=True)

    if reject:
        st.session_state["decision"] = "rejected"
        st.session_state.pop("approved_payloads", None)
    if approve:
        st.session_state["decision"] = "approved"
        st.session_state["approved_payloads"] = plan.mutate_payloads

    decision = st.session_state.get("decision")
    if decision == "rejected":
        st.error("Plan rejected. Tweak constraints in the sidebar and re-optimize.")
    elif decision == "approved":
        st.success("Plan approved (stub only). Downstream API bodies:")
        payloads = st.session_state.get("approved_payloads") or plan.mutate_payloads
        st.code(json.dumps(payloads, indent=2), language="json")


def main() -> None:
    # ---- Sidebar controls ----
    with st.sidebar:
        st.header("Controls")
        agent_flag = os.getenv("AGENT_ENABLED", "1").strip() or "1"
        agent_toggle = st.toggle(
            "Enable LLM agent (M1 leftovers)",
            value=agent_flag not in {"0", "false", "False", "no"},
            help="Uses OPENAI_API_KEY or Ollama per .env LLM_PROVIDER",
        )
        ok, agent_msg = agent_available()
        st.caption(f"Agent preflight: {'ready — ' + agent_msg if ok else agent_msg}")

        target_roas = st.slider("Target blended ROAS", 1.0, 6.0, 4.0, 0.1)
        target_nc_cpa = st.slider("Target NC-CPA ($)", 20.0, 120.0, 50.0, 1.0)
        soft_roas = st.checkbox("Soft ROAS constraint", value=True)
        soft_nc = st.checkbox("Soft NC-CPA constraint", value=True)

        refresh = st.button("🔄 Refresh pipeline (M1→M2)", use_container_width=True)
        reopt = st.button("🧮 Re-optimize (M3 only)", use_container_width=True, type="primary")

        st.divider()
        st.caption(f"Data dir: `{DATA_DIR}`")
        st.caption(f"LLM_PROVIDER={os.getenv('LLM_PROVIDER', 'openai')}")

    if refresh:
        load_pipeline.clear()
        st.session_state.pop("decision", None)
        st.session_state.pop("approved_payloads", None)
        st.session_state.pop("plan_override", None)

    agent_enabled_str = "1" if agent_toggle else "0"
    with st.spinner("Running ingestion → forecasting…"):
        bundle = load_pipeline(agent_enabled_str, _data_mtime())

    ingestion: IngestionResult = bundle["ingestion"]
    curves: list[CurveFit] = bundle["curves"]

    # Re-optimize on demand or first load
    if reopt or "plan_override" not in st.session_state:
        with st.spinner("Solving budget allocation (SLSQP)…"):
            plan = run_optimizer(
                ingestion,
                curves,
                target_roas=target_roas,
                target_nc_cpa=target_nc_cpa,
                soft_roas=soft_roas,
                soft_nc_cpa=soft_nc,
            )
            st.session_state["plan_override"] = plan
    plan: AllocationPlan = st.session_state["plan_override"]

    render_header(target_roas, target_nc_cpa)
    render_assumption_badge(ingestion, plan)
    render_metrics(plan, target_roas, target_nc_cpa)
    st.divider()
    render_shifts(plan)
    st.divider()
    render_approval(plan)
    st.divider()

    tab_curves, tab_waste, tab_ingest = st.tabs(
        ["M2 Curves", "Waste flags", "M1 Ingestion & Agent"]
    )
    with tab_curves:
        render_curves(curves)
    with tab_waste:
        render_waste(plan)
    with tab_ingest:
        render_ingestion(ingestion)


if __name__ == "__main__":
    main()
