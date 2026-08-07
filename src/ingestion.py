"""
ingestion.py — M1 deterministic unification + agentic fuzzy reconciliation + haircut.

Dollar math is deterministic. LLM agent (agent.py) only fuzzy-matches leftover
campaign names / SKUs when id_map misses.
See workflow.md.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .taxonomy import TaxonomyResult, load_overrides, parse_campaign_name

META_PURCHASE = "offsite_conversion.fb_pixel_purchase"
HAIRCUT_LO = 0.25
HAIRCUT_HI = 1.0
LOW_TAXONOMY_CONF = 0.6


@dataclass
class HaircutReport:
    platform: str
    platform_revenue_sum: float
    shopify_allocated_sum: float
    scale: float
    scale_clamped: bool
    before_after_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "platform_revenue_sum": round(self.platform_revenue_sum, 2),
            "shopify_allocated_sum": round(self.shopify_allocated_sum, 2),
            "scale": round(self.scale, 4),
            "scale_clamped": self.scale_clamped,
            "before_after_note": self.before_after_note,
        }


@dataclass
class IngestionResult:
    unified: pd.DataFrame
    shopify: pd.DataFrame
    shopify_daily: pd.DataFrame
    haircut_reports: list[HaircutReport]
    flags_summary: dict[str, int]
    taxonomy_results: list[TaxonomyResult] = field(default_factory=list)
    agent_matches: list[dict[str, Any]] = field(default_factory=list)
    agent_status: str = ""

    @property
    def total_spend(self) -> float:
        return float(self.unified["spend_usd"].sum())

    @property
    def total_reconciled_revenue(self) -> float:
        return float(self.unified["reconciled_revenue"].sum())


def _repo_data_dir(data_dir: str | Path | None = None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    return Path(__file__).resolve().parent.parent / "data"


def _agent_enabled() -> bool:
    return os.getenv("AGENT_ENABLED", "1").strip() not in {"0", "false", "False", "no"}


def load_meta(path: str | Path) -> pd.DataFrame:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for r in rows:
        actions = r.get("actions") or []
        action_values = r.get("action_values") or []
        purchases = 0.0
        platform_revenue = 0.0
        has_purchase = False
        for a in actions:
            if a.get("action_type") == META_PURCHASE:
                purchases = float(a.get("value") or 0)
                has_purchase = True
        for av in action_values:
            if av.get("action_type") == META_PURCHASE:
                platform_revenue = float(av.get("value") or 0)
                has_purchase = True

        date_start = r.get("date_start")
        date_stop = r.get("date_stop")
        flags: list[str] = []
        if date_start != date_stop:
            flags.append("date_mismatch")
        spend = float(r.get("spend") or 0)
        if spend > 0 and not has_purchase:
            flags.append("missing_purchase_action")
        if spend == 0:
            flags.append("zero_spend")

        records.append(
            {
                "date": date_start,
                "platform": "meta",
                "platform_campaign_id": str(r.get("campaign_id")),
                "campaign_name": r.get("campaign_name"),
                "spend_usd": spend,
                "impressions": float(r.get("impressions") or 0),
                "clicks": float(r.get("clicks") or 0),
                "platform_purchases": purchases,
                "platform_revenue": platform_revenue,
                "flags": flags,
            }
        )
    return pd.DataFrame.from_records(records)


def load_google(path: str | Path) -> pd.DataFrame:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for r in rows:
        campaign = r.get("campaign") or {}
        metrics = r.get("metrics") or {}
        segments = r.get("segments") or {}
        spend = float(metrics.get("costMicros") or 0) / 1_000_000.0
        flags: list[str] = []
        if spend == 0:
            flags.append("zero_spend")
        records.append(
            {
                "date": segments.get("date"),
                "platform": "google",
                "platform_campaign_id": str(campaign.get("id")),
                "campaign_name": campaign.get("name"),
                "spend_usd": spend,
                "impressions": float(metrics.get("impressions") or 0),
                "clicks": float(metrics.get("clicks") or 0),
                "platform_purchases": float(metrics.get("conversions") or 0),
                "platform_revenue": float(metrics.get("conversionsValue") or 0),
                "flags": flags,
            }
        )
    return pd.DataFrame.from_records(records)


def load_shopify(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["revenue"] = df["revenue"].astype(float)
    df["is_new_customer"] = (
        df["is_new_customer"].astype(str).str.lower().isin(["true", "1", "yes"])
    )
    return df


def load_id_map(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    return df


def _merge_flags(existing: list[str], extra: list[str]) -> list[str]:
    out = list(existing)
    for f in extra:
        if f not in out:
            out.append(f)
    return out


def _build_campaign_catalog(id_map: pd.DataFrame) -> list[Any]:
    """
    Catalog = mapped campaigns + hint-based pending entities for UNMAPPED rows.

    Pending rows give the agent something concrete to match leftovers to
    (e.g. Meta TOF Underwear → pending_meta_prospecting_underwear) without
    putting the exact query string into the catalog.
    """
    from .agent import CatalogEntity

    catalog: list[CatalogEntity] = []
    mapped = id_map[
        (id_map["platform"].isin(["meta", "google"]))
        & (id_map["map_status"] == "mapped")
        & (id_map["unified_campaign_id"].astype(str).str.len() > 0)
    ]
    for _, row in mapped.iterrows():
        catalog.append(
            CatalogEntity(
                entity_id=str(row["unified_campaign_id"]),
                name=str(row["campaign_name"]),
                entity_type="campaign",
                platform=str(row["platform"]),
                unified_sku=str(row.get("unified_sku") or ""),
                funnel_stage=str(row.get("funnel_stage_hint") or ""),
                product_category=str(row.get("product_hint") or ""),
            )
        )

    unmapped = id_map[
        (id_map["platform"].isin(["meta", "google"]))
        & (
            (id_map["map_status"].astype(str).str.upper() == "UNMAPPED")
            | (id_map["unified_campaign_id"].astype(str).str.len() == 0)
        )
    ]
    sku_for_product = {
        "tees": "TC-TEE-CREW",
        "shirts": "TC-SHIRT-OXFORD",
        "underwear": "TC-UND-BOXER",
        "mixed": "TC-MIXED",
    }
    for _, row in unmapped.iterrows():
        platform = str(row["platform"])
        funnel = str(row.get("funnel_stage_hint") or "unknown").strip() or "unknown"
        product = str(row.get("product_hint") or "unknown").strip() or "unknown"
        # Descriptive label ≠ raw campaign_name (avoids trivial exact self-match)
        pending_id = f"pending_{platform}_{funnel}_{product}"
        pending_name = f"{platform.title()} {funnel.replace('_', ' ')} {product} campaigns"
        catalog.append(
            CatalogEntity(
                entity_id=pending_id,
                name=pending_name,
                entity_type="campaign",
                platform=platform,
                unified_sku=sku_for_product.get(product, ""),
                funnel_stage=funnel,
                product_category=product,
                extra={"pending": True, "source_campaign_name": str(row.get("campaign_name") or "")},
            )
        )
    return catalog


def _build_sku_catalog(id_map: pd.DataFrame, unified: pd.DataFrame) -> list[Any]:
    from .agent import CatalogEntity

    skus: dict[str, CatalogEntity] = {}
    for _, row in id_map.iterrows():
        sku = str(row.get("unified_sku") or "").strip()
        if not sku or sku.upper() == "UNMAPPED":
            continue
        if str(row.get("map_status")) == "UNMAPPED" and not str(
            row.get("unified_campaign_id") or ""
        ):
            # orphan shopify row — still list known product hints separately below
            pass
        if sku.startswith("TC-UNKNOWN"):
            continue
        skus[sku] = CatalogEntity(
            entity_id=sku,
            name=sku,
            entity_type="sku",
            unified_sku=sku,
            product_category=str(row.get("product_hint") or ""),
        )
    for sku in unified["unified_sku"].dropna().unique():
        sku_s = str(sku).strip()
        if sku_s and not sku_s.startswith("TC-UNKNOWN") and sku_s not in skus:
            skus[sku_s] = CatalogEntity(
                entity_id=sku_s,
                name=sku_s,
                entity_type="sku",
                unified_sku=sku_s,
            )
    # Canonical True Classic SKUs always available for orphan matching
    for sku, product in [
        ("TC-TEE-CREW", "tees"),
        ("TC-SHIRT-OXFORD", "shirts"),
        ("TC-UND-BOXER", "underwear"),
        ("TC-MIXED", "mixed"),
    ]:
        skus.setdefault(
            sku,
            CatalogEntity(
                entity_id=sku,
                name=sku,
                entity_type="sku",
                unified_sku=sku,
                product_category=product,
            ),
        )
    # Human-readable aliases help the agent fuzzy-match messy Shopify SKUs
    alias_rows = [
        ("true classic crew tee", "TC-TEE-CREW", "tees"),
        ("TC TEE CREW", "TC-TEE-CREW", "tees"),
        ("oxford shirt clearance", "TC-SHIRT-OXFORD", "shirts"),
        ("boxers underwear", "TC-UND-BOXER", "underwear"),
        ("mixed clearance unknown", "TC-MIXED", "mixed"),
    ]
    catalog = list(skus.values())
    for alias_name, canonical, product in alias_rows:
        catalog.append(
            CatalogEntity(
                entity_id=canonical,
                name=alias_name,
                entity_type="sku",
                unified_sku=canonical,
                product_category=product,
                extra={"alias": True},
            )
        )
    return catalog


def apply_agent_campaign_reconciliation(
    unified: pd.DataFrame,
    id_map: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]], str]:
    """
    For UNMAPPED campaigns, call the LLM agent to fuzzy-match against the known catalog.
    High-confidence matches are applied; low-confidence are flagged for human review.
    """
    from .agent import (
        AgentError,
        agent_available,
        fuzzy_match,
        get_min_confidence,
        propose_unified_campaign_id,
    )

    if not _agent_enabled():
        return unified, [], "agent disabled (AGENT_ENABLED=0)"

    ok, status = agent_available()
    if not ok:
        return unified, [], f"agent unavailable: {status}"

    df = unified.copy()
    catalog = _build_campaign_catalog(id_map)
    min_conf = get_min_confidence()
    matches: list[dict[str, Any]] = []

    # Distinct unmapped campaigns
    mask = (df["map_status"] != "mapped") | (df["unified_campaign_id"].astype(str) == "")
    unmapped = (
        df.loc[mask, ["platform", "platform_campaign_id", "campaign_name"]]
        .drop_duplicates()
        .to_dict("records")
    )

    for row in unmapped:
        name = str(row["campaign_name"])
        platform = str(row["platform"])
        try:
            result = fuzzy_match(
                name,
                catalog,
                query_type="campaign",
                query_context={
                    "platform": platform,
                    "platform_campaign_id": row["platform_campaign_id"],
                },
            )
        except AgentError as exc:
            matches.append(
                {
                    "query": name,
                    "query_type": "campaign",
                    "error": str(exc),
                    "applied": False,
                }
            )
            continue

        applied = False
        new_unified = result.matched_id
        new_sku = result.unified_sku or ""
        # Normalize pending_* catalog hits into durable unified ids
        if new_unified and str(new_unified).startswith("pending_"):
            new_unified = propose_unified_campaign_id(name, platform)
        if result.create_new and not new_unified:
            new_unified = propose_unified_campaign_id(name, platform)

        # Apply when confident on a catalog/pending match, OR create_new with usable labels
        create_new_ok = bool(result.create_new) and (
            (result.funnel_stage and result.funnel_stage != "unknown")
            or (result.product_category and result.product_category != "unknown")
        )
        conf_ok = result.confidence >= min_conf
        create_conf_ok = result.confidence >= max(0.5, min_conf - 0.1)
        should_apply = bool(new_unified) and (
            conf_ok or (create_new_ok and create_conf_ok)
        )

        if should_apply:
            sel = (df["platform"] == platform) & (
                df["platform_campaign_id"] == str(row["platform_campaign_id"])
            )
            df.loc[sel, "unified_campaign_id"] = new_unified
            if new_sku:
                df.loc[sel, "unified_sku"] = new_sku
            df.loc[sel, "map_status"] = "agent_matched"
            df.loc[sel, "agent_match_confidence"] = result.confidence
            df.loc[sel, "agent_match_rationale"] = result.rationale
            applied = True
            result.applied = True

            # Update flags: remove unmapped_id where applied
            def _strip_unmapped(flags: list[str]) -> list[str]:
                return [f for f in (flags or []) if f != "unmapped_id"] + (
                    ["agent_reconciled"] if "agent_reconciled" not in (flags or []) else []
                )

            df.loc[sel, "flags"] = df.loc[sel, "flags"].apply(
                lambda fl: _strip_unmapped(list(fl) if isinstance(fl, list) else [])
            )
        else:
            # keep unmapped; add needs_review flag
            sel = (df["platform"] == platform) & (
                df["platform_campaign_id"] == str(row["platform_campaign_id"])
            )

            def _add_review(flags: list[str]) -> list[str]:
                return _merge_flags(list(flags) if isinstance(flags, list) else [], ["agent_needs_review"])

            df.loc[sel, "flags"] = df.loc[sel, "flags"].apply(_add_review)
            df.loc[sel, "agent_match_confidence"] = result.confidence
            df.loc[sel, "agent_match_rationale"] = result.rationale

        # Optionally enrich funnel/product from agent when taxonomy unknown
        if result.funnel_stage and result.funnel_stage != "unknown":
            sel = (df["platform"] == platform) & (
                df["platform_campaign_id"] == str(row["platform_campaign_id"])
            )
            unknown_funnel = df.loc[sel, "funnel_stage"].isin(["unknown", ""])
            df.loc[sel & unknown_funnel, "funnel_stage"] = result.funnel_stage
            if result.product_category and result.product_category != "unknown":
                unknown_prod = df.loc[sel, "product_category"].isin(["unknown", ""])
                df.loc[sel & unknown_prod, "product_category"] = result.product_category

        payload = result.to_dict()
        payload["applied"] = applied
        payload["platform"] = platform
        payload["platform_campaign_id"] = row["platform_campaign_id"]
        matches.append(payload)

    # Refresh slice ids after possible funnel fills
    df["slice_id"] = df["platform"] + ":" + df["funnel_stage"]
    df["is_prospecting_like"] = df["funnel_stage"].isin(["prospecting", "non_brand"])
    return df, matches, f"agent ok ({status}); campaign queries={len(unmapped)}"


def apply_agent_sku_reconciliation(
    shopify: pd.DataFrame,
    id_map: pd.DataFrame,
    unified: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]], str]:
    """Fuzzy-match orphan Shopify SKUs (e.g. TC-UNKNOWN-CLEARANCE) to catalog SKUs."""
    from .agent import AgentError, agent_available, fuzzy_match, get_min_confidence

    if not _agent_enabled():
        return shopify, [], "agent disabled"

    ok, status = agent_available()
    if not ok:
        return shopify, [], f"agent unavailable: {status}"

    df = shopify.copy()
    if "reconciled_sku" not in df.columns:
        df["reconciled_sku"] = df["sku"]
    if "sku_map_status" not in df.columns:
        df["sku_map_status"] = "passthrough"
    if "sku_agent_confidence" not in df.columns:
        df["sku_agent_confidence"] = np.nan

    known = set(unified["unified_sku"].dropna().astype(str).unique()) | {
        "TC-TEE-CREW",
        "TC-SHIRT-OXFORD",
        "TC-UND-BOXER",
        "TC-MIXED",
    }
    orphans = sorted(
        {
            str(s)
            for s in df["sku"].dropna().unique()
            if str(s) not in known or str(s).startswith("TC-UNKNOWN")
        }
    )
    if not orphans:
        return df, [], f"agent ok ({status}); no orphan SKUs"

    catalog = _build_sku_catalog(id_map, unified)
    min_conf = get_min_confidence()
    matches: list[dict[str, Any]] = []

    for sku in orphans:
        try:
            result = fuzzy_match(
                sku,
                catalog,
                query_type="sku",
                query_context={"source": "shopify_orders"},
            )
        except AgentError as exc:
            matches.append({"query": sku, "query_type": "sku", "error": str(exc), "applied": False})
            continue

        applied = False
        target = result.unified_sku or result.matched_id
        # Clearance / unknown orphans → prefer TC-MIXED when model is unsure but names hint mixed
        if (not target or str(target).startswith("TC-UNKNOWN")) and (
            "clearance" in sku.lower() or "unknown" in sku.lower()
        ):
            if result.product_category in {"mixed", "tees", "unknown"} and result.confidence >= 0.45:
                target = "TC-MIXED" if result.product_category in {"mixed", "unknown"} else "TC-TEE-CREW"

        sku_conf_ok = result.confidence >= min_conf or (
            result.confidence >= max(0.45, min_conf - 0.15) and bool(target)
        )
        if sku_conf_ok and target and not str(target).startswith("TC-UNKNOWN"):
            df.loc[df["sku"] == sku, "reconciled_sku"] = target
            df.loc[df["sku"] == sku, "sku_map_status"] = "agent_matched"
            df.loc[df["sku"] == sku, "sku_agent_confidence"] = result.confidence
            applied = True
            result.applied = True
        else:
            df.loc[df["sku"] == sku, "sku_map_status"] = "agent_needs_review"
            df.loc[df["sku"] == sku, "sku_agent_confidence"] = result.confidence

        payload = result.to_dict()
        payload["applied"] = applied
        matches.append(payload)

    return df, matches, f"agent ok ({status}); sku queries={len(orphans)}"


def normalize_and_reconcile(
    meta: pd.DataFrame,
    google: pd.DataFrame,
    id_map: pd.DataFrame,
    overrides: dict[str, TaxonomyResult],
) -> tuple[pd.DataFrame, list[TaxonomyResult]]:
    raw = pd.concat([meta, google], ignore_index=True)

    id_map = id_map.copy()
    id_map = id_map[id_map["platform"].isin(["meta", "google"])]
    merged = raw.merge(
        id_map[
            [
                "platform",
                "platform_campaign_id",
                "unified_campaign_id",
                "unified_sku",
                "funnel_stage_hint",
                "product_hint",
                "map_status",
            ]
        ],
        on=["platform", "platform_campaign_id"],
        how="left",
    )
    merged["map_status"] = merged["map_status"].fillna("UNMAPPED")
    merged["unified_campaign_id"] = merged["unified_campaign_id"].fillna("")
    merged["unified_sku"] = merged["unified_sku"].fillna("")
    merged["funnel_stage_hint"] = merged["funnel_stage_hint"].fillna("")
    merged["product_hint"] = merged["product_hint"].fillna("")
    merged["agent_match_confidence"] = np.nan
    merged["agent_match_rationale"] = ""

    # Taxonomy per distinct campaign name (rules → agent leftovers)
    names = sorted(merged["campaign_name"].dropna().unique().tolist())
    tax_results = [parse_campaign_name(n, overrides=overrides) for n in names]
    tax_by_name = {t.campaign_name: t for t in tax_results}

    funnels: list[str] = []
    products: list[str] = []
    tax_conf: list[float] = []
    tax_src: list[str] = []
    flag_col: list[list[str]] = []

    for _, row in merged.iterrows():
        flags = list(row["flags"]) if isinstance(row["flags"], list) else []
        t = tax_by_name[row["campaign_name"]]
        funnel = t.funnel_stage
        product = t.product_category
        conf = t.confidence
        src = t.source

        if funnel == "unknown" and row["funnel_stage_hint"]:
            funnel = row["funnel_stage_hint"]
            conf = min(conf, 0.5)
            src = f"{src}+hint"
        if product == "unknown" and row["product_hint"]:
            product = row["product_hint"]
            conf = min(conf, 0.5)
            src = f"{src}+hint"

        if row["map_status"] != "mapped" or not row["unified_campaign_id"]:
            flags = _merge_flags(flags, ["unmapped_id"])
        if conf < LOW_TAXONOMY_CONF:
            flags = _merge_flags(flags, ["low_taxonomy_confidence"])

        funnels.append(funnel)
        products.append(product)
        tax_conf.append(conf)
        tax_src.append(src)
        flag_col.append(flags)

    merged["funnel_stage"] = funnels
    merged["product_category"] = products
    merged["taxonomy_confidence"] = tax_conf
    merged["taxonomy_source"] = tax_src
    merged["flags"] = flag_col
    merged["slice_id"] = merged["platform"] + ":" + merged["funnel_stage"]
    merged["is_prospecting_like"] = merged["funnel_stage"].isin(
        ["prospecting", "non_brand"]
    )
    merged["date"] = pd.to_datetime(merged["date"]).dt.date.astype(str)
    return merged, tax_results


def compute_haircut(
    unified: pd.DataFrame,
    shopify: pd.DataFrame,
) -> tuple[pd.DataFrame, list[HaircutReport]]:
    """
    Per-platform scale toward Shopify bank using spend-share weights.
    reconciled_revenue = platform_revenue * scale_p
    """
    df = unified.copy()
    shopify_total = float(shopify["revenue"].sum())
    spend_by_platform = df.groupby("platform")["spend_usd"].sum()
    total_spend = float(spend_by_platform.sum()) or 1.0

    reports: list[HaircutReport] = []
    scales: dict[str, float] = {}

    for platform, spend in spend_by_platform.items():
        plat_rev = float(df.loc[df["platform"] == platform, "platform_revenue"].sum())
        weight = float(spend) / total_spend
        shopify_alloc = shopify_total * weight
        if plat_rev <= 0:
            scale = 1.0
            clamped = False
        else:
            raw_scale = shopify_alloc / plat_rev
            scale = float(np.clip(raw_scale, HAIRCUT_LO, HAIRCUT_HI))
            clamped = scale != raw_scale
        scales[platform] = scale
        reports.append(
            HaircutReport(
                platform=platform,
                platform_revenue_sum=plat_rev,
                shopify_allocated_sum=shopify_alloc,
                scale=scale,
                scale_clamped=clamped,
                before_after_note=(
                    f"{platform}: platform ${plat_rev:,.0f} → "
                    f"reconciled ${plat_rev * scale:,.0f} "
                    f"(scale={scale:.3f}, shopify_alloc=${shopify_alloc:,.0f})"
                ),
            )
        )

    df["haircut_scale"] = df["platform"].map(scales)
    df["reconciled_revenue"] = df["platform_revenue"] * df["haircut_scale"]

    new_flags: list[list[str]] = []
    for _, row in df.iterrows():
        flags = list(row["flags"]) if isinstance(row["flags"], list) else []
        plat_report = next(r for r in reports if r.platform == row["platform"])
        if plat_report.scale_clamped:
            flags = _merge_flags(flags, ["extreme_haircut"])
        new_flags.append(flags)
    df["flags"] = new_flags
    return df, reports


def flags_summary(unified: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for flags in unified["flags"]:
        for f in flags or []:
            counts[f] = counts.get(f, 0) + 1
    return dict(sorted(counts.items()))


def run_ingestion(data_dir: str | Path | None = None) -> IngestionResult:
    data_dir = _repo_data_dir(data_dir)
    meta = load_meta(data_dir / "raw_meta_export.json")
    google = load_google(data_dir / "raw_google_export.json")
    shopify = load_shopify(data_dir / "shopify_orders.csv")
    id_map = load_id_map(data_dir / "id_map.csv")
    overrides = load_overrides(data_dir / "taxonomy_overrides.csv")

    unified, tax_results = normalize_and_reconcile(meta, google, id_map, overrides)

    # Agentic fuzzy reconciliation for leftover campaigns + orphan SKUs
    unified, camp_matches, camp_status = apply_agent_campaign_reconciliation(unified, id_map)
    shopify, sku_matches, sku_status = apply_agent_sku_reconciliation(
        shopify, id_map, unified
    )
    agent_matches = camp_matches + sku_matches
    agent_status = f"{camp_status} | {sku_status}"

    unified, haircuts = compute_haircut(unified, shopify)

    shopify_daily = (
        shopify.groupby("order_date", as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            orders=("order_id", "count"),
            new_orders=("is_new_customer", "sum"),
        )
        .rename(columns={"order_date": "date"})
    )

    return IngestionResult(
        unified=unified,
        shopify=shopify,
        shopify_daily=shopify_daily,
        haircut_reports=haircuts,
        flags_summary=flags_summary(unified),
        taxonomy_results=tax_results,
        agent_matches=agent_matches,
        agent_status=agent_status,
    )


if __name__ == "__main__":
    result = run_ingestion()
    print("Ingestion complete")
    print(f"  rows: {len(result.unified)}")
    print(f"  spend: ${result.total_spend:,.2f}")
    print(f"  platform revenue: ${result.unified['platform_revenue'].sum():,.2f}")
    print(f"  reconciled revenue: ${result.total_reconciled_revenue:,.2f}")
    print(f"  shopify revenue: ${result.shopify['revenue'].sum():,.2f}")
    print(f"  agent: {result.agent_status}")
    if result.agent_matches:
        print(f"  agent matches ({len(result.agent_matches)}):")
        for m in result.agent_matches:
            if m.get("error"):
                print(f"    ERR {m.get('query_type')} {m.get('query')!r}: {m['error']}")
            else:
                print(
                    f"    [{m.get('provider')}/{m.get('model')}] "
                    f"{m.get('query_type')} {m.get('query')!r} → "
                    f"{m.get('matched_id')} conf={m.get('confidence')} "
                    f"applied={m.get('applied')}"
                )
    print("  haircuts:")
    for h in result.haircut_reports:
        print(f"    {h.before_after_note}")
    print(f"  flags: {result.flags_summary}")
    print("  taxonomy samples:")
    for t in result.taxonomy_results:
        print(
            f"    [{t.source} {t.confidence:.2f}] {t.campaign_name[:40]!r} "
            f"→ {t.funnel_stage}/{t.product_category}"
        )
