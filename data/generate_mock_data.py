#!/usr/bin/env python3
"""
Synthetic Meta + Google + Shopify fixtures for the True Classic Control Room demo.

See mockGenerator.md in this directory for design notes.
Truth is generated first; platform claims over-attribute; native API shapes are emitted unnormalized.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def hill_revenue(spend: float, beta: float, k: float, gamma: float) -> float:
    """Hill saturation: R(S) = beta * S^gamma / (K^gamma + S^gamma)."""
    if spend <= 0:
        return 0.0
    s_g = spend**gamma
    k_g = k**gamma
    return beta * s_g / (k_g + s_g)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Campaign definitions (channel × funnel + demo roles)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignSpec:
    platform: str  # meta | google
    funnel: str  # prospecting | retargeting | brand | non_brand
    campaign_id: str
    campaign_name: str
    product: str  # tees | shirts | underwear | mixed
    unified_campaign: str
    unified_sku: str
    base_spend: float
    beta: float
    k: float
    gamma: float
    role: str  # normal | messy_name | cash_burner | capped_hungry
    map_in_id_map: bool
    daily_budget_cap: float | None = None  # for hungry narrative metadata


CAMPAIGNS: list[CampaignSpec] = [
    CampaignSpec(
        platform="meta",
        funnel="prospecting",
        campaign_id="23851098234",
        campaign_name="TC_US_Prospecting_Tees_Broad_2026",
        product="tees",
        unified_campaign="uc_meta_prospecting_tees",
        unified_sku="TC-TEE-CREW",
        base_spend=3200.0,
        beta=14000.0,
        k=4500.0,
        gamma=1.35,
        role="normal",
        map_in_id_map=True,
    ),
    CampaignSpec(
        platform="meta",
        funnel="retargeting",
        campaign_id="23851098299",
        campaign_name="FB_Retargeting_Tees_Q3_V2",
        product="tees",
        unified_campaign="uc_meta_retargeting_tees",
        unified_sku="TC-TEE-CREW",
        base_spend=1100.0,
        beta=7000.0,
        k=1600.0,
        gamma=1.55,
        role="normal",
        map_in_id_map=True,
    ),
    CampaignSpec(
        platform="meta",
        funnel="prospecting",
        campaign_id="23851999001",
        campaign_name="August push - classic fit drop v2",
        product="tees",
        unified_campaign="uc_meta_messy_tees",
        unified_sku="TC-TEE-CREW",
        base_spend=900.0,
        beta=4500.0,
        k=2200.0,
        gamma=1.25,
        role="messy_name",
        map_in_id_map=True,
    ),
    CampaignSpec(
        platform="meta",
        funnel="prospecting",
        campaign_id="23851888012",
        campaign_name="Meta - TOF - Underwear - Broad",
        product="underwear",
        unified_campaign="uc_meta_prospecting_underwear",
        unified_sku="TC-UND-BOXER",
        base_spend=700.0,
        beta=3200.0,
        k=1800.0,
        gamma=1.3,
        role="normal",
        map_in_id_map=False,  # intentional unmapped ID
    ),
    CampaignSpec(
        platform="meta",
        funnel="retargeting",
        campaign_id="23851777077",
        campaign_name="TC_CashBurn_Retarget_AllProducts_Aug",
        product="mixed",
        unified_campaign="uc_meta_cashburn_mixed",
        unified_sku="TC-MIXED",
        base_spend=2400.0,
        beta=1800.0,  # weak response → cash burner
        k=800.0,
        gamma=0.9,
        role="cash_burner",
        map_in_id_map=True,
    ),
    CampaignSpec(
        platform="google",
        funnel="brand",
        campaign_id="987654321",
        campaign_name="Search_Brand_TrueClassic_Tees_Exact",
        product="tees",
        unified_campaign="uc_google_brand_tees",
        unified_sku="TC-TEE-CREW",
        base_spend=2800.0,
        beta=16000.0,
        k=3200.0,
        gamma=2.1,  # sharper cliff
        role="normal",
        map_in_id_map=True,
    ),
    CampaignSpec(
        platform="google",
        funnel="non_brand",
        campaign_id="987654322",
        campaign_name="Search_Generic_Mens_Tees_Broad",
        product="tees",
        unified_campaign="uc_google_generic_tees",
        unified_sku="TC-TEE-CREW",
        base_spend=1900.0,
        beta=7500.0,
        k=2800.0,
        gamma=1.7,
        role="normal",
        map_in_id_map=True,
    ),
    CampaignSpec(
        platform="google",
        funnel="brand",
        campaign_id="987654400",
        campaign_name="G_Search_Brand_Shirts_Promo_2026",
        product="shirts",
        unified_campaign="uc_google_brand_shirts",
        unified_sku="TC-SHIRT-OXFORD",
        base_spend=1200.0,
        beta=6200.0,
        k=2000.0,
        gamma=1.9,
        role="normal",
        map_in_id_map=True,
    ),
    CampaignSpec(
        platform="google",
        funnel="non_brand",
        campaign_id="987659999",
        campaign_name="Search_Generic_Hungry_Tees_Exact",
        product="tees",
        unified_campaign="uc_google_hungry_tees",
        unified_sku="TC-TEE-CREW",
        base_spend=650.0,  # capped budget room
        beta=9000.0,  # strong response → hungry
        k=900.0,
        gamma=1.8,
        role="capped_hungry",
        map_in_id_map=False,  # intentional unmapped ID
        daily_budget_cap=700.0,
    ),
]

GOOGLE_CUSTOMER = "1234567890"
AOV = 72.0  # average order value for converting truth revenue → order counts


# ---------------------------------------------------------------------------
# Simple seeded RNG (no numpy dependency for fixture generation)
# ---------------------------------------------------------------------------


class RNG:
    """Minimal LCG so the script runs with stdlib only."""

    def __init__(self, seed: int) -> None:
        self.state = seed % (2**31 - 1)

    def random(self) -> float:
        self.state = (1103515245 * self.state + 12345) % (2**31)
        return self.state / (2**31)

    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.random()

    def gauss(self, mu: float, sigma: float) -> float:
        # Box-Muller
        u1 = max(1e-12, self.random())
        u2 = self.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    def randint(self, a: int, b: int) -> int:
        return a + int(self.random() * (b - a + 1))

    def choice(self, items: list[Any]) -> Any:
        return items[int(self.random() * len(items)) % len(items)]


# ---------------------------------------------------------------------------
# Day-level truth simulation
# ---------------------------------------------------------------------------


@dataclass
class DayTruth:
    day: date
    campaign: CampaignSpec
    spend: float
    true_revenue: float
    true_orders: float
    true_new_orders: float


def date_window(end: date, days: int) -> list[date]:
    start = end - timedelta(days=days - 1)
    return [start + timedelta(days=i) for i in range(days)]


def simulate_truth(rng: RNG, days: list[date]) -> list[DayTruth]:
    rows: list[DayTruth] = []
    for i, day in enumerate(days):
        # mild weekly seasonality (weekends slightly higher apparel demand)
        weekend_boost = 1.08 if day.weekday() >= 5 else 1.0
        for spec in CAMPAIGNS:
            noise = clamp(rng.gauss(1.0, 0.07), 0.82, 1.18)
            spend = spec.base_spend * weekend_boost * noise

            # one intentional zero-spend day on Meta underwear campaign
            if spec.campaign_id == "23851888012" and i == len(days) // 3:
                spend = 0.0

            # hungry campaign stays near cap
            if spec.role == "capped_hungry" and spec.daily_budget_cap is not None:
                spend = min(spend, spec.daily_budget_cap * rng.uniform(0.92, 1.0))

            true_rev = hill_revenue(spend, spec.beta, spec.k, spec.gamma)
            true_rev *= clamp(rng.gauss(1.0, 0.05), 0.88, 1.12)
            true_orders = true_rev / AOV
            # prospecting / non_brand / messy → more new customers
            if spec.funnel in {"prospecting", "non_brand"} or spec.role == "messy_name":
                new_share = rng.uniform(0.72, 0.88)
            elif spec.funnel == "brand":
                new_share = rng.uniform(0.35, 0.55)
            else:  # retargeting
                new_share = rng.uniform(0.12, 0.28)
            if spec.role == "cash_burner":
                new_share = rng.uniform(0.05, 0.15)

            rows.append(
                DayTruth(
                    day=day,
                    campaign=spec,
                    spend=round(spend, 2),
                    true_revenue=max(0.0, true_rev),
                    true_orders=max(0.0, true_orders),
                    true_new_orders=max(0.0, true_orders * new_share),
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Platform claim inflation (attribution overlap)
# ---------------------------------------------------------------------------


def claimed_revenue(true_rev: float, platform: str, rng: RNG) -> float:
    """
    Inflate platform-reported conversion value so Meta + Google overlap.
    Combined with Shopify bank discount, target ~1.6–2.2x platform/Shopify.
    """
    if true_rev <= 0:
        return 0.0
    if platform == "meta":
        # 7d click / 1d view style inflation
        mult = rng.uniform(1.45, 1.75)
    else:
        # last-click / data-driven still over-claims vs bank in aggregate
        mult = rng.uniform(1.35, 1.65)
    return true_rev * mult


def claimed_orders(true_orders: float, platform: str, rng: RNG) -> float:
    if true_orders <= 0:
        return 0.0
    if platform == "meta":
        mult = rng.uniform(1.40, 1.70)
    else:
        mult = rng.uniform(1.30, 1.60)
    return true_orders * mult


# Bank takes less than sum of channel-assisted "true" revenue (cross-channel overlap).
SHOPIFY_BANK_SHARE = 0.72


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------


def meta_row(t: DayTruth, rng: RNG, skip_purchase: bool) -> dict[str, Any]:
    spec = t.campaign
    claimed_rev = claimed_revenue(t.true_revenue, "meta", rng)
    claimed_purch = claimed_orders(t.true_orders, "meta", rng)

    # Meta clicks are "engagement-ish" — not comparable to Google
    impressions = int(max(0, t.spend * rng.uniform(55, 85)))
    clicks = int(max(0, impressions * rng.uniform(0.012, 0.028)))
    lpv = int(clicks * rng.uniform(0.65, 0.90))
    atc = int(max(0, claimed_purch * rng.uniform(2.5, 4.5)))

    actions: list[dict[str, str]] = [
        {"action_type": "landing_page_view", "value": str(lpv)},
        {"action_type": "add_to_cart", "value": str(atc)},
    ]
    action_values: list[dict[str, str]] = []

    if not skip_purchase and t.spend > 0:
        actions.append(
            {
                "action_type": "offsite_conversion.fb_pixel_purchase",
                "value": str(int(round(claimed_purch))),
            }
        )
        action_values.append(
            {
                "action_type": "offsite_conversion.fb_pixel_purchase",
                "value": f"{claimed_rev:.2f}",
            }
        )

    day_s = t.day.isoformat()
    return {
        "campaign_id": spec.campaign_id,
        "campaign_name": spec.campaign_name,
        "spend": f"{t.spend:.2f}",
        "impressions": str(impressions),
        "clicks": str(clicks),
        "actions": actions,
        "action_values": action_values,
        "date_start": day_s,
        "date_stop": day_s,
    }


def google_row(t: DayTruth, rng: RNG) -> dict[str, Any]:
    spec = t.campaign
    claimed_rev = claimed_revenue(t.true_revenue, "google", rng)
    claimed_conv = claimed_orders(t.true_orders, "google", rng)

    impressions = int(max(0, t.spend * rng.uniform(8, 18)))
    # Google Search clicks ≈ real site clicks
    clicks = int(max(0, impressions * rng.uniform(0.12, 0.28)))
    cost_micros = int(round(t.spend * 1_000_000))

    return {
        "customer": f"customers/{GOOGLE_CUSTOMER}",
        "campaign": {
            "resourceName": f"customers/{GOOGLE_CUSTOMER}/campaigns/{spec.campaign_id}",
            "id": spec.campaign_id,
            "name": spec.campaign_name,
        },
        "metrics": {
            "costMicros": str(cost_micros),
            "impressions": str(impressions),
            "clicks": str(clicks),
            "conversions": f"{claimed_conv:.1f}",
            "conversionsValue": f"{claimed_rev:.2f}",
        },
        "segments": {"date": t.day.isoformat()},
    }


def build_shopify_orders(truth: list[DayTruth], rng: RNG) -> list[dict[str, Any]]:
    """
    Materialize order-level Shopify bank truth from per-campaign assisted revenue.

    Channel-assisted truths are additive in the simulator; the bank only realizes
    SHOPIFY_BANK_SHARE of that sum (cross-channel overlap). Platforms then
    over-claim on top — producing the haircut ratio for M1.
    utm_source loosely hints platform but is not a perfect join key (on purpose).
    """
    orders: list[dict[str, Any]] = []
    order_seq = 100000

    for t in truth:
        bank_rev = t.true_revenue * SHOPIFY_BANK_SHARE
        if bank_rev <= 0:
            continue
        # fewer, chunkier orders keeps the CSV demo-friendly
        approx_orders = max(1, int(round(bank_rev / AOV)))
        n_orders = max(1, min(approx_orders, int(round(t.true_orders * SHOPIFY_BANK_SHARE))))
        remaining = bank_rev
        new_left = int(round(t.true_new_orders * SHOPIFY_BANK_SHARE))
        for j in range(n_orders):
            order_seq += 1
            if j == n_orders - 1:
                rev = remaining
            else:
                piece = bank_rev / n_orders * rng.uniform(0.75, 1.25)
                rev = min(remaining * 0.9, max(15.0, piece))
                remaining -= rev
            is_new = j < new_left
            if t.campaign.platform == "meta":
                utm = rng.choice(["facebook", "instagram", "meta", ""])
            else:
                utm = rng.choice(["google", "google_ads", "cpc", ""])

            # intentional SKU orphan on a small fraction of cash-burner orders
            sku = t.campaign.unified_sku
            if t.campaign.role == "cash_burner" and rng.random() < 0.08:
                sku = "TC-UNKNOWN-CLEARANCE"

            orders.append(
                {
                    "order_id": f"TC{order_seq}",
                    "order_date": t.day.isoformat(),
                    "revenue": f"{rev:.2f}",
                    "is_new_customer": "true" if is_new else "false",
                    "sku": sku,
                    "utm_source": utm,
                }
            )
    return orders


def build_id_map() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in CAMPAIGNS:
        status = "mapped" if spec.map_in_id_map else "UNMAPPED"
        unified_c = spec.unified_campaign if spec.map_in_id_map else ""
        unified_s = spec.unified_sku if spec.map_in_id_map else ""
        rows.append(
            {
                "platform": spec.platform,
                "platform_campaign_id": spec.campaign_id,
                "campaign_name": spec.campaign_name,
                "unified_campaign_id": unified_c,
                "unified_sku": unified_s,
                "funnel_stage_hint": spec.funnel,
                "product_hint": spec.product,
                "map_status": status,
            }
        )
    # orphan Shopify SKU with no platform path
    rows.append(
        {
            "platform": "shopify",
            "platform_campaign_id": "",
            "campaign_name": "",
            "unified_campaign_id": "",
            "unified_sku": "TC-UNKNOWN-CLEARANCE",
            "funnel_stage_hint": "",
            "product_hint": "unknown",
            "map_status": "UNMAPPED",
        }
    )
    return rows


def build_taxonomy_overrides() -> list[dict[str, str]]:
    # One example human override; messy name left for LLM/rules path
    return [
        {
            "campaign_name": "Meta - TOF - Underwear - Broad",
            "funnel_stage": "prospecting",
            "product_category": "underwear",
            "confidence": "1.0",
            "source": "human_override",
            "notes": "TOF mapped to prospecting by media buyer",
        }
    ]


# ---------------------------------------------------------------------------
# Totals / sanity
# ---------------------------------------------------------------------------


def meta_purchase_value(row: dict[str, Any]) -> float:
    total = 0.0
    for av in row.get("action_values", []):
        if av.get("action_type") == "offsite_conversion.fb_pixel_purchase":
            total += float(av["value"])
    return total


def google_conversion_value(row: dict[str, Any]) -> float:
    return float(row["metrics"]["conversionsValue"])


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ununified Meta/Google/Shopify mock fixtures")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--days", type=int, default=21)
    p.add_argument("--end-date", type=str, default="2026-08-05")
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Output directory (default: this data/ folder)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    days = date_window(end, args.days)
    rng = RNG(args.seed)

    truth = simulate_truth(rng, days)

    meta_rows: list[dict[str, Any]] = []
    google_rows: list[dict[str, Any]] = []

    meta_day_truth = [t for t in truth if t.campaign.platform == "meta"]
    # ~5% of Meta rows missing purchase action
    skip_purchase_ids: set[tuple[str, str]] = set()
    n_skip = max(1, int(0.05 * len(meta_day_truth)))
    candidates = [
        (t.campaign.campaign_id, t.day.isoformat())
        for t in meta_day_truth
        if t.spend > 0
    ]
    rng2 = RNG(args.seed + 7)
    while len(skip_purchase_ids) < n_skip and candidates:
        pick = rng2.choice(candidates)
        skip_purchase_ids.add(pick)
        candidates = [c for c in candidates if c not in skip_purchase_ids]

    for t in truth:
        if t.campaign.platform == "meta":
            skip = (t.campaign.campaign_id, t.day.isoformat()) in skip_purchase_ids
            meta_rows.append(meta_row(t, rng, skip_purchase=skip))
        else:
            google_rows.append(google_row(t, rng))

    shopify_orders = build_shopify_orders(truth, rng)
    id_map = build_id_map()
    taxonomy_overrides = build_taxonomy_overrides()

    meta_rev = sum(meta_purchase_value(r) for r in meta_rows)
    google_rev = sum(google_conversion_value(r) for r in google_rows)
    shopify_rev = sum(float(o["revenue"]) for o in shopify_orders)
    platform_rev = meta_rev + google_rev
    haircut_ratio = platform_rev / shopify_rev if shopify_rev else float("inf")

    write_json(out_dir / "raw_meta_export.json", meta_rows)
    write_json(out_dir / "raw_google_export.json", google_rows)
    write_csv(
        out_dir / "shopify_orders.csv",
        shopify_orders,
        ["order_id", "order_date", "revenue", "is_new_customer", "sku", "utm_source"],
    )
    write_csv(
        out_dir / "id_map.csv",
        id_map,
        [
            "platform",
            "platform_campaign_id",
            "campaign_name",
            "unified_campaign_id",
            "unified_sku",
            "funnel_stage_hint",
            "product_hint",
            "map_status",
        ],
    )
    write_csv(
        out_dir / "taxonomy_overrides.csv",
        taxonomy_overrides,
        [
            "campaign_name",
            "funnel_stage",
            "product_category",
            "confidence",
            "source",
            "notes",
        ],
    )

    manifest = {
        "seed": args.seed,
        "days": args.days,
        "start_date": days[0].isoformat(),
        "end_date": days[-1].isoformat(),
        "n_meta_rows": len(meta_rows),
        "n_google_rows": len(google_rows),
        "n_shopify_orders": len(shopify_orders),
        "meta_reported_purchase_value": round(meta_rev, 2),
        "google_reported_conversions_value": round(google_rev, 2),
        "shopify_revenue": round(shopify_rev, 2),
        "platform_over_shopify_ratio": round(haircut_ratio, 3),
        "meta_rows_missing_purchase_action": len(skip_purchase_ids),
        "unmapped_campaign_ids": [
            c.campaign_id for c in CAMPAIGNS if not c.map_in_id_map
        ],
        "messy_name_campaigns": [
            c.campaign_name for c in CAMPAIGNS if c.role == "messy_name"
        ],
        "cash_burner_campaigns": [
            c.campaign_name for c in CAMPAIGNS if c.role == "cash_burner"
        ],
        "capped_hungry_campaigns": [
            c.campaign_name for c in CAMPAIGNS if c.role == "capped_hungry"
        ],
        "notes": (
            "Synthetic fixtures. Platform revenue is intentionally > Shopify. "
            "Raw files are unnormalized (Meta dollars + nested actions; Google micros)."
        ),
    }
    write_json(out_dir / "generation_manifest.json", manifest)

    # Sanity console report
    print("Mock generation complete.")
    print(f"  out_dir: {out_dir}")
    print(f"  window:  {days[0]} → {days[-1]} ({args.days} days), seed={args.seed}")
    print(f"  meta rows:   {len(meta_rows)}")
    print(f"  google rows: {len(google_rows)}")
    print(f"  shopify orders: {len(shopify_orders)}")
    print(f"  Meta purchase value:   ${meta_rev:,.2f}")
    print(f"  Google conv. value:    ${google_rev:,.2f}")
    print(f"  Shopify revenue:       ${shopify_rev:,.2f}")
    print(f"  platform / shopify:    {haircut_ratio:.3f}x  (target ~1.6–2.2x)")
    if not (1.4 <= haircut_ratio <= 2.5):
        print("  WARNING: haircut ratio outside expected band; re-seed or tweak multipliers.")
    sample_meta = meta_rows[0]
    sample_google = google_rows[0]
    assert "spend" in sample_meta and "." in sample_meta["spend"]
    assert "costMicros" in sample_google["metrics"]
    assert int(sample_google["metrics"]["costMicros"]) > 1000  # still micros-scale
    print("  sanity: Meta dollar spend + Google micros OK")


if __name__ == "__main__":
    main()
