"""
taxonomy.py — Campaign name parsing (rules → LLM agent leftovers).

Does NOT unify schemas, reconcile IDs, or touch dollars.
Fuzzy campaign/SKU reconciliation lives in agent.py + ingestion.py.
See workflow.md.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

FUNNEL_ENUM = frozenset({"prospecting", "retargeting", "brand", "non_brand", "unknown"})
PRODUCT_ENUM = frozenset({"tees", "shirts", "underwear", "mixed", "unknown"})


@dataclass
class TaxonomyResult:
    campaign_name: str
    funnel_stage: str
    product_category: str
    confidence: float
    source: str  # human_override | rules | agent:openai | agent:ollama | unknown
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_name(name: str) -> str:
    s = name.lower().strip()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def load_overrides(path: str | Path | None) -> dict[str, TaxonomyResult]:
    """Load human overrides keyed by exact campaign_name."""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, TaxonomyResult] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("campaign_name") or "").strip()
            if not name:
                continue
            funnel = _coerce_funnel(row.get("funnel_stage", "unknown"))
            product = _coerce_product(row.get("product_category", "unknown"))
            try:
                conf = float(row.get("confidence") or 1.0)
            except ValueError:
                conf = 1.0
            out[name] = TaxonomyResult(
                campaign_name=name,
                funnel_stage=funnel,
                product_category=product,
                confidence=conf,
                source="human_override",
                rationale=(row.get("notes") or "human override").strip(),
            )
    return out


def _coerce_funnel(value: str | None) -> str:
    v = (value or "unknown").strip().lower().replace(" ", "_")
    aliases = {
        "tof": "prospecting",
        "top_of_funnel": "prospecting",
        "bof": "retargeting",
        "bottom_of_funnel": "retargeting",
        "nonbrand": "non_brand",
        "non-brand": "non_brand",
        "generic": "non_brand",
    }
    v = aliases.get(v, v)
    return v if v in FUNNEL_ENUM else "unknown"


def _coerce_product(value: str | None) -> str:
    v = (value or "unknown").strip().lower()
    aliases = {
        "tee": "tees",
        "tshirt": "tees",
        "t_shirt": "tees",
        "shirt": "shirts",
        "oxford": "shirts",
        "boxer": "underwear",
        "boxers": "underwear",
        "allproducts": "mixed",
        "all_products": "mixed",
    }
    v = aliases.get(v, v)
    return v if v in PRODUCT_ENUM else "unknown"


def parse_with_rules(campaign_name: str) -> TaxonomyResult | None:
    """
    Deterministic token heuristics.
    Returns None if funnel or product cannot be resolved without conflict.
    """
    n = _normalize_name(campaign_name)
    funnel_hits: list[str] = []
    product_hits: list[str] = []

    if re.search(r"\b(retarget\w*|remarket\w*|bof|dpa)\b", n):
        funnel_hits.append("retargeting")
    if re.search(r"\b(prospect\w*|tof|cold|awareness)\b", n):
        funnel_hits.append("prospecting")
    if re.search(r"\bbrand\b", n):
        funnel_hits.append("brand")
    if re.search(r"\b(generic|non brand|nonbrand|competitor)\b", n):
        funnel_hits.append("non_brand")
    if not funnel_hits and re.search(r"\bsearch\b", n) and re.search(r"\bexact\b", n):
        funnel_hits.append("brand")
    if not funnel_hits and re.search(r"\bbroad\b", n) and re.search(r"\b(meta|fb|tc)\b", n):
        funnel_hits.append("prospecting")

    if re.search(r"\btees?\b", n):
        product_hits.append("tees")
    if re.search(r"\bshirts?\b|\boxford\b", n):
        product_hits.append("shirts")
    if re.search(r"\b(underwear|boxers?)\b", n):
        product_hits.append("underwear")
    if re.search(r"\b(allproducts|all products|mixed)\b", n):
        product_hits.append("mixed")

    def uniq(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    funnel_hits = uniq(funnel_hits)
    product_hits = uniq(product_hits)

    if len(funnel_hits) != 1 or len(product_hits) != 1:
        return None

    return TaxonomyResult(
        campaign_name=campaign_name,
        funnel_stage=funnel_hits[0],
        product_category=product_hits[0],
        confidence=0.9,
        source="rules",
        rationale=f"rule tokens → funnel={funnel_hits[0]}, product={product_hits[0]}",
    )


def _validate(result: TaxonomyResult) -> TaxonomyResult:
    funnel = result.funnel_stage if result.funnel_stage in FUNNEL_ENUM else "unknown"
    product = result.product_category if result.product_category in PRODUCT_ENUM else "unknown"
    conf = float(result.confidence)
    if funnel == "unknown" or product == "unknown":
        conf = min(conf, 0.4)
    return TaxonomyResult(
        campaign_name=result.campaign_name,
        funnel_stage=funnel,
        product_category=product,
        confidence=conf,
        source=result.source,
        rationale=result.rationale,
    )


def _agent_enabled() -> bool:
    return os.getenv("AGENT_ENABLED", "1").strip() not in {"0", "false", "False", "no"}


def parse_with_agent(campaign_name: str) -> TaxonomyResult | None:
    """Live LLM agent taxonomy parse (OpenAI gpt-4.1-mini or Ollama Llama 3.1 8B)."""
    if not _agent_enabled():
        return None
    try:
        from .agent import AgentError, classify_campaign_name
    except ImportError:
        from agent import AgentError, classify_campaign_name  # type: ignore

    try:
        data = classify_campaign_name(campaign_name)
    except AgentError:
        return None
    except Exception:
        return None

    return TaxonomyResult(
        campaign_name=campaign_name,
        funnel_stage=str(data.get("funnel_stage", "unknown")),
        product_category=str(data.get("product_category", "unknown")),
        confidence=float(data.get("confidence", 0.0)),
        source=str(data.get("source", "agent")),
        rationale=str(data.get("rationale", "")),
    )


def parse_campaign_name(
    campaign_name: str,
    overrides: dict[str, TaxonomyResult] | None = None,
) -> TaxonomyResult:
    """
    Waterfall: human override → rules → LLM agent → unknown.
    """
    overrides = overrides or {}
    if campaign_name in overrides:
        return _validate(overrides[campaign_name])

    ruled = parse_with_rules(campaign_name)
    if ruled is not None:
        return _validate(ruled)

    agented = parse_with_agent(campaign_name)
    if agented is not None:
        return _validate(agented)

    return _validate(
        TaxonomyResult(
            campaign_name=campaign_name,
            funnel_stage="unknown",
            product_category="unknown",
            confidence=0.1,
            source="unknown",
            rationale="rules missed and agent unavailable/failed",
        )
    )


def tag_campaigns(
    names: Iterable[str],
    overrides: dict[str, TaxonomyResult] | None = None,
) -> list[TaxonomyResult]:
    seen: set[str] = set()
    results: list[TaxonomyResult] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        results.append(parse_campaign_name(name, overrides=overrides))
    return results


def result_to_json(result: TaxonomyResult) -> str:
    """Strict JSON shape used as the LLM output contract."""
    payload = {
        "funnel_stage": result.funnel_stage,
        "product_category": result.product_category,
        "confidence": result.confidence,
        "rationale": result.rationale,
    }
    return json.dumps(payload)
