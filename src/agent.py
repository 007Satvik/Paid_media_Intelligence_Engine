"""
agent.py — LLM agent for fuzzy name reconciliation (campaigns / SKUs / orders).

Providers:
  - OpenAI: gpt-4.1-mini (default)
  - Ollama: llama3.1:8b (FP16 local)

Money math stays deterministic in ingestion/forecasting/optimizer.
This module only proposes entity matches + taxonomy leftovers with confidence.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import requests
from dotenv import load_dotenv

# Load repo-root .env once
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

Provider = Literal["openai", "ollama"]

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MIN_CONF = 0.55

FUNNEL_ENUM = ("prospecting", "retargeting", "brand", "non_brand", "unknown")
PRODUCT_ENUM = ("tees", "shirts", "underwear", "mixed", "unknown")


@dataclass
class CatalogEntity:
    """A known unified entity the agent can match against."""

    entity_id: str
    name: str
    entity_type: str  # campaign | sku
    platform: str = ""
    unified_sku: str = ""
    funnel_stage: str = ""
    product_category: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_prompt_row(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "platform": self.platform,
            "unified_sku": self.unified_sku,
            "funnel_stage": self.funnel_stage,
            "product_category": self.product_category,
        }


@dataclass
class FuzzyMatchResult:
    query: str
    query_type: str  # campaign | sku | order_sku
    matched_id: str | None
    matched_name: str | None
    unified_sku: str | None
    funnel_stage: str | None
    product_category: str | None
    confidence: float
    rationale: str
    create_new: bool
    provider: str
    model: str
    applied: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


class AgentError(RuntimeError):
    pass


def get_provider() -> Provider:
    raw = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    if raw not in {"openai", "ollama"}:
        raise AgentError(f"Unsupported LLM_PROVIDER={raw!r}; use 'openai' or 'ollama'")
    return raw  # type: ignore[return-value]


def get_min_confidence() -> float:
    try:
        return float(os.getenv("AGENT_MATCH_MIN_CONF", str(DEFAULT_MIN_CONF)))
    except ValueError:
        return DEFAULT_MIN_CONF


def _extract_json(text: str) -> dict[str, Any]:
    """Parse model output into a dict; tolerate fenced markdown."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise AgentError(f"Model did not return JSON: {text[:300]!r}")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise AgentError("JSON payload was not an object")
    return data


# ---------------------------------------------------------------------------
# Provider callables
# ---------------------------------------------------------------------------


def call_openai(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.0,
) -> str:
    """
    Call OpenAI Chat Completions with gpt-4.1-mini (default).
    Requires OPENAI_API_KEY in .env.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key.startswith("your_"):
        raise AgentError(
            "OPENAI_API_KEY missing or placeholder. Set it in the repo-root .env file."
        )

    model = model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise AgentError("openai package not installed. pip install openai") from exc

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or ""
    if not content.strip():
        raise AgentError("OpenAI returned empty content")
    return content


def call_ollama(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.0,
) -> str:
    """
    Call local Ollama chat API with Llama 3.1 8B (FP16 weights via Ollama).
    Default model tag: llama3.1:8b
    Requires Ollama running at OLLAMA_BASE_URL.
    """
    base = (os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_URL).rstrip("/")
    model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    # Prefer /api/chat; ask for JSON-shaped answers in the system prompt.
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
        },
    }
    try:
        r = requests.post(f"{base}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise AgentError(
            f"Ollama call failed ({base}, model={model}). "
            f"Is Ollama running and has `ollama pull {model}` been done? ({exc})"
        ) from exc

    data = r.json()
    content = (data.get("message") or {}).get("content") or ""
    if not content.strip():
        raise AgentError(f"Ollama returned empty content: {data!r}")
    return content


def llm_chat(
    messages: list[dict[str, str]],
    *,
    provider: Provider | None = None,
    temperature: float = 0.0,
) -> tuple[str, str, str]:
    """
    Route to OpenAI or Ollama.
    Returns (content, provider_name, model_name).
    """
    provider = provider or get_provider()
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        return call_openai(messages, model=model, temperature=temperature), provider, model
    model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    return call_ollama(messages, model=model, temperature=temperature), provider, model


# ---------------------------------------------------------------------------
# Fuzzy matching / reconciliation prompts
# ---------------------------------------------------------------------------


_SYSTEM_RECONCILE = """You are a paid-media data reconciliation agent for True Classic apparel.
Your job is fuzzy name matching only — never invent spend, revenue, or currency amounts.

Given a QUERY entity and a CATALOG of known entities, pick the best match OR propose creating a new unified id.
Return STRICT JSON with this schema:
{
  "matched_id": "string or null",
  "matched_name": "string or null",
  "unified_sku": "string or null",
  "funnel_stage": "prospecting|retargeting|brand|non_brand|unknown",
  "product_category": "tees|shirts|underwear|mixed|unknown",
  "confidence": 0.0,
  "create_new": false,
  "rationale": "short reason"
}

Rules:
- Prefer an existing catalog entity_id when names clearly refer to the same campaign/SKU
  (including "pending_*" catalog rows that encode funnel/product hints).
- If the query is a real leftover campaign/SKU with no close twin, set create_new=true,
  matched_id=null, fill funnel_stage + product_category from the name, and use confidence >= 0.75.
- For orphan SKUs like TC-UNKNOWN-CLEARANCE / mixed clearance, prefer unified_sku TC-MIXED when reasonable.
- confidence in [0,1]. Use <0.5 only when truly unsure.
- Do not copy dollars. Closed enums only for funnel_stage and product_category.
"""


_SYSTEM_TAXONOMY = """You parse messy advertising campaign names into closed-vocab metadata for True Classic.
Return STRICT JSON:
{
  "funnel_stage": "prospecting|retargeting|brand|non_brand|unknown",
  "product_category": "tees|shirts|underwear|mixed|unknown",
  "confidence": 0.0,
  "rationale": "short reason"
}
If unsure, use unknown and low confidence. Never invent numbers.
"""


def fuzzy_match(
    query: str,
    candidates: list[CatalogEntity],
    *,
    query_type: str = "campaign",
    query_context: dict[str, Any] | None = None,
    provider: Provider | None = None,
) -> FuzzyMatchResult:
    """
    Core agent call: fuzzy-match a campaign name, SKU, or order SKU against a catalog.
    """
    catalog_rows = [c.to_prompt_row() for c in candidates]
    user_payload = {
        "query_type": query_type,
        "query": query,
        "query_context": query_context or {},
        "catalog": catalog_rows,
        "allowed_funnel_stage": list(FUNNEL_ENUM),
        "allowed_product_category": list(PRODUCT_ENUM),
    }
    messages = [
        {"role": "system", "content": _SYSTEM_RECONCILE},
        {"role": "user", "content": json.dumps(user_payload)},
    ]

    content, prov, model = llm_chat(messages, provider=provider)
    raw = _extract_json(content)

    matched_id = raw.get("matched_id")
    if matched_id is not None:
        matched_id = str(matched_id).strip() or None
    matched_name = raw.get("matched_name")
    if matched_name is not None:
        matched_name = str(matched_name).strip() or None

    funnel = str(raw.get("funnel_stage") or "unknown").lower()
    product = str(raw.get("product_category") or "unknown").lower()
    if funnel not in FUNNEL_ENUM:
        funnel = "unknown"
    if product not in PRODUCT_ENUM:
        product = "unknown"

    try:
        conf = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    unified_sku = raw.get("unified_sku")
    if unified_sku is not None:
        unified_sku = str(unified_sku).strip() or None

    return FuzzyMatchResult(
        query=query,
        query_type=query_type,
        matched_id=matched_id,
        matched_name=matched_name,
        unified_sku=unified_sku,
        funnel_stage=funnel,
        product_category=product,
        confidence=conf,
        rationale=str(raw.get("rationale") or ""),
        create_new=bool(raw.get("create_new", False)),
        provider=prov,
        model=model,
        raw=raw,
    )


def classify_campaign_name(
    campaign_name: str,
    *,
    provider: Provider | None = None,
) -> dict[str, Any]:
    """Leftover taxonomy parse via the same LLM agent (closed enums)."""
    messages = [
        {"role": "system", "content": _SYSTEM_TAXONOMY},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "campaign_name": campaign_name,
                    "allowed_funnel_stage": list(FUNNEL_ENUM),
                    "allowed_product_category": list(PRODUCT_ENUM),
                }
            ),
        },
    ]
    content, prov, model = llm_chat(messages, provider=provider)
    raw = _extract_json(content)
    funnel = str(raw.get("funnel_stage") or "unknown").lower()
    product = str(raw.get("product_category") or "unknown").lower()
    if funnel not in FUNNEL_ENUM:
        funnel = "unknown"
    if product not in PRODUCT_ENUM:
        product = "unknown"
    try:
        conf = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return {
        "funnel_stage": funnel,
        "product_category": product,
        "confidence": max(0.0, min(1.0, conf)),
        "rationale": str(raw.get("rationale") or ""),
        "provider": prov,
        "model": model,
        "source": f"agent:{prov}",
    }


def propose_unified_campaign_id(campaign_name: str, platform: str) -> str:
    """Deterministic slug if agent asks to create_new without an id."""
    slug = re.sub(r"[^a-z0-9]+", "_", campaign_name.lower()).strip("_")
    slug = "_".join(slug.split("_")[:6]) or "unknown"
    return f"uc_agent_{platform}_{slug}"


def agent_available() -> tuple[bool, str]:
    """Quick check for demo / UI: can we call the configured provider?"""
    try:
        provider = get_provider()
    except AgentError as exc:
        return False, str(exc)
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key or key.startswith("your_"):
            return False, "OPENAI_API_KEY not set in .env"
        return True, f"openai/{os.getenv('OPENAI_MODEL', DEFAULT_OPENAI_MODEL)}"
    # ollama: ping tags endpoint
    base = (os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_URL).rstrip("/")
    try:
        r = requests.get(f"{base}/api/tags", timeout=3)
        r.raise_for_status()
    except requests.RequestException as exc:
        return False, f"Ollama unreachable at {base}: {exc}"
    return True, f"ollama/{os.getenv('OLLAMA_MODEL', DEFAULT_OLLAMA_MODEL)}"
