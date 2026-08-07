# Source Workflow — M1 → M2 → M3

How data moves through `src/`, what each script owns, and the contracts between them.

**Live path:** Meta + Google mocks + Shopify CSV only. Amazon/Microsoft are out of scope.

```
data/raw_meta_export.json ──┐
data/raw_google_export.json─┤
data/id_map.csv ────────────┼─► ingestion.py ──► UnifiedDaily + Flags + Haircut
data/shopify_orders.csv ────┤         │
data/taxonomy_overrides.csv─┘         │
                                      ├─ taxonomy.py (rules → agent leftovers)
                                      └─ agent.py (OpenAI gpt-4.1-mini | Ollama Llama3.1:8b)
                                            ▲
                                            │ fuzzy match UNMAPPED campaigns + orphan SKUs
                                            │
                                   forecasting.py ──► Curves + mROAS + CI
                                            │
                                            ▼
                                     optimizer.py ──► Channel×funnel plan
                                                      + campaign waste flags
                                            │
                                            ▼
                                   app.py Control Room (human approve)
```

**LLM agent role (automation showcase):** fuzzy **name reconciliation** of leftover campaigns and order SKUs in M1 — not dollar math, not forecasting, not budget solves.

---

## End-to-end data flow

### Stage 0 — Raw fixtures (inputs)

| File | Shape | Why it exists |
|---|---|---|
| `raw_meta_export.json` | Nested Meta Graph–style rows (dollar `spend`, `actions[]`) | Ununified vendor A |
| `raw_google_export.json` | GAQL-like rows (`costMicros`, camelCase metrics) | Ununified vendor B |
| `shopify_orders.csv` | Order-level bank truth (`revenue`, `is_new_customer`) | Ground truth for haircut + NC-CPA |
| `id_map.csv` | Partial platform ID → unified campaign/SKU | Deterministic map; intentional `UNMAPPED` leftovers for the agent |
| `taxonomy_overrides.csv` | Human funnel/product overrides | Beats model/rules when present |
| `.env` | API keys + provider switch | OpenAI vs Ollama |

Raw files are **never** pre-normalized. Micros stay micros; Meta purchases stay nested until `ingestion.py`.

### Stage 1a — Taxonomy assist (`taxonomy.py`)

Called **per campaign name** (not per dollar). Returns closed-enum tags + confidence.

Priority order:

1. Human override table  
2. Deterministic rule pass  
3. **LLM agent leftover pass** (`agent.classify_campaign_name`) — OpenAI or Ollama  
4. Validate → else `unknown`

Output is enrichment only. It does **not** invent spend, revenue, or IDs.

### Stage 1b — LLM reconciliation agent (`agent.py`)  ★ automation showcase

Called only for **leftovers** after deterministic `id_map` join:

1. **UNMAPPED campaigns** → `fuzzy_match(..., query_type="campaign")` against known unified catalog  
2. **Orphan Shopify SKUs** (e.g. `TC-UNKNOWN-CLEARANCE`) → `fuzzy_match(..., query_type="sku")`  

Provider (from `.env`):

| `LLM_PROVIDER` | Model | Runtime |
|---|---|---|
| `openai` (default) | `gpt-4.1-mini` | OpenAI Chat Completions + JSON mode |
| `ollama` | `llama3.1:8b` (FP16 via Ollama) | Local `POST /api/chat` with `format=json` |

Apply rule: if `confidence >= AGENT_MATCH_MIN_CONF` (default 0.7) → write `map_status=agent_matched` and fill unified keys; else flag `agent_needs_review` for the Control Room. Dollars are never taken from the model.

### Stage 2 — Ingestion / unification (`ingestion.py`) — M1

1. Load Meta + Google JSON  
2. **Normalize schemas** → flat daily campaign facts  
3. **Deterministic ID reconcile** via `id_map.csv`; surface `UNMAPPED`  
4. **Tag taxonomy** via `taxonomy.py` (rules → agent)  
5. **Agentic fuzzy reconciliation** via `agent.py` for leftover campaigns + orphan order SKUs  
6. Aggregate Shopify by day  
7. **Attribution haircut** — per-platform scale toward Shopify bank  
8. Emit **data-quality flags**

**Primary artifact:** `UnifiedDaily` + `HaircutReport` + `FlagList` + `agent_matches[]`.

CPC / clicks are diagnostic only — **not** used for allocation.

### Stage 3 — Response curves (`forecasting.py`) — M2

Fit Hill saturation per channel × funnel; mROAS + confidence. No LLM. Not a full Prophet/XGBoost stack.

### Stage 4 — Allocation (`optimizer.py`) — M3

SLSQP maximize forecasted reconciled revenue under guardrails; waste flags; stub mutate payloads. No LLM.

### Stage 5 — Human gate (`app.py`)

Streamlit Control Room at repo root. Shows current vs after metrics, shifts, Hill curves, waste flags, M1 haircut + agent match table, and Approve/Reject with stub mutate JSON.

```bash
.venv/bin/streamlit run app.py
```

---

## Shared contracts

### `TaxonomyResult`

```text
campaign_name, funnel_stage, product_category, confidence, source, rationale
source ∈ {human_override, rules, agent:openai, agent:ollama, unknown}
```

### `FuzzyMatchResult` (`agent.py`)

```text
query, query_type (campaign|sku),
matched_id, matched_name, unified_sku,
funnel_stage, product_category,
confidence, rationale, create_new,
provider, model, applied
```

### `UnifiedDaily`

```text
..., map_status ∈ {mapped, UNMAPPED, agent_matched},
agent_match_confidence, agent_match_rationale, flags[]
```

---

## File deep-dives

---

## 0. `agent.py` — LLM fuzzy reconciliation agent

### Purpose

Interview automation showcase: fuzzy-match messy campaign names and orphan order SKUs when deterministic keys fail.

### Callables

| Function | Provider wiring |
|---|---|
| `call_openai(messages)` | `gpt-4.1-mini` via OpenAI SDK; requires `OPENAI_API_KEY` |
| `call_ollama(messages)` | `llama3.1:8b` FP16 via Ollama `/api/chat` |
| `llm_chat(messages)` | Routes by `LLM_PROVIDER` |
| `fuzzy_match(query, catalog, query_type=...)` | Core reconciliation → strict JSON |
| `classify_campaign_name(name)` | Taxonomy leftover parse |
| `agent_available()` | Preflight for demo / UI |

### Prompt contract (reconciliation)

```json
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
```

### Guardrails

- Never invent spend / revenue / micros  
- Closed enums validated in Python after the call  
- `create_new=true` → deterministic slug via `propose_unified_campaign_id`  
- Low confidence → `agent_needs_review`, no silent apply  
- `AGENT_ENABLED=0` skips network calls  

### Config (`.env`)

```bash
LLM_PROVIDER=openai          # or ollama
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
AGENT_MATCH_MIN_CONF=0.7
AGENT_ENABLED=1
```

### Ollama setup

```bash
ollama pull llama3.1:8b
ollama serve
# then set LLM_PROVIDER=ollama
```

---

## 1. `taxonomy.py` — campaign name parsing

Waterfall: override → rules → **LLM agent** → unknown.  
No hardcoded stub table. Does not reconcile cross-platform IDs (that’s `fuzzy_match` in ingestion).

---

## 2. `ingestion.py` — M1

1. Normalize Meta/Google  
2. Deterministic `id_map` join (surface `UNMAPPED`)  
3. Taxonomy (rules → agent)  
4. **`apply_agent_campaign_reconciliation`** ★  
5. **`apply_agent_sku_reconciliation`** ★  
6. Shopify haircut (deterministic)  
7. Flags including `agent_reconciled` / `agent_needs_review`  

---

## 3. `forecasting.py` — M2 Hill + mROAS (no LLM)

---

## 4. `optimizer.py` — M3 SLSQP + waste flags (no LLM)

---

## Suggested run order

```python
from pathlib import Path
from src.ingestion import run_ingestion
from src.forecasting import fit_all_slices
from src.optimizer import run_optimizer

ingestion = run_ingestion(Path("data"))
print(ingestion.agent_status, len(ingestion.agent_matches))
curves = fit_all_slices(ingestion.unified)
plan = run_optimizer(ingestion, curves)
```

Offline without keys:

```bash
AGENT_ENABLED=0 python -m src.ingestion
```

---

## Dependencies

- `pandas`, `numpy`, `scipy` — M1–M3 math  
- `openai`, `python-dotenv`, `requests` — agent providers  
- Streamlit/plotly — `app.py` (later)
