# True Classic — Paid Media Intelligence Engine

Prototype for a paid-media **Control Room**: ingest fragmented ad data, fit spend→revenue response curves, and recommend budget reallocations with a human approval gate.

**Live demo scope:** Meta + Google only (mocked APIs). Shopify/GA4 as ground-truth revenue. Amazon / Microsoft are **out of the live path** for this prototype; they can reuse the same unified schema later.

**UI choice:** Streamlit is an **operator Control Room** for the interview walkthrough — not a production product UI. Production path would likely be FastAPI services + Looker / internal app.

```bash
# from repo root (after .venv + requirements)
.venv/bin/streamlit run app.py
```

---

## The Core Problem

| Friction | What breaks |
|---|---|
| Different languages | Platforms use mismatched taxonomies, IDs, currencies, and metric definitions |
| Attribution overlap | Meta and Google both claim the same purchase → double-counted revenue |
| Diminishing returns | More spend in one day lowers efficiency; average ROAS hides marginal waste |

**Job of this system:** pull receipts → reconcile truth → estimate response curves → propose a budget plan → let a human approve/reject.

---

## Success Metrics (optimize for these)

1. **Blended ROAS ≥ 4.0x** — every ad dollar accountable; surface diminishing returns and reallocate toward blended lift
2. **NC-CPA ≤ Target** — keep new-customer CPA under ceiling without starving prospecting
3. **100% efficient utilization** — flag cash-burners and capped-but-hungry campaigns

**Optimization objective (important):** maximize **forecasted revenue** under ROAS / NC-CPA / shift-cap guardrails — **not** maximize ROAS alone. Maximizing ROAS often shrinks spend into only the safest dollars and under-utilizes budget.

**Cross-platform KPIs we do *not* optimize on:** CPC / raw click rates. Meta “click” ≠ Google Search “click”. Clicks stay diagnostic only; allocation uses spend, reconciled revenue, and new customers.

---

## Architecture (3 Modules)

```
Raw Meta/Google JSON ──► M1 Ingestion ──► Unified daily facts + data-quality flags
         │                     │
         │                     ├─ taxonomy.py (rules → agent leftovers)
         │                     └─ agent.py ★ fuzzy reconcile UNMAPPED campaigns + orphan SKUs
         │                              (OpenAI gpt-4.1-mini | Ollama Llama 3.1 8B)
         ▼
M2 Response curves ──► Hill saturation + mROAS + CIs
         │
         ▼
M3 Optimizer ──► Channel×funnel plan + campaign waste flags
         │
         ▼
Control Room UI ──► Human Approve / Reject / Tweak (incl. agent match review)
         │
         ▼
Stubbed Meta/Google budget mutate APIs
```

| Module | Role | AI / math |
|---|---|---|
| **M1** | Deterministic unify + **LLM agent fuzzy reconciliation** | Rules for schema/IDs/haircut; **agent.py** for leftover campaign/SKU name matching (`gpt-4.1-mini` or Ollama `llama3.1:8b`) |
| **M2** | Spend→revenue **saturation / response** model | SciPy Hill `curve_fit` + mROAS (no LLM) |
| **M3** | Allocate at **channel × funnel**; flag campaigns | SciPy SLSQP (no LLM); human final decision |

**Uncertainty as a feature:** M1 data-quality flags and M2 confidence intervals flow into M3 — low-confidence slices get tighter shift caps. UI copy frames outputs as recommendations under stated assumptions, not “the optimal media plan.”

---

## Mock API Shapes (assumed inputs)

### Meta Marketing API (nested actions)

```json
{
  "campaign_id": "23851098234",
  "campaign_name": "TC_US_Prospecting_Tees_Broad_2026",
  "spend": "1250.50",
  "impressions": "85400",
  "clicks": "1420",
  "actions": [
    { "action_type": "landing_page_view", "value": "1100" },
    { "action_type": "add_to_cart", "value": "180" },
    { "action_type": "offsite_conversion.fb_pixel_purchase", "value": "45" }
  ],
  "action_values": [
    { "action_type": "offsite_conversion.fb_pixel_purchase", "value": "3150.00" }
  ],
  "date_start": "2026-08-01",
  "date_stop": "2026-08-01"
}
```

### Google Ads API (GAQL / camelCase / micros)

```json
{
  "customer": "customers/1234567890",
  "campaign": {
    "resourceName": "customers/1234567890/campaigns/987654321",
    "id": "987654321",
    "name": "Search_Brand_TrueClassic_Tees_Exact"
  },
  "metrics": {
    "costMicros": "420500000",
    "impressions": "12300",
    "clicks": "2100",
    "conversions": "62.0",
    "conversionsValue": "4340.00"
  },
  "segments": {
    "date": "2026-08-01"
  }
}
```

---

## M1 — Data Ingestion & Cross-Platform Unification

**Demo bar:** Meta + Google mock exports → unified rows; SKU/ID reconciliation when keys mismatch; flags for missing data, date gaps, attribution conflicts.

### Design principle: deterministic core, LLM agent on leftovers

Unification is **not** “ask an LLM to merge everything.” The money path must be reproducible. The **automation showcase** is an LLM agent that fuzzy-matches names when keys don’t.

| Step | Owner | Why |
|---|---|---|
| Schema normalize, known ID maps, date align, revenue haircut | Deterministic code | Auditable; no hallucination risk on dollars |
| Campaign-name → `funnel_stage` / `product_category` | Rules first, **agent for leftovers** | Language is messy |
| **UNMAPPED campaigns + orphan order SKUs → unified keys** | **`agent.py` fuzzy match** ★ | Real reconciliation automation when `id_map` misses |
| Confidence gate + human review flags | System + marketer | Hallucination control |

### Issues M1 must absorb

**A. Schema & naming mismatches**
- Currency: Meta spend is dollars (`1250.50`); Google is micros (`420500000` → `$420.50`). Forget `/ 1e6` and every number is wrong.
- Conversions: Meta nests purchases in `actions` / `action_values`; Google puts them on `metrics`.
- Hierarchy: Meta = Campaign → Ad Set → Ad; Google = Campaign → Ad Group → Keyword/Asset. Map to a shared grain (`platform_campaign_id`, `channel`, `funnel_stage`), not 1:1 table joins.

**B. Taxonomy chaos (where the LLM helps)**
- Marketers name campaigns inconsistently (`Meta - Top of Funnel - Men's Crew Tees - Aug 2026` vs `Google_Search_Brand_Tees_Exact`).
- No shared product tag → cannot roll up “Tees” spend across platforms without parsing names (or a maintained map).

**C. Attribution overlap (biggest math failure)**
- Meta (7d click / 1d view) and Google (data-driven / last-click) can both claim 100% of the same order.
- Platforms report $200; Shopify banked $100. Blind trust → over-forecast and overspend.
- Prototype stance: this is a **visible reconciliation haircut**, not a claim that we solved multi-touch attribution (not Northbeam).

**D. Metric definitions don’t align**
- Meta “click” ≠ Google Search “click”. We **do not** force-normalize CPC into one truth or use it as a cross-platform optimization signal.

### M1 pipeline

1. **Ingest** — Meta + Google mock pulls
2. **Normalize schemas** — micros → dollars; Meta actions → flat columns
3. **Deterministic ID reconcile** — `id_map.csv`; **surface** `UNMAPPED` (never silent drop)
4. **Taxonomy tagging** — rules first; agent when rules miss
5. **Agentic fuzzy reconciliation** ★ — leftover campaigns + orphan Shopify SKUs via `src/agent.py`
6. **Truth adjustment** — per-platform haircut toward Shopify (deterministic; UI shows before/after)
7. **Flags** — `unmapped_id`, `agent_matched` / `agent_needs_review`, missing purchases, extreme haircut, …

### LLM agent use case (fuzzy reconciliation) — how it works ★

**Goal:** when platform campaign IDs / order SKUs don’t join cleanly, an LLM agent proposes the best unified entity match from a known catalog — with confidence — so ingestion can automate reconciliation instead of a brittle hardcoded switch.

**Providers (`.env`):**

| Provider | Model | When |
|---|---|---|
| OpenAI | `gpt-4.1-mini` | Default cloud demo (`LLM_PROVIDER=openai`) |
| Ollama | `llama3.1:8b` (FP16) | Local / air-gapped (`LLM_PROVIDER=ollama`) |

**What the agent receives**

- Query: unmapped campaign name *or* orphan Shopify SKU  
- Catalog: known unified campaigns / SKUs from mapped `id_map` rows  
- Closed enums for optional funnel/product enrichment  

**Strict JSON response**

```json
{
  "matched_id": "uc_meta_prospecting_underwear",
  "matched_name": "Meta - TOF - Underwear - Broad",
  "unified_sku": "TC-UND-BOXER",
  "funnel_stage": "prospecting",
  "product_category": "underwear",
  "confidence": 0.82,
  "create_new": false,
  "rationale": "TOF underwear broad aligns with underwear prospecting entity"
}
```

**Runtime flow**

1. Deterministic `id_map` join first  
2. Collect leftovers (`UNMAPPED` campaigns, `TC-UNKNOWN-*` SKUs)  
3. Call `agent.fuzzy_match` (OpenAI or Ollama)  
4. If `confidence >= AGENT_MATCH_MIN_CONF` (default 0.7) → apply (`map_status=agent_matched`)  
5. Else → keep unmapped + flag `agent_needs_review` for human gate  
6. Dollars / haircut **never** come from the model  

**Also:** leftover taxonomy parse (`August push - classic fit drop v2`) uses the same agent via `classify_campaign_name` after rules miss — no hardcoded stub table.

**Hallucination controls**
- Closed enums + JSON parse/validate in Python  
- Confidence gate; low conf → human review, not silent apply  
- `create_new` uses a deterministic slug helper, not free-form money fields  
- `AGENT_ENABLED=0` for offline runs without keys  

See `src/agent.py`, `src/workflow.md`, and repo-root `.env` / `.env.example`.

---

## M2 — Spend-Response / Saturation Engine (not a full forecaster)

**Demo bar:** Named model; per-channel (× funnel) spend-response curves; confidence intervals / fit quality.

**Question answered:** “If I change spend on this slice, how does reconciled revenue tend to respond?” — i.e. **shape of the efficiency curve**, not a full “tomorrow’s ROAS with seasonality” forecast.

### Clarification / tradeoff

| What M2 is | What M2 is not (v1) |
|---|---|
| Hill saturation fit on recent spend × reconciled revenue | Prophet / XGBoost demand forecasting stack |
| Transparent mROAS at current spend | Black-box “ML forecast” |
| Counterfactual: revenue at alternate budgets via the curve | Rich seasonality, promo calendar, creative fatigue model |

**Why this tradeoff:** for a 15-minute live demo, a named, inspectable curve is more defensible than a thin time-series stack. Prophet / XGBoost stay in **week-2** if we need seasonality decomposition or richer priors — they are not required to hit the brief.

### Why not linear ROAS

Average ROAS assumes `$1k → $4k` scales to `$10k → $40k`. Real channels saturate: high-intent inventory exhausts, then you pay for lower-intent reach.

| Metric | Definition | Use |
|---|---|---|
| **aROAS** | Revenue / Spend | What dashboards show |
| **mROAS** | dRevenue / dSpend (from the fitted curve) | What allocation should follow |

**Rule:** keep funding a slice until **marginal** ROAS falls below the efficiency target — not average ROAS.

### Model choice

Fit a **Hill saturation** spend→revenue curve per **channel × funnel** slice (e.g. Meta Prospecting, Meta Retargeting, Google Search) with `scipy.optimize.curve_fit`. Generate enough synthetic history for a stable fit if fixtures are thin — and be transparent that history is synthetic where needed.

$$
R(S) = \frac{\beta \cdot S^{\gamma}}{K^{\gamma} + S^{\gamma}}
$$

| Symbol | Meaning |
|---|---|
| $S$ | Daily spend |
| $R(S)$ | Expected **reconciled** revenue |
| $\beta$ | Saturation ceiling (max revenue) |
| $K$ | Half-saturation spend |
| $\gamma$ | Shape / how fast diminishing returns hit |

Expose **mROAS at current spend** (derivative of $R$ at $S_{\text{current}}$) as the bridge into M3.

### How curves differ in practice

```
Revenue ($)
   ^
   |                    Google Search (high intent, hard cap)
   |                     /-------------------------  max search volume
   |                    /
   |     Meta Ads      /
   |     (smooth)    /
   |       . - - ' '
   |   . '
   +---------------------------------------------> Spend ($)
```

- **Meta:** large audience, smoother scale, earlier fatigue / diminishing returns
- **Google Search:** efficient until a sharp cliff when search volume runs out

### Trustworthy outputs

Prefer intervals / fit diagnostics over point estimates:

- Weak: “Meta will generate $9,100”
- Strong: “Meta Prospecting → $9,100 \[band $8,400–$9,800\], fit R² / residual notes”

Low CI width / poor fit / M1 quality flags → M3 applies **tighter shift caps** for that slice.

---

## M3 — Budget Allocation & Execution Automation

**Demo bar:** Runnable reallocation across ≥2 platforms with visible inputs; diminishing-returns detection; marketer approval before any execution (API calls stubbed).

### Optimization grain (scope change)

| Layer | What happens |
|---|---|
| **Optimizer decision variables** | **Channel × funnel** budgets only — e.g. Meta Prospecting, Meta Retargeting, Google Search (optional Shopping later) |
| **Campaign-level** | **Waste / underspend flags only** — not SciPy decision vars |

Dozens of campaign-level variables are noisy and hard to explain live. Channel×funnel keeps the Control Room readable and still hits the brief.

Amazon is **not** in the live optimizer or UI until it has the same unified schema + fixtures.

### Optimizer

Constrained non-linear program via `scipy.optimize` (e.g. SLSQP).

**Objective — maximize forecasted reconciled revenue** (not maximize ROAS):

$$
\max_{S_1,\dots,S_n} \sum_i R_i(S_i)
$$

**Constraints (guardrails)**

| Constraint | Intent | Hard vs soft (demo) |
|---|---|---|
| $\sum S_i = B_{\text{total}}$ | Full budget use, no overspend | Hard |
| $\sum R_i(S_i) / \sum S_i \ge 4.0$ | Blended ROAS floor | Prefer **soft** (penalty) if hard solve is often infeasible on synthetic data; surface “relaxed X” in UI |
| Prospecting spend / forecasted new customers ≤ NC-CPA target | Protect growth economics | Soft or hard with clear infeasibility message |
| Base shift cap: $0.8 \cdot S_{i,\text{current}} \le S_{i,\text{proposed}} \le 1.3 \cdot S_{i,\text{current}}$ | Protect learning phase | Hard |
| Tighter caps when M1/M2 confidence is low | Don’t swing on bad data | Hard (derived) |

Prospecting vs retargeting splits stay explicit so ROAS can’t be juiced by starving new-customer acquisition.

If the solver is infeasible: **do not** invent a plan — show “no feasible plan under current constraints; relax ROAS / NC-CPA / shift cap.”

### Waste & underspend flags (campaign-level)

| Flag | Condition | Action |
|---|---|---|
| **A — Capped & hungry** | Hits daily cap early **and** mROAS ≫ target (e.g. > 4.5x) | Recommend increase funded by inefficient slices |
| **B — Cash burner** | Fully spends **and** mROAS / ROAS below breakeven (e.g. mROAS < 1.5x or blended < 2.5x) | Cap or pause |

### Human-in-the-loop Control Room

No auto-execution. UI shows:

- Current vs forecast-after-reallocation (revenue, blended ROAS, NC-CPA)
- Recommended **channel × funnel** shifts + reasons (mROAS, constraints, confidence)
- Campaign-level waste flags
- Constraint tweaks, Approve / Reject
- Assumption badge: recommendation under haircut + curve + constraints

On Approve (stubbed — show exact payloads):
- Meta: `POST` ad-set `daily_budget`
- Google: `mutateCampaignBudgets` with `amount_micros`

---

## Proposed Repo Layout

```
true_classic_media_engine/
├── data/
│   ├── raw_meta_export.json      # Mock Meta API payload
│   ├── raw_google_export.json    # Mock Google Ads API payload
│   ├── shopify_orders.csv        # Ground-truth revenue; new vs returning
│   ├── id_map.csv                # Partial maps (intentional UNMAPPED leftovers)
│   └── taxonomy_overrides.csv    # Human overrides for funnel/product tags
├── src/
│   ├── agent.py                  # ★ LLM fuzzy reconcile (OpenAI 4.1-mini | Ollama Llama3.1:8b)
│   ├── ingestion.py              # M1: normalize, id_map, agent reconcile, haircut
│   ├── taxonomy.py               # Rules → agent leftover name parse
│   ├── forecasting.py            # M2: Hill fit, mROAS, confidence bands
│   ├── optimizer.py              # M3: SLSQP at channel×funnel, waste flags
│   └── workflow.md               # Data-flow + per-file deep dive
├── .env / .env.example           # API keys + LLM_PROVIDER switch
├── app.py                        # Streamlit Control Room (operator UI)
└── requirements.txt
```

---

## What’s Real vs Mocked (fill in as you build)

| Piece | Status |
|---|---|
| Meta / Google API payloads | Mock JSON fixtures |
| Shopify / GA4 ground truth | Synthetic CSV |
| Schema normalize + known ID map + haircut | Live (deterministic) |
| **LLM agent fuzzy campaign/SKU reconcile** | Live via `agent.py` once `.env` key / Ollama is set |
| Taxonomy leftovers | Live agent after rules (same providers) |
| Hill response curves + mROAS | Live |
| SLSQP at channel × funnel | Live |
| Amazon / Microsoft | Out of live path |
| Budget mutate APIs | Stubbed payloads on Approve |

**With two more weeks:** live API connectors; Amazon/Microsoft adapters; Prophet/XGBoost seasonality on top of curves; Looker export; agent match eval set + cached approved matches.
