# Mock Data Generator

How we acquire **ununified** Meta + Google ad payloads (plus Shopify ground truth) for the True Classic case-study prototype.

**Script:** `generate_mock_data.py` (this directory)  
**Run:** `python data/generate_mock_data.py` from the repo root (or `python generate_mock_data.py` from `data/`)

---

## Why synthesize instead of pulling live APIs?

For this case study, **don’t scrape live Meta/Google as the primary path**. Synthesize fixtures that **intentionally encode the mess** M1–M3 must demonstrate.

| Approach | Role |
|---|---|
| **Python generator (this)** | Recommended for interview demo — controllable narrative, reproducible seed |
| Hand-authored tiny fixtures | Fine for smoke tests; too thin alone for Hill curve fits |
| Public MMM / ad datasets | Useful as *ideas* only; almost never match Graph API / GAQL shapes |
| API sandboxes (Meta test app, Google Ads test account) | Optional week-2 realism check; won’t give Shopify overlap + haircut story for free |

Live sandboxes are good for “we know the real field names.” They are bad for forcing attribution double-count + bank-truth reconciliation in one weekend. Keep mocks for the interview path.

---

## What “good” mock data must include

Target shape for a 15‑minute demo:

- **~21 days** of history (stable enough for `curve_fit`)
- **~8–12 campaigns** across Meta + Google at **daily** grain
- **Shopify orders** for the same window (new vs returning)
- Intentional fractures so the Control Room story is real, not tidy ETL

| Layer | Must look “real” | Must break on purpose |
|---|---|---|
| Meta JSON | Nested `actions` / `action_values`, dollar `spend`, campaign names | Missing/odd actions on some days, messy names, inflated attributed revenue |
| Google JSON | `costMicros`, camelCase `metrics` / `segments.date` | Micros scale only, different hierarchy IDs, brand vs non-brand |
| Cross-platform | Same products (Tees/Shirts) sold on both | **No shared campaign/SKU IDs**; both claim overlapping revenue |
| Shopify CSV | Order `$`, new vs returning, date | Total revenue **&lt; sum of platform revenue** (haircut story) |
| Taxonomy | Mix of clean + garbage names | A few names rules can’t parse → LLM leftovers |
| Curves | Spend/revenue that saturates | Meta smoother; Google Search steeper then flatter |
| M3 flags | At least one inefficient + one starved slice | Cash burner + capped & hungry campaigns |

Without those fractures, the prototype looks like a join, not a Control Room.

---

## Design: truth → claims → native serialization

Never invent Meta and Google metrics independently. Pipeline:

```
1. Define TRUTH
   - Daily DTC demand (Shopify revenue, new vs returning share)
   - True spend allocation by channel × funnel
   - True contribution using Hill-like response curves

2. Emit PLATFORM CLAIMS (over-attribution)
   - Meta and Google each inflate purchase value vs assisted truth
   - Shopify bank only realizes ~72% of summed assisted truth (cross-channel overlap)
   - Combined platform revenue ≈ 1.6–2.2× Shopify over the window

3. Serialize native shapes
   - Meta: nested actions, dollar strings
   - Google: micros, GAQL-like objects

4. Emit supporting maps
   - Partial ID / SKU map with intentional misses
   - Empty-ish taxonomy overrides (human table for M1)
```

### Generation tricks that make the demo land

- **Haircut target:** Σ platform-reported revenue ≈ **1.6–2.2×** Shopify so the scale factor is obvious in the UI
- **History length:** ≥14 days (we use 21) per slice; label history as **synthetic** in the main README
- **ID chaos:** Meta `2385…`, Google `987654…`, Shopify SKUs like `TC-TEE-CREW-BLK` with a **partial** map (1–2 intentional misses)
- **Units:** Google costs **only** in micros; Meta **only** in dollar strings — never pre-normalize in raw files
- **RNG seed:** `seed=42` so interview day is reproducible
- **Don’t** paste identical metrics into both platforms, ship a pre-joined CSV as the only input, or make ROAS already perfect

---

## Campaign set (channel × funnel)

| Platform | Funnel | Campaign name | Role in demo |
|---|---|---|---|
| Meta | Prospecting | `TC_US_Prospecting_Tees_Broad_2026` | Clean taxonomy; solid mROAS |
| Meta | Retargeting | `FB_Retargeting_Tees_Q3_V2` | Clean taxonomy |
| Meta | Prospecting (messy) | `August push - classic fit drop v2` | LLM leftover name |
| Meta | Prospecting | `Meta - TOF - Underwear - Broad` | Mild rule ambiguity |
| Google | Brand Search | `Search_Brand_TrueClassic_Tees_Exact` | High intent, efficient |
| Google | Non-brand Search | `Search_Generic_Mens_Tees_Broad` | Saturates harder |
| Google | Brand Search | `G_Search_Brand_Shirts_Promo_2026` | Second product |
| Meta | Retargeting | `TC_CashBurn_Retarget_AllProducts_Aug` | **Cash burner** (flag B) |
| Google | Non-brand | `Search_Generic_Hungry_Tees_Exact` | **Capped & hungry** (flag A) |
| Meta | Prospecting | `TC US Prospecting Men's Tees Broad - 2026` | **Fuzzy alias** → should agent-match `uc_meta_prospecting_tees` |
| Meta | Retargeting | `Facebook Retarget Tees Q3 v2` | **Fuzzy alias** → `uc_meta_retargeting_tees` |
| Google | Brand | `Brand Search - True Classic Tees Exact` | **Fuzzy alias** → `uc_google_brand_tees` |
| Google | Brand | `G Ads Brand Oxford Shirts Promo 26` | **Fuzzy alias** → `uc_google_brand_shirts` |

**Orphan Shopify SKUs** (also UNMAPPED; agent should map to canonical SKUs):  
`TC-UNKNOWN-CLEARANCE` → `TC-MIXED`, `TC_TEE_CREW` / `true-classic-crew-tee` → `TC-TEE-CREW`, `TC-SHIRT-OXFORD-CLR` → `TC-SHIRT-OXFORD`, `boxers und sku draft` → `TC-UND-BOXER`.

Amazon / Microsoft are **out of scope** for generated live fixtures (same as main README).

---

## Output files

Written next to this doc (overwrite on each run):

| File | Description |
|---|---|
| `raw_meta_export.json` | Array of Meta-style daily campaign performance objects |
| `raw_google_export.json` | Array of Google Ads–style daily campaign performance objects |
| `shopify_orders.csv` | Ground-truth orders: revenue, new vs returning, SKU, date |
| `id_map.csv` | Partial platform_campaign_id / SKU → unified keys (includes intentional `UNMAPPED` leftovers for the LLM reconciliation agent) |
| `taxonomy_overrides.csv` | Starter human override table (mostly empty; one example override) |
| `generation_manifest.json` | Seed, date range, haircut ratio, campaign list — for audit / README honesty |

**Why intentional UNMAPPED rows exist:** M1 first joins deterministically, then `src/agent.py` fuzzy-matches leftovers (OpenAI `gpt-4.1-mini` or Ollama `llama3.1:8b`). The generator must leave gaps so the agent has something real to reconcile in the demo.

### Meta object shape (per campaign × day)

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

### Google object shape (per campaign × day)

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

### Shopify CSV columns

`order_id,order_date,revenue,is_new_customer,sku,utm_source`

---

## Hill-like spend response (internal truth only)

Generator uses a Hill-style function **internally** to create saturating truth revenue from spend. That truth is **not** written as a clean curve file — M2 must rediscover shape from noisy reconciled history.

$$
R(S) = \frac{\beta \cdot S^{\gamma}}{K^{\gamma} + S^{\gamma}}
$$

Approximate slice personalities:

- **Meta prospecting:** smoother scale, earlier diminishing returns (fatigue)
- **Meta retargeting:** higher efficiency, lower ceiling
- **Google brand search:** efficient until a sharper cliff (search volume)
- **Google non-brand:** lower efficiency, hits noise sooner
- **Cash burner:** high spend, weak true response (flag B)
- **Hungry:** strong mROAS but artificially budget-capped mid-day signal in metadata notes via high efficiency + lower absolute spend room

Platform **claims** then inflate purchase counts/values so Meta + Google overlap.

---

## Intentional data-quality fractures

Emitted so M1 can flag them:

1. **Missing Meta purchase action** on ~5% of Meta rows (empty / no purchase action)
2. **Unmapped IDs** — 1 Meta campaign and 1 Google campaign omitted or marked `UNMAPPED` in `id_map.csv`
3. **SKU mismatch** — one Shopify SKU with no platform product tag path
4. **Date alignment** — all campaigns share the same 21-day window (no silent gaps), but one campaign has a single zero-spend day
5. **Click semantics** — Meta CTR/CPC will not be comparable to Google; generator does not try to equalize them
6. **Messy names** — at least one campaign requires LLM leftover parsing

---

## Reproducibility & CLI

```bash
python data/generate_mock_data.py
python data/generate_mock_data.py --seed 42 --days 21 --end-date 2026-08-05
```

| Flag | Default | Meaning |
|---|---|---|
| `--seed` | `42` | RNG seed |
| `--days` | `21` | Number of daily rows per campaign |
| `--end-date` | `2026-08-05` | Last date in the window (inclusive) |
| `--out-dir` | directory of this script | Output folder |

After generation, sanity checks printed by the script:

- Google `costMicros` present; Meta `spend` is dollar-like strings
- Σ Meta `action_values` purchase + Σ Google `conversionsValue` **>** Σ Shopify `revenue`
- Haircut ratio printed in the manifest

---

## What this is not

- Not production ad data (no real PII, no live accounts)
- Not a solved multi-touch attribution dataset (prototype reconciliation only)
- Not Amazon / Microsoft fixtures
- Not pre-normalized unified tables — **M1 must do that**
