# Automotive Parts Dynamic Pricing & Inventory Optimization

A complete SKU-level dynamic pricing decision system for automotive parts retail. The system recommends customer-tier prices based on demand forecasting, price elasticity estimation, inventory position, and business guardrails.

> **The public demo uses a representative synthetic sample. The workflow is designed for production catalogs containing millions of SKUs.**

## In one sentence

**English —** Instead of guessing "what a part is worth," this system predicts *how many units will sell at each possible price*, then picks the price that best meets a business goal (profit, revenue, or clearing excess stock) — while respecting margin floors, stockout risk, and a human-approval step.

**中文 —** 这个项目不是去猜"一个零件值多少钱",而是预测*每个候选价格下能卖出多少件*,再结合毛利底线、缺货风险和库存情况,挑出最符合经营目标(利润 / 营收 / 去库存)的价格——所有建议都需人工审批后才能上线。

## Who is this for / what it demonstrates

This is a portfolio project built to show an end-to-end data-science workflow: synthetic data generation, demand forecasting, causal elasticity estimation, constrained optimization, inventory policy, backtesting, and a polished decision dashboard. No real company data is used. The UI ships **bilingual (English / 中文)** — switch from the language selector at the top of the sidebar.

**Tech stack:** Python · pandas / NumPy · scikit-learn (HistGradientBoosting) · Plotly · Streamlit · pytest.

## Executive Summary

This system does **not** predict "what a part should cost." Instead, it predicts **how many units will sell at each candidate price** and computes the resulting revenue, gross profit, margin, inventory, and stockout risk. Recommendations are constrained by nine business guardrails and require human approval before implementation.

- **3,000 SKUs** × 4 regions × 3 customer tiers × 104 weeks of synthetic data
- **Demand model**: HistGradientBoostingRegressor with stockout-adjusted targets
- **Elasticity**: Log-log regression on exogenous price-test observations
- **Optimization**: 4 objectives, 31 candidate prices per SKU, 9 guardrails
- **UI**: 8-page Streamlit dashboard with AI Analyst, bilingual EN / 中文

## Business Problem

Automotive parts retailers manage thousands of SKUs across regions and customer tiers (Retail, Trade, Fleet). Static pricing leaves margin on the table; reactive competitor matching erodes profitability. The system balances:

- Gross profit maximization
- Revenue growth
- Inventory health (excess reduction, stockout prevention)
- Customer-tier price consistency
- Competitive positioning

## Why Dynamic Pricing ≠ Price Prediction

| Price Prediction | Dynamic Pricing (This System) |
|---|---|
| "This part is worth $X" | "At price $X, expect Y units, $Z profit" |
| Single point estimate | Candidate price simulation |
| No inventory consideration | Weeks of cover, stockout risk |
| No guardrails | Margin floor, MAP, tier ladder |
| Auto-apply | Human approval queue |

## Quick Start

**Just want to see the demo?** The repo ships with precomputed recommendation artifacts in `outputs/`, so the dashboard runs straight away — no data generation or model training (and no scikit-learn) required:

```bash
cd parts-dynamic-pricing-ai
pip install -r requirements.txt   # streamlit, pandas, numpy, plotly only
streamlit run app.py              # switch EN / 中文 from the sidebar
```

**Want to rebuild everything from scratch?** Install the dev dependencies and run the full pipeline (this trains the models and regenerates the artifacts):

```bash
pip install -r requirements-dev.txt          # adds scikit-learn, joblib, pytest

# 1) Generate synthetic data
python data/generate_synthetic_data.py

# 2) Train models and generate recommendations
python -c "from src.pipeline import run_full_pipeline; run_full_pipeline()"

# 3) Launch dashboard
streamlit run app.py

# 4) Run tests
pytest tests/ -v
```

> **Deployment note:** `requirements.txt` is intentionally minimal (no scikit-learn) and pinned to exact versions, because the deployed app loads the precomputed `outputs/` artifacts via a lightweight runtime path rather than training on the fly. The full 851 MB sales file is git-ignored; the app reads a representative sample.

## Synthetic Data Design

All data is independently generated. No former-employer data, no external datasets, no data from other portfolio projects.

**Demand generation formula:**

```
latent_demand = base_demand
  × region_factor
  × (price / reference_price) ^ elasticity
  × seasonality
  × promotion_effect
  × competitor_effect
  × lifecycle_factor
  × noise

observed_units = min(latent_demand, available_inventory)
lost_sales = max(0, latent_demand - available_inventory)
```

**Price endogeneity design:** Synthetic data includes exogenous price changes (supplier cost changes, randomized regional tests, policy-driven tier discounts) to enable credible elasticity estimation.

**Stockout censoring:** Observed sales are capped at available inventory. The demand model trains on adjusted (uncensored) demand.

## Feature Engineering

23 numeric + 4 categorical features with strict no-leakage design:
- Lag features (1, 2, 4, 13 weeks) using `shift()` within SKU×region×tier groups
- Rolling aggregates computed on shifted data (prior week and earlier only)
- Time-based split: Train 78w / Validation 13w / Test 13w

## Demand Model Comparison

| Model | Type | Purpose |
|---|---|---|
| Seasonal Naive | Baseline | Same-week-last-year or lag-1 |
| Ridge Regression | Linear | Interpretable benchmark |
| HistGradientBoosting | Primary | Best predictive performance |

Metrics: MAE, RMSE, WAPE, RMSLE, Bias, Baseline improvement

## Elasticity Methodology

Log-log model: `log(units+1) = β × log(price) + controls`

- Prioritizes `price_test_flag = true` observations
- Estimates at category×tier and category×region×tier levels
- Shrinks low-confidence estimates toward global prior
- Range: -3.0 to -0.1
- **Predictive importance ≠ causal elasticity**

## Optimization

Four objective functions:
1. Maximize Gross Profit
2. Maximize Revenue
3. Reduce Excess Inventory
4. Balanced Objective (weighted composite)

Candidate prices: current ± 15% in 1% steps. Each simulated through the demand model for units, revenue, profit, and inventory impact.

## Guardrails

1. Maximum price move (±10%, ±15% for excess inventory)
2. Margin floor
3. Minimum Advertised Price (MAP)
4. Customer tier ladder (Retail ≥ Trade ≥ Fleet)
5. Stockout protection (no aggressive markdown)
6. Excess inventory clearance rules
7. Low-confidence move limits
8. Price rounding (.99, .95, nearest dollar)
9. Human approval required

## Backtesting

Compares four policies over the 13-week test period:
- Current pricing (historical)
- Cost-plus pricing
- Competitor-match pricing
- Dynamic pricing (model recommendations)

Results are **modeled/simulated estimates**, not proven business impact. A production implementation requires controlled price experiments.

## UI Overview — what each page shows

The dashboard has 8 pages (sidebar navigation). A language selector at the top of the sidebar switches the whole UI between **English** and **中文**.

| Page | In plain terms |
|---|---|
| **Executive Command Center** | The headline numbers: revenue, gross profit, modeled profit lift, inventory health, and where the biggest opportunity is. |
| **Demand Model** | How well the forecasting model predicts unit sales (accuracy metrics, actual-vs-predicted, error by category). |
| **Elasticity Explorer** | How price-sensitive each category/region/tier is — the heatmap shows where customers react most to price. |
| **SKU Decision Workbench** | Drill into one part: current vs recommended price, why, the price→units→profit curve, and the linked inventory action. |
| **Inventory Control Tower** | Stock health across the catalog — excess, stockout risk, weeks of cover, transfer and replenishment candidates. |
| **Backtest & Rollback** | Compares pricing strategies over a held-out period, plus a slider to simulate rolling back part of the changes. |
| **AI Analyst** | Ask plain-language questions; a local, deterministic analyst answers from the data (no external LLM calls). |
| **Data & Governance** | The data model, metric definitions, approval workflow, and production-monitoring checklist. |

Modern B2B analytics design with custom CSS, metric cards, status pills, and Plotly charts.

## Project Architecture

```
parts-dynamic-pricing-ai/
├── app.py                  # Streamlit dashboard
├── src/
│   ├── config.py           # All weights and parameters
│   ├── data_generator.py   # Synthetic data engine
│   ├── features.py         # Feature pipeline
│   ├── demand_model.py     # ML demand forecasting
│   ├── elasticity.py       # Price elasticity estimation
│   ├── optimizer.py        # Price optimization + guardrails
│   ├── inventory.py        # Inventory analysis
│   ├── backtest.py         # Backtest + rollback
│   ├── ai_analyst.py       # Local deterministic analyst
│   └── pipeline.py         # End-to-end orchestration
├── data/                   # Synthetic CSV files
├── models/                 # Trained model artifacts
├── outputs/                # Recommendations + backtest results
├── tests/                  # Pytest suite
└── docs/                   # Architecture and model card
```

## Testing

```bash
pytest tests/ -v
python -m py_compile app.py src/*.py
```

20 tests covering: data reproducibility, elasticity signs, inventory censoring, financial formulas, feature leakage, time splits, guardrails, rollback boundaries, AI analyst accuracy, and model persistence.

## Inventory Decision Engine (v2.0)

Inventory is integrated as a **constraint and operational-action layer** around dynamic pricing—not a separate project.

### Decision Flow

```
Demand Forecast → Inventory Position → Elasticity → Candidate Prices
→ Pricing Guardrails → Inventory Action Evaluation → Joint Recommendation
→ Human Approval → Monitoring / Rollback
```

### Inventory Metrics (centralized in `inventory_metrics.py`)

- Average weekly demand (adjusted for stockout censoring)
- On-hand vs available weeks of cover
- Inventory turns, sell-through, safety stock, reorder point
- Stockout probability, excess units/value, obsolescence risk score

### Inventory Actions

`REPLENISH`, `EXPEDITE_ORDER`, `INTER_REGION_TRANSFER`, `PRICE_MARKDOWN`, `HOLD_PRICE`, `STOP_OR_DELAY_ORDER`, `LIQUIDATION_REVIEW`, `MANUAL_REVIEW`, `NO_ACTION`

Decision order: **Transfer → Replenish → Stop order → Markdown (only if net economics positive)**

### Key Rules

- **Stockout protection**: No aggressive markdown on understocked SKUs
- **Low elasticity + overstock**: Prefer transfer/stop-order over markdown
- **Markdown economics**: Only recommend if `net_markdown_value > 0`
- **Joint confidence**: LOW confidence restricts price moves and transfers

### Backtest Strategies

Compares: Current, Pricing-Only, Inventory-Only, Joint Pricing+Inventory, Cost-Plus, Competitor Match

Results are **modeled/simulated estimates**, not proven causal lift.

---

## Limitations

- All data is synthetic; real-world performance will differ
- Observational backtest, not randomized experiment
- No external signals (weather, macro, web traffic)
- Single demand model, no uncertainty quantification
- Elasticity estimates have wide confidence intervals for sparse groups
- Recommendations require human approval; no auto-writeback

## Production Roadmap

1. Connect to production catalog and POS data feeds
2. Implement randomized price A/B testing
3. Deploy model serving API with drift monitoring
4. Scale scoring to millions of SKUs (distributed batch)
5. Integrate approval workflow with ERP/pricing systems
6. Add LLM-powered analyst (pluggable provider interface ready)

## Privacy Statement

- All data is 100% synthetic, generated by this project's data engine
- No former-employer data is used
- No data, code, or models from other portfolio projects are reused
- No API keys, external databases, or paid datasets required
- No online LLM calls; AI Analyst runs locally and deterministically

## License

Portfolio demonstration project. Not for commercial use without modification.
