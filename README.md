# Automotive Parts Dynamic Pricing & Inventory Optimization

A complete SKU-level dynamic pricing decision system for automotive parts retail. The system recommends customer-tier prices based on demand forecasting, price elasticity estimation, inventory position, and business guardrails.

> **The public demo uses a representative synthetic sample. The workflow is designed for production catalogs containing millions of SKUs.**

## Executive Summary

This system does **not** predict "what a part should cost." Instead, it predicts **how many units will sell at each candidate price** and computes the resulting revenue, gross profit, margin, inventory, and stockout risk. Recommendations are constrained by nine business guardrails and require human approval before implementation.

- **3,000 SKUs** × 4 regions × 3 customer tiers × 104 weeks of synthetic data
- **Demand model**: HistGradientBoostingRegressor with stockout-adjusted targets
- **Elasticity**: Log-log regression on exogenous price-test observations
- **Optimization**: 4 objectives, 31 candidate prices per SKU, 9 guardrails
- **UI**: 8-page Streamlit dashboard with AI Analyst

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

```bash
cd parts-dynamic-pricing-ai
pip install -r requirements.txt

# Generate synthetic data
python data/generate_synthetic_data.py

# Train models and generate recommendations
python -c "from src.pipeline import run_full_pipeline; run_full_pipeline()"

# Launch dashboard
streamlit run app.py

# Run tests
pytest tests/ -v
```

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

## UI Overview

8 tabs: Executive Command Center, Demand Model, Elasticity Explorer, Pricing Studio, Inventory & Markdown, Backtest & Rollback, AI Analyst, Data & Governance.

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
└── docs/                   # Architecture, model card, interview story
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
