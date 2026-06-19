# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                     │
│  Executive | Demand | Elasticity | Pricing | Inventory |    │
│  Backtest | AI Analyst | Governance                          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Pipeline (pipeline.py)                      │
│  Load → Features → Train → Elasticity → Optimize → Backtest│
└──┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬────────┘
   │      │      │      │      │      │      │      │
   ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
 Data  Features Demand Elastic Optim  Invent  Back   AI
 Gen              Model  ity           ory    test  Analyst
```

## Data Flow

1. **Synthetic Data Generation** (`data_generator.py`)
   - 3,000 representative SKUs × 4 regions × 3 tiers × 104 weeks
   - Demand function with price elasticity, seasonality, promotions
   - Inventory censoring (observed ≤ available)
   - Exogenous price changes for elasticity identification

2. **Feature Engineering** (`features.py`)
   - Lag features (1, 2, 4, 13 weeks) — no future leakage
   - Rolling aggregates (4, 8, 13 weeks)
   - Stockout-adjusted demand target
   - Time-based train/val/test split (78/13/13)

3. **Demand Model** (`demand_model.py`)
   - Seasonal naive baseline
   - Ridge / ElasticNet linear model
   - HistGradientBoostingRegressor (primary)
   - Predicts unconstrained demand at candidate prices

4. **Elasticity Estimation** (`elasticity.py`)
   - Log-log regression on price-test observations
   - Category × tier and category × region × tier levels
   - Shrinkage to global prior for low-confidence groups

5. **Price Optimization** (`optimizer.py`)
   - Candidate price simulation (-15% to +15%, 1% steps)
   - Four objective functions (GP, Revenue, Inventory, Balanced)
   - Nine guardrail rules
   - Human approval queue output

6. **Backtest** (`backtest.py`)
   - Compare: Current, Cost-Plus, Competitor Match, Dynamic
   - Modeled lift (not proven causal impact)
   - Rollback simulator (0–100%)

## Key Design Decisions

- **Demand-first, not price-first**: System predicts units at candidate prices, then computes revenue/profit/inventory impact
- **Stockout censoring**: Adjusted demand target prevents model from learning "low demand" during stockouts
- **Price endogeneity**: Exogenous price tests enable more credible elasticity estimation
- **Guardrails before optimization**: Business rules constrain the search space
- **Human-in-the-loop**: All recommendations require approval; no auto-writeback

## Module Dependencies

```
config.py ← all modules
utils.py ← metrics, inventory, optimizer, ai_analyst
data_generator.py ← features, pipeline
features.py ← demand_model, pipeline
demand_model.py ← optimizer, pipeline
elasticity.py ← optimizer, pipeline
optimizer.py ← pipeline, backtest
inventory.py ← optimizer, pipeline
backtest.py ← pipeline
ai_analyst.py ← app.py
pipeline.py ← app.py
```

## Production Roadmap

- Replace synthetic data with production catalog feed
- Add real-time competitor price ingestion
- Implement A/B price testing framework
- Deploy model serving API with monitoring
- Connect approval workflow to ERP/POS systems
- Scale to millions of SKUs with distributed scoring
