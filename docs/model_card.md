# Model Card: Demand Forecasting Model

## Model Details

- **Model type**: HistGradientBoostingRegressor (primary), Ridge Regression (comparison)
- **Version**: v1.0.0-synthetic-demo
- **Training date**: Generated at pipeline runtime
- **Framework**: scikit-learn

## Intended Use

- Predict unconstrained weekly unit demand at candidate price points
- Support dynamic pricing optimization and inventory planning
- **Not intended for**: Direct price setting without guardrails and human approval

## Training Data

- **Source**: Independently generated synthetic data
- **Size**: 3,000 SKUs × 4 regions × 3 tiers × 104 weeks ≈ 3.7M rows
- **Split**: Time-based — Train 78w / Validation 13w / Test 13w
- **Target**: `adjusted_units` (stockout-censored demand)

## Features

23 numeric + 4 categorical (one-hot encoded):
- Lag features: lag_1/2/4/13_units
- Rolling: rolling_4/8/13_units, rolling_4_revenue, rolling_13_margin
- Price: realized_price, price_change_pct, price_index_vs_competitor
- Context: category, region, customer_tier, lifecycle_stage, seasonality_index
- Inventory: prior_week_inventory, weeks_of_cover, lead_time_days

**Excluded**: `true_price_elasticity` (ground truth only)

## Performance Metrics

Evaluated on 13-week test set:
- MAE, RMSE, WAPE, RMSLE, Bias
- Improvement over seasonal naive baseline

## Limitations

1. **Synthetic data**: Performance on real data will differ
2. **Observational**: Model learns correlations, not causal price effects
3. **Stockout adjustment**: Approximate; production needs better censoring models
4. **No external signals**: Weather, macro, competitor promotions not modeled
5. **Single model**: No ensemble or uncertainty quantification

## Ethical Considerations

- Customer tier pricing must comply with fair business practices
- Recommendations require human review to prevent discriminatory pricing
- All data is synthetic; no real customer or transaction data used

## Monitoring (Production)

- Demand model drift (weekly WAPE tracking)
- Feature distribution shift
- Forecast bias by category/region
- Override rate by pricing analysts

## Elasticity Model (Separate)

- Log-log regression on exogenous price-test observations
- Hierarchical: global → category×tier → category×region×tier
- Shrinkage for low-sample groups
- Used for explanation and sanity check, not primary demand prediction
