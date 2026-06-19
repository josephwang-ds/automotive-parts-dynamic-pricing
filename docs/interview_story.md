# Interview Story: Automotive Parts Dynamic Pricing

## 1. 30-Second Introduction

"I built a dynamic pricing and inventory optimization system for automotive parts retail. Instead of predicting what a part 'should cost,' the system predicts how many units will sell at each candidate price, then optimizes for gross profit, revenue, or inventory reduction. It handles customer-tier pricing across Retail, Trade, and Fleet, with guardrails for margin floors, MAP constraints, and stockout protection. All recommendations go through a human approval queue. The demo uses independently generated synthetic data designed for a production-scale workflow."

## 2. Two-Minute Introduction

"This portfolio project modernizes the type of pricing, margin, customer-tier, and inventory analysis I worked with earlier in my career, using independently generated synthetic data and a modern modeling architecture.

The business problem: an auto parts retailer with thousands of SKUs across multiple regions and customer tiers needs to set prices that balance margin, competitiveness, and inventory health. Static cost-plus pricing leaves money on the table; reactive competitor matching erodes margin.

My solution has four layers:

1. **Demand forecasting** — A gradient boosting model predicts unconstrained weekly demand at any candidate price, adjusted for stockout censoring so the model doesn't learn 'low demand' during inventory shortages.

2. **Elasticity estimation** — Log-log regression on exogenous price-test observations estimates price sensitivity by category and customer tier, with shrinkage for low-sample groups.

3. **Price optimization** — For each SKU, the system simulates 31 candidate prices (-15% to +15%), scores them against four objectives, and applies nine business guardrails.

4. **Decision support** — A Streamlit dashboard with eight analytical views, a backtest comparing four pricing policies, a rollback simulator, and a local AI analyst that answers questions using actual computed metrics.

Key design choices: demand-first (not price-first), time-based model validation, human-in-the-loop approval, and honest labeling of backtest results as 'modeled lift' rather than proven business impact."

## 3. Why Predict Demand, Not Price?

Price is an **action** we control; demand is the **outcome** we need to understand. Predicting "the right price" conflates causation with correlation — historical prices were set by managers responding to costs, competition, and inventory, creating endogeneity. By predicting demand at candidate prices, we can simulate the full business impact: revenue, gross profit, margin, ending inventory, weeks of cover, and stockout risk. This enables multi-objective optimization and transparent trade-off analysis.

## 4. How to Estimate Elasticity?

I use log-log regression: log(units) = β × log(price) + controls. Key design choices:
- **Exogenous variation**: Filter to `price_test_flag = true` and scheduled price reviews, avoiding prices set reactively to demand
- **Hierarchical estimation**: Category × tier → category × region × tier, with shrinkage to global prior for low-sample groups
- **Range constraints**: Clip to [-3.0, -0.1], classify as Highly Elastic to Inelastic
- **Separate from demand model**: Elasticity is for explanation and sanity check; the ML demand model handles the primary prediction

## 5. Stockout Censoring

When inventory runs out, observed sales < true demand. Without adjustment, the model learns that low inventory periods have low demand — backwards causality. Solution: create `adjusted_units = latent_demand` for stockout weeks (using the data generator's latent demand or units + lost_sales). The model trains on unconstrained demand, then inventory constraints are applied during price simulation.

## 6. Price Endogeneity

Historical prices aren't randomly assigned — they're set by managers who see demand signals. This creates simultaneity bias in elasticity estimation. Mitigations:
- Include exogenous price changes in synthetic data (supplier cost changes, randomized regional tests, policy-driven tier discounts)
- Prefer price-test observations for elasticity regression
- Use ML demand model (which captures complex interactions) separately from causal elasticity
- Document that predictive feature importance ≠ causal elasticity

## 7. Customer-Tier Pricing Design

Three tiers (Retail, Trade, Fleet) with typical discount ladders. The optimizer enforces Retail ≥ Trade ≥ Fleet via guardrails. Tier-specific elasticity estimation captures that Fleet and Trade buyers are more price-sensitive. Anomaly detection flags tier price inversions in the data.

## 8. Why Guardrails?

ML optimization can recommend prices that are mathematically optimal but business-infeasible: below margin floor, violating MAP, breaking tier ladder, or aggressively cutting prices on stockout-risk items. Nine guardrails ensure recommendations are actionable. Low-confidence elasticity estimates trigger smaller move limits and "Test" or "Manual Review" actions.

## 9. Backtesting Approach

Compare four policies over the 13-week test period: Current, Cost-Plus, Competitor Match, Dynamic. Results are labeled "modeled lift" / "simulated lift" — not "proven business impact." This is observational backtest on synthetic data; causal lift requires randomized price experiments in production.

## 10. Production Monitoring

Track: demand model drift (WAPE), elasticity drift, cost changes, competitor price changes, forecast error by segment, margin-floor violations, stockout rate changes, analyst override rate, and realized vs expected lift after controlled rollout.

## 11. Combining BI, Pricing Analytics, and AI Analyst

- **Traditional BI**: Executive dashboard with KPI cards, category/region breakdowns, inventory metrics
- **Pricing Analytics**: Elasticity explorer, candidate price curves, guardrail audit, backtest comparison
- **AI Analyst**: Natural language interface to computed metrics — deterministic, no hallucinated numbers, shows evidence and caveats. Designed with a pluggable provider interface for future LLM integration.

---

"I designed inventory as a constraint and operational-action layer around dynamic pricing. The system first determines whether a problem should be solved by price, replenishment, transfer, or purchasing control. This prevents the optimizer from treating every inventory imbalance as a markdown problem."
