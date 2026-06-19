# Field: sku_id
Type: string
Description: Unique SKU identifier (e.g., SKU-00001)

# Field: product_name
Type: string
Description: Human-readable product name

# Field: category
Type: string
Values: Brakes, Filters, Electrical, Suspension, Fluids, Tools, Batteries, Engine Components

# Field: subcategory
Type: string
Description: Product subcategory within category

# Field: brand_tier
Type: string
Values: Premium, Mid-Tier, Economy, Private Label

# Field: unit_cost
Type: float
Description: Current unit cost from supplier

# Field: regular_retail_price
Type: float
Description: Standard retail list price

# Field: minimum_margin_pct
Type: float
Description: Minimum acceptable gross margin percentage

# Field: minimum_advertised_price
Type: float
Description: MAP constraint for advertised pricing

# Field: true_price_elasticity
Type: float
Description: Ground-truth elasticity used ONLY for data generation and validation. NOT a model feature.

---

## Weekly Sales (synthetic_sales.csv)

# Field: week_start
Type: date
Description: Monday of the sales week

# Field: sku_id, category, region, customer_tier
Type: string
Description: Dimensional keys

# Field: realized_price
Type: float
Description: Actual transaction price after tier discounts and promotions

# Field: competitor_price
Type: float
Description: Primary competitor's price for comparable product

# Field: units_sold
Type: float
Description: Observed units sold (censored by inventory availability)

# Field: latent_demand
Type: float
Description: True demand before inventory censoring

# Field: lost_sales_estimate
Type: float
Description: Estimated lost sales due to stockout

# Field: price_change_reason
Type: string
Values: scheduled_review, supplier_cost_change, regional_price_test, promotion, competitor_response, tier_discount_policy, random_regional_test, no_change

# Field: price_test_flag
Type: boolean
Description: Whether this week's price was set by an exogenous test

# Field: policy_version
Type: string
Description: Pricing policy version identifier
