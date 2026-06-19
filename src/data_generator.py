"""合成业务数据生成器：产品主数据与周度销售数据。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    CATEGORIES,
    CATEGORY_ELASTICITY_PRIORS,
    CUSTOMER_TIERS,
    N_SKUS,
    N_WEEKS,
    POLICY_VERSION,
    RANDOM_SEED,
    REGION_DEMAND_FACTORS,
    REGIONS,
    TIER_SENSITIVITY,
)

# 子品类映射
SUBCATEGORIES = {
    "Brakes": ["Brake Pads", "Rotors", "Calipers", "Brake Fluid"],
    "Filters": ["Oil Filter", "Air Filter", "Cabin Filter", "Fuel Filter"],
    "Electrical": ["Alternator", "Starter", "Wiring Harness", "Fuses"],
    "Suspension": ["Shocks", "Struts", "Control Arms", "Bushings"],
    "Fluids": ["Motor Oil", "Coolant", "Transmission Fluid", "Power Steering"],
    "Tools": ["Wrenches", "Sockets", "Jacks", "Diagnostic Tools"],
    "Batteries": ["Car Battery", "Marine Battery", "Battery Charger"],
    "Engine Components": ["Spark Plugs", "Belts", "Gaskets", "Pistons"],
}

BRAND_TIERS = ["Premium", "Mid-Tier", "Economy", "Private Label"]
SUPPLIERS = ["AutoParts Co", "Pacific Supply", "Northern Dist", "ValueLine Parts"]
LIFECYCLE_STAGES = ["New", "Growth", "Mature", "Decline"]

PRICE_CHANGE_REASONS = [
    "scheduled_review",
    "supplier_cost_change",
    "regional_price_test",
    "promotion",
    "competitor_response",
    "tier_discount_policy",
    "random_regional_test",
    "no_change",
]


def _rng(seed: int = RANDOM_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_product_master(n_skus: int = N_SKUS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """生成产品主数据。"""
    rng = _rng(seed)
    records = []

    for i in range(n_skus):
        sku_id = f"SKU-{i+1:05d}"
        category = rng.choice(CATEGORIES)
        subcategory = rng.choice(SUBCATEGORIES[category])
        brand_tier = rng.choice(BRAND_TIERS, p=[0.25, 0.35, 0.25, 0.15])
        private_label = brand_tier == "Private Label"

        # 成本与定价
        base_cost = rng.uniform(5, 500)
        if category == "Tools":
            base_cost = rng.uniform(15, 300)
        elif category == "Engine Components":
            base_cost = rng.uniform(20, 800)

        unit_cost = round(base_cost, 2)
        margin_pct = rng.uniform(0.18, 0.45)
        regular_retail_price = round(unit_cost / (1 - margin_pct), 2)
        minimum_margin_pct = round(rng.uniform(0.10, 0.20), 3)
        minimum_advertised_price = round(regular_retail_price * rng.uniform(0.85, 0.98), 2)

        # 弹性
        el_lo, el_hi = CATEGORY_ELASTICITY_PRIORS[category]
        true_elasticity = round(rng.uniform(el_lo, el_hi), 3)

        # 生命周期
        product_age_weeks = int(rng.integers(4, 200))
        if product_age_weeks < 26:
            lifecycle = "New"
        elif product_age_weeks < 78:
            lifecycle = "Growth" if rng.random() < 0.6 else "Mature"
        elif product_age_weeks < 130:
            lifecycle = "Mature"
        else:
            lifecycle = "Decline"

        base_demand = rng.uniform(2, 80)
        if category == "Filters":
            base_demand = rng.uniform(10, 120)
        elif category == "Tools":
            base_demand = rng.uniform(1, 25)

        seasonality_strength = 0.0
        if category in ("Batteries", "Fluids"):
            seasonality_strength = rng.uniform(0.2, 0.6)

        records.append({
            "sku_id": sku_id,
            "product_name": f"{subcategory} - {brand_tier} #{i+1}",
            "category": category,
            "subcategory": subcategory,
            "brand_tier": brand_tier,
            "private_label_flag": private_label,
            "unit_cost": unit_cost,
            "regular_retail_price": regular_retail_price,
            "minimum_margin_pct": minimum_margin_pct,
            "minimum_advertised_price": minimum_advertised_price,
            "supplier": rng.choice(SUPPLIERS),
            "lead_time_days": int(rng.integers(3, 45)),
            "case_pack": int(rng.choice([1, 1, 1, 6, 12, 24])),
            "product_age_weeks": product_age_weeks,
            "lifecycle_stage": lifecycle,
            "base_demand": round(base_demand, 2),
            "true_price_elasticity": true_elasticity,
            "seasonality_strength": round(seasonality_strength, 3),
            "inventory_holding_cost": round(unit_cost * rng.uniform(0.02, 0.08), 4),
            "obsolescence_risk": round(rng.uniform(0.01, 0.3), 3),
            "minimum_order_quantity": int(rng.choice([1, 1, 6, 12, 24])),
        })

    return pd.DataFrame(records)


def _seasonality_index(week_num: int, strength: float, category: str) -> float:
    """季节性指数。"""
    if strength <= 0:
        return 1.0
    # 电池冬季、流体夏季
    if category == "Batteries":
        peak_week = 48  # 冬季
    elif category == "Fluids":
        peak_week = 26  # 夏季
    else:
        peak_week = 1
    phase = 2 * np.pi * (week_num - peak_week) / 52
    return 1.0 + strength * 0.5 * np.cos(phase)


def _lifecycle_factor(stage: str, age_weeks: int) -> float:
    """生命周期需求因子。"""
    if stage == "New":
        return min(1.0, 0.3 + age_weeks * 0.05)
    if stage == "Growth":
        return 1.0 + min(0.3, age_weeks * 0.003)
    if stage == "Decline":
        return max(0.4, 1.0 - (age_weeks - 130) * 0.005)
    return 1.0


def _tier_discount(tier: str, rng: np.random.Generator) -> float:
    """客户层级折扣。"""
    base = {"Retail": 0.0, "Trade": 0.08, "Fleet": 0.15}[tier]
    return base + rng.uniform(-0.02, 0.02)


def _tier_price(regular: float, tier: str, discount: float) -> float:
    """层级价格。"""
    return round(regular * (1 - discount), 2)


def generate_weekly_sales(
    products: pd.DataFrame,
    n_weeks: int = N_WEEKS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """生成周度销售数据。"""
    rng = _rng(seed + 1)
    start_date = pd.Timestamp("2022-01-03")
    weeks = [start_date + pd.Timedelta(weeks=w) for w in range(n_weeks)]

    records = []
    # 状态追踪
    inventory_state: dict[tuple, float] = {}

    for week_idx, week_start in enumerate(weeks):
        week_num = week_idx + 1
        economic_index = 1.0 + 0.05 * np.sin(2 * np.pi * week_num / 52) + rng.normal(0, 0.02)

        for _, prod in products.iterrows():
            sku_id = prod["sku_id"]
            category = prod["category"]
            true_el = prod["true_price_elasticity"]
            season_str = prod["seasonality_strength"]
            season_idx = _seasonality_index(week_num, season_str, category)
            lc_factor = _lifecycle_factor(prod["lifecycle_stage"], prod["product_age_weeks"] + week_idx)

            for region in REGIONS:
                region_factor = REGION_DEMAND_FACTORS[region]

                for tier in CUSTOMER_TIERS:
                    key = (sku_id, region, tier)
                    tier_sens = TIER_SENSITIVITY[tier]

                    # 价格设定（部分外生）
                    unit_cost = prod["unit_cost"]
                    # 供应商成本变化
                    if rng.random() < 0.02:
                        unit_cost = round(unit_cost * rng.uniform(0.95, 1.08), 2)

                    regular_price = prod["regular_retail_price"]
                    tier_disc = _tier_discount(tier, rng)
                    base_tier_price = _tier_price(regular_price, tier, tier_disc)

                    # 价格变化原因
                    price_change_reason = "no_change"
                    price_test_flag = False
                    policy_ver = POLICY_VERSION

                    if rng.random() < 0.04:
                        price_change_reason = "scheduled_review"
                        base_tier_price = round(base_tier_price * rng.uniform(0.92, 1.08), 2)
                    elif rng.random() < 0.03:
                        price_change_reason = "supplier_cost_change"
                        base_tier_price = round(base_tier_price * rng.uniform(0.95, 1.10), 2)
                    elif rng.random() < 0.02:
                        price_change_reason = "random_regional_test"
                        price_test_flag = True
                        base_tier_price = round(base_tier_price * rng.uniform(0.90, 1.10), 2)
                    elif rng.random() < 0.02:
                        price_change_reason = "tier_discount_policy"
                        tier_disc = _tier_discount(tier, rng)
                        base_tier_price = _tier_price(regular_price, tier, tier_disc)

                    # 促销
                    promotion_flag = rng.random() < 0.08
                    promotion_depth = 0.0
                    if promotion_flag:
                        price_change_reason = "promotion"
                        promotion_depth = rng.uniform(0.05, 0.20)
                        base_tier_price = round(base_tier_price * (1 - promotion_depth), 2)

                    realized_price = base_tier_price

                    # 少量层级价格倒挂异常
                    if rng.random() < 0.005 and tier == "Fleet":
                        realized_price = round(realized_price * rng.uniform(1.05, 1.15), 2)

                    # 竞争对手价格
                    comp_noise = rng.uniform(0.92, 1.08)
                    competitor_price = round(regular_price * comp_noise * (0.95 if tier != "Retail" else 1.0), 2)
                    competitor_price_index = realized_price / competitor_price if competitor_price > 0 else 1.0

                    # 库存（含地区不平衡注入）
                    if key not in inventory_state:
                        init_inv = rng.uniform(20, 200)
                        sku_num = int(sku_id.split("-")[1])
                        if sku_num % 25 == 0:
                            if region == "Lower Mainland":
                                init_inv *= 3.5
                            elif region == "Northern BC":
                                init_inv *= 0.25
                            elif region == "Vancouver Island" and sku_num % 50 == 0:
                                init_inv *= 0.15
                        if sku_num % 17 == 0 and prod["lifecycle_stage"] == "Decline":
                            init_inv *= 2.5
                        inventory_state[key] = init_inv

                    beginning_inventory = inventory_state[key]
                    receipts = 0.0
                    if beginning_inventory < 30 and rng.random() < 0.3:
                        receipts = rng.uniform(50, 150)
                    available = beginning_inventory + receipts

                    # 潜在需求
                    ref_price = regular_price
                    price_ratio = realized_price / ref_price if ref_price > 0 else 1.0
                    price_response = price_ratio ** (true_el * tier_sens)

                    promo_effect = 1.0 + promotion_depth * 1.5 if promotion_flag else 1.0
                    comp_effect = 1.0 - max(0, competitor_price_index - 1.0) * 0.5
                    comp_effect = max(0.5, comp_effect)

                    latent_demand = (
                        prod["base_demand"]
                        * region_factor
                        * price_response
                        * season_idx
                        * lc_factor
                        * promo_effect
                        * comp_effect
                        * economic_index
                        * rng.lognormal(0, 0.15)
                    )
                    latent_demand = max(0, latent_demand)

                    # 观测销量（库存截断）
                    units_sold = min(round(min(latent_demand, available), 2), available)
                    lost_sales = max(0, latent_demand - available)
                    ending_inventory = max(0, available - units_sold)
                    inventory_state[key] = ending_inventory

                    on_order = receipts * 0.5 if receipts > 0 else (rng.uniform(0, 30) if ending_inventory < 20 else 0)

                    revenue = round(realized_price * units_sold, 2)
                    cogs = round(unit_cost * units_sold, 2)
                    gross_profit = round(revenue - cogs, 2)
                    gross_margin_pct = gross_profit / revenue if revenue > 0 else 0.0

                    weekly_demand_avg = prod["base_demand"] * region_factor
                    weeks_of_cover = ending_inventory / weekly_demand_avg if weekly_demand_avg > 0 else 999

                    stockout_flag = ending_inventory < 5 and lost_sales > 0
                    excess_flag = weeks_of_cover > 16
                    slow_flag = weeks_of_cover > 20 and units_sold < prod["base_demand"] * 0.3

                    records.append({
                        "week_start": week_start,
                        "week_num": week_num,
                        "sku_id": sku_id,
                        "category": category,
                        "region": region,
                        "customer_tier": tier,
                        "unit_cost": unit_cost,
                        "regular_price": regular_price,
                        "realized_price": realized_price,
                        "customer_tier_discount_pct": round(tier_disc + promotion_depth, 4),
                        "competitor_price": competitor_price,
                        "competitor_price_index": round(competitor_price_index, 4),
                        "promotion_flag": promotion_flag,
                        "promotion_depth": round(promotion_depth, 4),
                        "seasonality_index": round(season_idx, 4),
                        "economic_demand_index": round(economic_index, 4),
                        "beginning_inventory": round(beginning_inventory, 2),
                        "receipts": round(receipts, 2),
                        "units_sold": round(units_sold, 2),
                        "latent_demand": round(latent_demand, 2),
                        "lost_sales_estimate": round(lost_sales, 2),
                        "ending_inventory": round(ending_inventory, 2),
                        "on_order": round(on_order, 2),
                        "revenue": revenue,
                        "cogs": cogs,
                        "gross_profit": gross_profit,
                        "gross_margin_pct": round(gross_margin_pct, 4),
                        "weeks_of_cover": round(weeks_of_cover, 2),
                        "stockout_flag": stockout_flag,
                        "excess_inventory_flag": excess_flag,
                        "slow_moving_flag": slow_flag,
                        "price_change_reason": price_change_reason,
                        "price_test_flag": price_test_flag,
                        "policy_version": policy_ver,
                    })

    return pd.DataFrame(records)


def generate_all_data(
    n_skus: int = N_SKUS,
    n_weeks: int = N_WEEKS,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成全部合成数据。"""
    products = generate_product_master(n_skus, seed)
    sales = generate_weekly_sales(products, n_weeks, seed)
    return products, sales
