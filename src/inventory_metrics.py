"""统一库存指标定义 — 所有模块必须调用此处的函数。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import INVENTORY_POLICY


def _cfg(key: str, override: dict | None = None) -> float:
    if override and key in override:
        return override[key]
    return INVENTORY_POLICY[key]


def compute_average_weekly_demand(
    history: pd.DataFrame,
    window: int | None = None,
    override: dict | None = None,
) -> tuple[float, float]:
    """
    计算平均周需求与标准差。
    优先使用 adjusted/latent demand，避免 stockout 截断偏差。
    """
    window = window or int(_cfg("demand_window_weeks", override))
    if history.empty:
        return 0.0, 0.0

    h = history.sort_values("week_num").tail(window)
    if "adjusted_units" in h.columns:
        demand = h["adjusted_units"]
    elif "latent_demand" in h.columns:
        demand = h["latent_demand"]
    else:
        demand = h["units_sold"] + h.get("lost_sales_estimate", 0)

    avg = float(demand.mean()) if len(demand) > 0 else 0.0
    std = float(demand.std()) if len(demand) > 1 else avg * 0.2
    return max(avg, 0.0), max(std, 0.0)


def compute_weeks_of_cover(
    inventory: float,
    weekly_demand: float,
    override: dict | None = None,
) -> float:
    """库存覆盖周数。"""
    eps = _cfg("demand_epsilon", override)
    if weekly_demand <= eps:
        return 999.0
    return inventory / weekly_demand


def compute_inventory_turns(
    annual_cogs: float,
    avg_inventory_value: float,
) -> float:
    """库存周转率，分母为0时返回0。"""
    if avg_inventory_value <= 0:
        return 0.0
    return annual_cogs / avg_inventory_value


def compute_sell_through(
    units_sold: float,
    beginning_inventory: float,
    receipts: float,
) -> float:
    """售罄率。"""
    denom = beginning_inventory + receipts
    if denom <= 0:
        return 0.0
    return min(1.0, units_sold / denom)


def compute_safety_stock(
    demand_std: float,
    lead_time_weeks: float,
    override: dict | None = None,
) -> float:
    """安全库存。"""
    z = _cfg("service_level_z", override)
    lt = max(lead_time_weeks, 0.1)
    return max(0.0, z * demand_std * math.sqrt(lt))


def compute_reorder_point(
    expected_weekly_demand: float,
    lead_time_weeks: float,
    safety_stock: float,
) -> float:
    """再订货点。"""
    lead_time_demand = expected_weekly_demand * max(lead_time_weeks, 0.1)
    return max(0.0, lead_time_demand + safety_stock)


def compute_stockout_probability(
    on_hand: float,
    on_order: float,
    expected_weekly_demand: float,
    demand_std: float,
    lead_time_weeks: float,
) -> float:
    """基于 lead-time 需求分布近似缺货概率。"""
    if expected_weekly_demand <= 0:
        return 0.0
    available = on_hand + on_order
    lt = max(lead_time_weeks, 0.1)
    lead_time_demand = expected_weekly_demand * lt
    lead_time_std = demand_std * math.sqrt(lt)
    if lead_time_std <= 0:
        return 1.0 if available < lead_time_demand else 0.0
    z_score = (available - lead_time_demand) / lead_time_std
    # 标准正态 CDF 近似
    prob = 0.5 * (1 + math.erf(-z_score / math.sqrt(2)))
    return float(np.clip(prob, 0.0, 1.0))


def compute_excess_units(
    available_inventory: float,
    expected_weekly_demand: float,
    override: dict | None = None,
) -> float:
    """过剩库存单位。"""
    target_woc = _cfg("target_weeks_of_cover", override)
    target_units = target_woc * max(expected_weekly_demand, _cfg("demand_epsilon", override))
    return max(0.0, available_inventory - target_units)


def compute_excess_inventory_value(excess_units: float, unit_cost: float) -> float:
    """过剩库存价值。"""
    return max(0.0, excess_units * unit_cost)


def compute_lost_sales_estimate(
    latent_demand: float,
    units_sold: float,
) -> float:
    """丢失销量估计。"""
    return max(0.0, latent_demand - units_sold)


def compute_holding_cost(
    excess_inventory_value: float,
    override: dict | None = None,
    period_fraction: float = 1 / 52,
) -> float:
    """库存持有成本。"""
    rate = _cfg("annual_holding_cost_rate", override)
    return excess_inventory_value * rate * period_fraction


def compute_obsolescence_risk_score(
    product_age_weeks: float,
    lifecycle_stage: str,
    weeks_of_cover: float,
    demand_trend: float,
    sell_through: float,
    base_obsolescence: float = 0.0,
) -> float:
    """
    过时风险评分 0–1。
    demand_trend: 正值=增长，负值=下降。
    """
    score = base_obsolescence
    if lifecycle_stage in ("Decline", "End-of-Life"):
        score += 0.35
    if product_age_weeks > 130:
        score += 0.15
    if weeks_of_cover > 20:
        score += 0.20
    if demand_trend < -0.1:
        score += 0.15
    if sell_through < 0.2:
        score += 0.15
    return float(np.clip(score, 0.0, 1.0))


def compute_demand_trend(history: pd.DataFrame, window: int = 8) -> float:
    """需求趋势：近半 vs 远半变化率。"""
    h = history.sort_values("week_num").tail(window)
    if len(h) < 4:
        return 0.0
    col = "adjusted_units" if "adjusted_units" in h.columns else "units_sold"
    half = len(h) // 2
    recent = h[col].tail(half).mean()
    prior = h[col].head(half).mean()
    if prior <= 0:
        return 0.0
    return (recent - prior) / prior


def round_to_case_pack(quantity: float, case_pack: int) -> float:
    """按箱规向上取整。"""
    if case_pack <= 1:
        return max(0.0, math.ceil(quantity))
    if quantity <= 0:
        return 0.0
    return float(math.ceil(quantity / case_pack) * case_pack)


def build_sku_region_metrics(
    sales: pd.DataFrame,
    products: pd.DataFrame,
    override: dict | None = None,
) -> pd.DataFrame:
    """构建 SKU × Region 层级库存指标快照。"""
    prod_cols = [
        "sku_id", "category", "unit_cost", "lead_time_days", "case_pack",
        "product_age_weeks", "lifecycle_stage", "obsolescence_risk",
        "inventory_holding_cost", "minimum_margin_pct",
    ]
    if "minimum_order_quantity" in products.columns:
        prod_cols.append("minimum_order_quantity")
    prod = products[[c for c in prod_cols if c in products.columns]].copy()

    # 按 SKU×Region 聚合（跨 tier）
    latest = (
        sales.sort_values("week_num")
        .groupby(["sku_id", "region"])
        .last()
        .reset_index()
    )
    agg = (
        sales.groupby(["sku_id", "region"])
        .agg(
            total_units_sold=("units_sold", "sum"),
            avg_ending_inv=("ending_inventory", "mean"),
            total_lost_sales=("lost_sales_estimate", "sum"),
        )
        .reset_index()
    )

    records = []
    for _, row in latest.iterrows():
        sku, region = row["sku_id"], row["region"]
        hist = sales[(sales["sku_id"] == sku) & (sales["region"] == region)]
        tier_hist = hist  # 全部 tier

        avg_demand, demand_std = compute_average_weekly_demand(tier_hist, override=override)
        trend = compute_demand_trend(tier_hist)

        # 跨 tier 汇总库存
        tier_latest = (
            sales[(sales["sku_id"] == sku) & (sales["region"] == region)]
            .sort_values("week_num")
            .groupby("customer_tier")
            .last()
        )
        on_hand = float(tier_latest["ending_inventory"].sum())
        on_order = float(tier_latest["on_order"].sum()) if "on_order" in tier_latest.columns else 0.0
        available = on_hand + on_order

        prod_row = prod[prod["sku_id"] == sku]
        if prod_row.empty:
            continue
        p = prod_row.iloc[0]
        lead_time_weeks = p.get("lead_time_days", 14) / 7.0
        unit_cost = p["unit_cost"]

        on_hand_woc = compute_weeks_of_cover(on_hand, avg_demand, override)
        avail_woc = compute_weeks_of_cover(available, avg_demand, override)
        safety = compute_safety_stock(demand_std, lead_time_weeks, override)
        rop = compute_reorder_point(avg_demand, lead_time_weeks, safety)
        excess = compute_excess_units(available, avg_demand, override)
        excess_val = compute_excess_inventory_value(excess, unit_cost)

        latent = float(tier_latest.get("latent_demand", tier_latest["units_sold"]).sum()) if len(tier_latest) else 0
        sold = float(tier_latest["units_sold"].sum())
        lost = compute_lost_sales_estimate(latent, sold) if latent > sold else float(
            tier_latest.get("lost_sales_estimate", pd.Series([0])).sum()
        )

        begin_inv = float(tier_latest["beginning_inventory"].sum()) if "beginning_inventory" in tier_latest.columns else on_hand
        receipts = float(tier_latest["receipts"].sum()) if "receipts" in tier_latest.columns else 0
        sell_through = compute_sell_through(sold, begin_inv, receipts)

        stockout_prob = compute_stockout_probability(
            on_hand, on_order, avg_demand, demand_std, lead_time_weeks
        )
        obs_score = compute_obsolescence_risk_score(
            p.get("product_age_weeks", 52),
            p.get("lifecycle_stage", "Mature"),
            avail_woc, trend, sell_through,
            p.get("obsolescence_risk", 0.1),
        )

        annual_cogs = unit_cost * tier_hist["units_sold"].sum()
        avg_inv_val = unit_cost * max(on_hand, 1)
        turns = compute_inventory_turns(annual_cogs, avg_inv_val * 52)

        records.append({
            "sku_id": sku,
            "category": p.get("category", row.get("category", "")),
            "region": region,
            "unit_cost": unit_cost,
            "on_hand_inventory": on_hand,
            "on_order_inventory": on_order,
            "available_inventory": available,
            "average_weekly_demand": round(avg_demand, 3),
            "demand_std": round(demand_std, 3),
            "demand_trend": round(trend, 4),
            "on_hand_weeks_of_cover": round(on_hand_woc, 2),
            "available_weeks_of_cover": round(avail_woc, 2),
            "inventory_turns": round(turns, 3),
            "sell_through_rate": round(sell_through, 4),
            "stockout_probability": round(stockout_prob, 4),
            "safety_stock": round(safety, 2),
            "reorder_point": round(rop, 2),
            "excess_units": round(excess, 2),
            "excess_inventory_value": round(excess_val, 2),
            "lost_sales_estimate": round(lost, 2),
            "obsolescence_risk_score": round(obs_score, 4),
            "lead_time_weeks": round(lead_time_weeks, 2),
            "case_pack": int(p.get("case_pack", 1)),
            "minimum_order_quantity": int(p.get("minimum_order_quantity", 1)),
            "lifecycle_stage": p.get("lifecycle_stage", "Mature"),
            "product_age_weeks": p.get("product_age_weeks", 52),
        })

    return pd.DataFrame(records)
