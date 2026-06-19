"""库存分析与规则 — 委托至 inventory_metrics。"""

from __future__ import annotations

import pandas as pd

from src.config import SLOW_MOVING_WEEKS, WEEKS_OF_COVER_EXCESS, WEEKS_OF_COVER_STOCKOUT_RISK
from src.inventory_metrics import (
    build_sku_region_metrics,
    compute_weeks_of_cover,
    compute_inventory_turns,
)
from src.inventory_policy import classify_inventory_status
from src.utils import safe_divide


def analyze_inventory(
    sales: pd.DataFrame,
    products: pd.DataFrame,
    override: dict | None = None,
) -> dict:
    """库存全景分析（SKU×Region 层级）。"""
    inv_metrics = build_sku_region_metrics(sales, products, override)
    if inv_metrics.empty:
        return {"latest_snapshot": pd.DataFrame(), "total_inventory_value": 0}

    inv_metrics["inventory_status"] = inv_metrics.apply(
        lambda r: classify_inventory_status(r, override), axis=1
    )
    inv_metrics["inventory_value"] = inv_metrics["on_hand_inventory"] * inv_metrics["unit_cost"]
    inv_metrics["is_excess"] = inv_metrics["inventory_status"].isin(
        ["OVERSTOCKED", "SLOW_MOVING", "OBSOLETE_RISK"]
    )
    inv_metrics["is_stockout_risk"] = inv_metrics["inventory_status"].isin(
        ["STOCKOUT", "STOCKOUT_RISK", "UNDERSTOCKED"]
    )
    inv_metrics["is_slow_moving"] = inv_metrics["inventory_status"] == "SLOW_MOVING"

    total_value = inv_metrics["inventory_value"].sum()
    excess_value = inv_metrics.loc[inv_metrics["is_excess"], "excess_inventory_value"].sum()
    stockout_risk_count = inv_metrics["is_stockout_risk"].sum()
    slow_count = inv_metrics["is_slow_moving"].sum()
    avg_woc = inv_metrics["available_weeks_of_cover"].mean()
    total_lost = inv_metrics["lost_sales_estimate"].sum()

    cogs_by_sr = sales.groupby(["sku_id", "region"]).apply(
        lambda g: (g["unit_cost"] * g["units_sold"]).sum()
    ).reset_index(name="annual_cogs")
    inv_m = inv_metrics.merge(cogs_by_sr, on=["sku_id", "region"], how="left")
    annual_cogs = inv_m["annual_cogs"].sum()
    avg_inv_val = inv_metrics["inventory_value"].sum()
    turns = compute_inventory_turns(annual_cogs, avg_inv_val) if avg_inv_val > 0 else 0

    status_dist = inv_metrics["inventory_status"].value_counts().to_dict()
    action_dist = inv_metrics.get("inventory_action", pd.Series()).value_counts().to_dict() if "inventory_action" in inv_metrics.columns else {}

    return {
        "latest_snapshot": inv_metrics,
        "total_inventory_value": total_value,
        "excess_inventory_value": excess_value,
        "stockout_risk_skus": int(stockout_risk_count),
        "slow_moving_skus": int(slow_count),
        "avg_weeks_of_cover": avg_woc,
        "avg_inventory_turns": turns,
        "estimated_lost_sales": total_lost,
        "status_distribution": status_dist,
        "excess_by_category": (
            inv_metrics[inv_metrics["is_excess"]]
            .groupby("category")["excess_inventory_value"]
            .sum()
            .to_dict()
        ),
    }


def classify_inventory_action(row: pd.Series) -> str:
    """向后兼容：库存行动分类。"""
    from src.inventory_policy import determine_inventory_action
    action, _ = determine_inventory_action(row)
    mapping = {
        "REPLENISH": "Replenishment",
        "EXPEDITE_ORDER": "Replenishment",
        "INTER_REGION_TRANSFER": "Inter-region transfer",
        "PRICE_MARKDOWN": "Price Markdown",
        "STOP_OR_DELAY_ORDER": "Hold",
        "NO_ACTION": "Hold",
        "HOLD_PRICE": "Hold",
    }
    return mapping.get(action, "Hold")


def simulate_inventory_impact(
    current_inventory: float,
    predicted_units: float,
    receipts: float = 0,
) -> dict:
    """模拟库存影响。"""
    ending = max(0, current_inventory + receipts - predicted_units)
    weekly_demand = predicted_units if predicted_units > 0 else 1
    woc = compute_weeks_of_cover(ending, weekly_demand)
    stockout_prob = 1.0 if ending < 2 and predicted_units > current_inventory else (
        0.8 if ending < 5 else 0.1 if ending < 10 else 0.0
    )
    return {
        "expected_ending_inventory": ending,
        "expected_weeks_of_cover": woc,
        "stockout_probability": stockout_prob,
        "excess_inventory_remaining": ending if woc > WEEKS_OF_COVER_EXCESS else 0,
    }
