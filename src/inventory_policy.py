"""库存健康分类与行动决策规则。"""

from __future__ import annotations

import pandas as pd

from src.config import INVENTORY_POLICY, INVENTORY_STATUS_PRIORITY


def classify_inventory_status(row: pd.Series, override: dict | None = None) -> str:
    """为 SKU×Region 分配唯一主库存状态。"""
    cfg = {**INVENTORY_POLICY, **(override or {})}

    flags = {}

    on_hand = row.get("on_hand_inventory", row.get("ending_inventory", 0))
    avail_woc = row.get("available_weeks_of_cover", row.get("weeks_of_cover", 99))
    on_hand_woc = row.get("on_hand_weeks_of_cover", avail_woc)
    stockout_prob = row.get("stockout_probability", 0)
    lead_time_weeks = row.get("lead_time_weeks", 2)
    turns = row.get("inventory_turns", 10)
    obs_score = row.get("obsolescence_risk_score", 0)
    lifecycle = row.get("lifecycle_stage", "Mature")
    demand_trend = row.get("demand_trend", 0)

    if on_hand <= 0:
        flags["STOCKOUT"] = True
    if stockout_prob >= cfg["stockout_probability_threshold"]:
        flags["STOCKOUT_RISK"] = True
    if on_hand_woc < lead_time_weeks and on_hand > 0:
        flags["STOCKOUT_RISK"] = True
    if obs_score >= cfg["obsolescence_threshold"] or (
        lifecycle in ("Decline", "End-of-Life") and avail_woc > cfg["minimum_weeks_of_cover"]
    ):
        flags["OBSOLETE_RISK"] = True
    if avail_woc < cfg["minimum_weeks_of_cover"] and on_hand > 0:
        flags["UNDERSTOCKED"] = True
    if turns < cfg["slow_moving_turns_threshold"] and avail_woc > cfg["target_weeks_of_cover"]:
        flags["SLOW_MOVING"] = True
    if avail_woc > cfg["maximum_weeks_of_cover"]:
        flags["OVERSTOCKED"] = True
    if (
        cfg["minimum_weeks_of_cover"] <= avail_woc <= cfg["maximum_weeks_of_cover"]
        and turns >= cfg["slow_moving_turns_threshold"]
        and demand_trend >= -0.15
    ):
        flags["HEALTHY"] = True

    for status in INVENTORY_STATUS_PRIORITY:
        if flags.get(status):
            return status
    return "HEALTHY"


def evaluate_markdown_economics(
    baseline_gp: float,
    projected_gp: float,
    excess_value: float,
    obsolescence_score: float,
    override: dict | None = None,
) -> dict:
    """评估 markdown 净经济价值。"""
    cfg = {**INVENTORY_POLICY, **(override or {})}
    holding_reduction = compute_holding_reduction(excess_value, cfg)
    obs_avoided = excess_value * obsolescence_score * 0.1
    net_value = (
        projected_gp - baseline_gp + holding_reduction + obs_avoided
    )
    return {
        "net_markdown_value": net_value,
        "holding_cost_reduction": holding_reduction,
        "avoided_obsolescence_loss": obs_avoided,
        "markdown_recommended": net_value > cfg["minimum_markdown_value"],
    }


def compute_holding_reduction(excess_value: float, cfg: dict) -> float:
    """持有成本节约。"""
    return excess_value * cfg.get("annual_holding_cost_rate", 0.25) * (4 / 52)


# Pricing vs Non-pricing Action Matrix（代码规则）
ACTION_MATRIX = {
    ("HEALTHY", "high"): ("Optimize margin", "NO_ACTION"),
    ("HEALTHY", "low"): ("Optimize margin", "NO_ACTION"),
    ("HEALTHY", "any"): ("Optimize margin", "NO_ACTION"),
    ("OVERSTOCKED", "high"): ("Markdown candidate", "STOP_OR_DELAY_ORDER"),
    ("OVERSTOCKED", "low"): ("Hold/Test", "INTER_REGION_TRANSFER"),
    ("STOCKOUT_RISK", "any"): ("Hold or small increase", "REPLENISH"),
    ("STOCKOUT", "any"): ("Hold", "EXPEDITE_ORDER"),
    ("UNDERSTOCKED", "any"): ("Hold or small increase", "REPLENISH"),
    ("SLOW_MOVING", "high"): ("Markdown candidate", "STOP_OR_DELAY_ORDER"),
    ("SLOW_MOVING", "low"): ("Hold", "INTER_REGION_TRANSFER"),
    ("OBSOLETE_RISK", "high"): ("Clearance markdown", "STOP_OR_DELAY_ORDER"),
    ("OBSOLETE_RISK", "low"): ("Manual review", "LIQUIDATION_REVIEW"),
}


def get_matrix_action(inventory_status: str, elasticity: float) -> tuple[str, str]:
    """查决策矩阵。"""
    el_bucket = "high" if abs(elasticity) >= 1.0 else "low" if abs(elasticity) >= 0.5 else "any"
    key = (inventory_status, el_bucket)
    if key in ACTION_MATRIX:
        return ACTION_MATRIX[key]
    key2 = (inventory_status, "any")
    return ACTION_MATRIX.get(key2, ("Hold", "NO_ACTION"))


def determine_inventory_action(
    row: pd.Series,
    transfer_available: bool = False,
    markdown_economics: dict | None = None,
    override: dict | None = None,
) -> tuple[str, str]:
    """
    确定库存行动及原因代码。
    决策顺序：调拨 → 补货 → 停采 → markdown → manual review
    """
    status = row.get("inventory_status", classify_inventory_status(row, override))
    elasticity = abs(row.get("estimated_elasticity", row.get("elasticity", -1.0)))
    on_order = row.get("on_order_inventory", 0)
    avail_woc = row.get("available_weeks_of_cover", 99)
    target_woc = INVENTORY_POLICY.get("target_weeks_of_cover", 8)
    obs_score = row.get("obsolescence_risk_score", 0)
    confidence = row.get("elasticity_confidence", row.get("joint_confidence_score", 0.5))

    # 1. 调拨
    if transfer_available and status in ("OVERSTOCKED", "SLOW_MOVING", "OBSOLETE_RISK"):
        return "INTER_REGION_TRANSFER", "REGIONAL_IMBALANCE_TRANSFER"

    # 2. 补货 / 加急
    if status == "STOCKOUT":
        return "EXPEDITE_ORDER", "STOCKOUT_EXPEDITE"
    if status in ("STOCKOUT_RISK", "UNDERSTOCKED"):
        if on_order >= row.get("reorder_point", 0) * 0.8:
            return "NO_ACTION", "ON_ORDER_SUFFICIENT"
        if row.get("stockout_probability", 0) >= 0.7:
            return "EXPEDITE_ORDER", "HIGH_STOCKOUT_RISK"
        return "REPLENISH", "UNDERSTOCKED_REPLENISH"

    # 3. 停采
    if avail_woc > target_woc * 2 and on_order > 0:
        return "STOP_OR_DELAY_ORDER", "EXCESS_ON_ORDER"
    if status in ("SLOW_MOVING", "OBSOLETE_RISK") and obs_score > 0.5:
        return "STOP_OR_DELAY_ORDER", "OBSOLESCENCE_STOP_ORDER"
    if status == "OVERSTOCKED" and elasticity < 0.5:
        return "STOP_OR_DELAY_ORDER", "INELASTIC_OVERSTOCK"

    # 4. Markdown（仅当经济学为正）
    if status in ("OVERSTOCKED", "SLOW_MOVING") and elasticity >= 0.8:
        md = markdown_economics or {}
        if md.get("markdown_recommended", False):
            return "PRICE_MARKDOWN", "POSITIVE_MARKDOWN_ECONOMICS"
        if status == "OVERSTOCKED" and elasticity >= 1.0:
            return "PRICE_MARKDOWN", "EXCESS_HIGH_ELASTICITY"

    # 5. Obsolete
    if status == "OBSOLETE_RISK":
        if obs_score >= 0.8:
            return "LIQUIDATION_REVIEW", "HIGH_OBSOLESCENCE"
        return "MANUAL_REVIEW", "OBSOLETE_RISK_REVIEW"

    # 6. Healthy
    if status == "HEALTHY":
        return "NO_ACTION", "HEALTHY_INVENTORY"

    if confidence < 0.4:
        return "MANUAL_REVIEW", "LOW_CONFIDENCE_INVENTORY"

    return "HOLD_PRICE", "NO_INVENTORY_ACTION"


def resolve_pricing_inventory_conflict(
    pricing_action: str,
    inventory_action: str,
    inventory_status: str,
    price_change_pct: float,
) -> tuple[str, str, bool]:
    """
    解决定价与库存行动冲突。
    返回：(adjusted_pricing_action, adjusted_price_change_hint, manual_review)
    """
    manual = False
    pa = pricing_action

    # 缺货保护：禁止降价
    if inventory_status in ("STOCKOUT", "STOCKOUT_RISK", "UNDERSTOCKED"):
        if pa == "Decrease" or price_change_pct < -0.005:
            pa = "Hold"
        if inventory_action in ("REPLENISH", "EXPEDITE_ORDER") and pa == "Decrease":
            manual = True

    # 过剩+低弹性：不应 markdown
    if inventory_action in ("STOP_OR_DELAY_ORDER", "INTER_REGION_TRANSFER", "LIQUIDATION_REVIEW"):
        if pa == "Decrease":
            pa = "Hold"

    # 调拨优先于 markdown
    if inventory_action == "INTER_REGION_TRANSFER" and pa == "Decrease":
        pa = "Hold"

    if inventory_action == "MANUAL_REVIEW":
        manual = True

    return pa, inventory_action, manual
