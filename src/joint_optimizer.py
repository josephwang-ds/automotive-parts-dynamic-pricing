"""定价与库存联合优化器。"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.config import POLICY_VERSION, SCENARIOS
from src.inventory_metrics import build_sku_region_metrics
from src.inventory_policy import (
    classify_inventory_status,
    determine_inventory_action,
    evaluate_markdown_economics,
    resolve_pricing_inventory_conflict,
)
from src.optimizer import PriceOptimizer
from src.replenishment import ReplenishmentEngine
from src.transfer_optimizer import TransferOptimizer


def compute_joint_confidence(
    elasticity_confidence: float,
    demand_history_weeks: int,
    stockout_censoring_pct: float,
    forecast_stability: float = 0.7,
) -> tuple[str, float]:
    """计算联合置信度。"""
    completeness = min(1.0, demand_history_weeks / 13)
    censor_penalty = min(0.3, stockout_censoring_pct * 0.5)
    score = (
        0.35 * elasticity_confidence
        + 0.25 * completeness
        + 0.20 * forecast_stability
        + 0.20 * (1 - censor_penalty)
    )
    score = max(0.0, min(1.0, score))
    if score >= 0.7:
        level = "HIGH"
    elif score >= 0.4:
        level = "MEDIUM"
    else:
        level = "LOW"
    return level, round(score, 3)


class JointOptimizer:
    """联合定价与库存决策优化器。"""

    def __init__(
        self,
        demand_model=None,
        elasticity_estimator=None,
        scenario: str = "Recommended",
    ):
        scenario_cfg = SCENARIOS.get(scenario, SCENARIOS["Recommended"])
        self.scenario_cfg = scenario_cfg
        self.price_optimizer = PriceOptimizer(
            demand_model=demand_model,
            elasticity_estimator=elasticity_estimator,
            objective=scenario_cfg.get("objective", "balanced"),
            max_move_pct=scenario_cfg.get("max_price_move_pct", 0.10),
            scenario_config=scenario_cfg,
        )
        self.replenishment = ReplenishmentEngine(scenario_cfg)
        self.transfer_optimizer = TransferOptimizer(scenario_cfg)

    def generate_joint_recommendations(
        self,
        sales: pd.DataFrame,
        products: pd.DataFrame,
        elasticity_estimator,
    ) -> dict:
        """生成联合推荐：定价 + 库存 + 调拨。"""
        # 定价推荐
        pricing_recs = self.price_optimizer.generate_recommendations(
            sales, products, elasticity_estimator
        )

        # SKU×Region 库存指标
        inv_metrics = build_sku_region_metrics(sales, products, self.scenario_cfg)
        inv_metrics["inventory_status"] = inv_metrics.apply(
            lambda r: classify_inventory_status(r, self.scenario_cfg), axis=1
        )

        # 调拨
        transfers = self.transfer_optimizer.find_transfers(inv_metrics)
        transfer_lookup = set()
        if not transfers.empty:
            for _, t in transfers.iterrows():
                transfer_lookup.add((t["sku_id"], t["source_region"]))
                transfer_lookup.add((t["sku_id"], t["destination_region"]))

        # 补货
        replen_df = self.replenishment.evaluate_all(inv_metrics)
        replen_cols = [c for c in replen_df.columns if c not in inv_metrics.columns]
        inv_metrics = pd.concat([inv_metrics.reset_index(drop=True), replen_df[replen_cols].reset_index(drop=True)], axis=1)

        inv_lookup = inv_metrics.set_index(["sku_id", "region"]).to_dict("index")

        # 历史周数（用于 confidence）
        hist_weeks = sales.groupby(["sku_id", "region"])["week_num"].nunique().to_dict()

        joint_recs = []
        for _, rec in pricing_recs.iterrows():
            key = (rec["sku_id"], rec["region"])
            inv = inv_lookup.get(key, {})
            status = inv.get("inventory_status", "HEALTHY")
            elasticity = rec.get("elasticity", -1.0)
            el_conf = rec.get("elasticity_confidence", 0.5)

            md_econ = evaluate_markdown_economics(
                rec.get("current_gross_profit", 0),
                rec.get("projected_gross_profit", 0),
                inv.get("excess_inventory_value", 0),
                inv.get("obsolescence_risk_score", 0),
                self.scenario_cfg,
            )

            transfer_flag = key in transfer_lookup
            inv_row = pd.Series({**inv, "estimated_elasticity": elasticity, "elasticity_confidence": el_conf})
            inv_action, inv_reason = determine_inventory_action(
                inv_row, transfer_flag, md_econ, self.scenario_cfg
            )

            pricing_action = rec.get("recommendation_action", "Hold")
            adj_pricing, final_inv_action, manual = resolve_pricing_inventory_conflict(
                pricing_action, inv_action, status, rec.get("price_change_pct", 0)
            )

            weeks = hist_weeks.get(key, 8)
            stockout_pct = 0.1 if status in ("STOCKOUT", "STOCKOUT_RISK") else 0.02
            joint_level, joint_score = compute_joint_confidence(el_conf, weeks, stockout_pct)

            if joint_level == "LOW" and abs(rec.get("price_change_pct", 0)) > 0.03:
                adj_pricing = "Manual Review"
                manual = True
            if joint_level == "LOW" and inv_action == "INTER_REGION_TRANSFER":
                final_inv_action = "MANUAL_REVIEW"
                manual = True

            row = rec.to_dict()
            row.update({
                # 定价字段别名
                "pricing_action": adj_pricing,
                "pricing_reason_code": rec.get("reason_code", ""),
                "modeled_gp_lift": rec.get("gross_profit_lift", 0),
                # 库存字段
                "on_hand_inventory": inv.get("on_hand_inventory", 0),
                "on_order_inventory": inv.get("on_order_inventory", 0),
                "average_weekly_demand": inv.get("average_weekly_demand", 0),
                "demand_std": inv.get("demand_std", 0),
                "inventory_turns": inv.get("inventory_turns", 0),
                "on_hand_weeks_of_cover": inv.get("on_hand_weeks_of_cover", 0),
                "available_weeks_of_cover": inv.get("available_weeks_of_cover", 0),
                "sell_through_rate": inv.get("sell_through_rate", 0),
                "stockout_probability": inv.get("stockout_probability", 0),
                "safety_stock": inv.get("safety_stock", 0),
                "reorder_point": inv.get("reorder_point", 0),
                "excess_units": inv.get("excess_units", 0),
                "excess_inventory_value": inv.get("excess_inventory_value", 0),
                "lost_sales_estimate": inv.get("lost_sales_estimate", 0),
                "obsolescence_risk_score": inv.get("obsolescence_risk_score", 0),
                "inventory_status": status,
                "inventory_action": final_inv_action,
                "inventory_reason_code": inv_reason,
                "recommended_order_quantity": inv.get("recommended_order_quantity", 0),
                "transfer_candidate_flag": transfer_flag,
                # 治理
                "pricing_confidence": el_conf,
                "inventory_confidence": joint_score,
                "joint_confidence": joint_level,
                "joint_confidence_score": joint_score,
                "manual_review_required": manual,
                "recommendation_action": adj_pricing,  # 向后兼容
                "reason_code": rec.get("reason_code", inv_reason),
                "generated_at": datetime.now().isoformat(),
            })
            if "all_candidate_sims" in row:
                del row["all_candidate_sims"]
            joint_recs.append(row)

        return {
            "recommendations": pd.DataFrame(joint_recs),
            "inventory_metrics": inv_metrics,
            "transfers": transfers,
            "inv_snapshot": inv_metrics,
        }
