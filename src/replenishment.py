"""补货引擎。"""

from __future__ import annotations

import pandas as pd

from src.config import INVENTORY_POLICY
from src.inventory_metrics import compute_reorder_point, compute_safety_stock, round_to_case_pack


class ReplenishmentEngine:
    """补货决策引擎。"""

    def __init__(self, override: dict | None = None):
        self.override = override or {}

    def evaluate(self, inv_row: pd.Series) -> dict:
        """评估单个 SKU×Region 补货需求。"""
        on_hand = inv_row.get("on_hand_inventory", 0)
        on_order = inv_row.get("on_order_inventory", 0)
        avg_demand = inv_row.get("average_weekly_demand", 0)
        demand_std = inv_row.get("demand_std", avg_demand * 0.2)
        lead_time_weeks = inv_row.get("lead_time_weeks", 2)
        case_pack = int(inv_row.get("case_pack", 1))
        moq = int(inv_row.get("minimum_order_quantity", INVENTORY_POLICY["minimum_order_quantity"]))
        status = inv_row.get("inventory_status", "HEALTHY")
        rop = inv_row.get("reorder_point", 0)
        safety = inv_row.get("safety_stock", 0)

        if rop <= 0:
            safety = compute_safety_stock(demand_std, lead_time_weeks, self.override)
            rop = compute_reorder_point(avg_demand, lead_time_weeks, safety)

        net_req = max(0.0, rop - on_hand - on_order)
        order_qty = round_to_case_pack(net_req, case_pack)
        if order_qty > 0 and order_qty < moq:
            order_qty = float(moq)

        action = "NO_ACTION"
        reason = "SUFFICIENT_INVENTORY"

        if status == "STOCKOUT":
            action = "EXPEDITE_ORDER"
            reason = "STOCKOUT_EXPEDITE"
            order_qty = max(order_qty, round_to_case_pack(avg_demand * 4, case_pack))
        elif status in ("STOCKOUT_RISK", "UNDERSTOCKED"):
            if on_order >= rop * 0.9:
                action = "NO_ACTION"
                reason = "ON_ORDER_COVERS_REQUIREMENT"
                order_qty = 0.0
            elif inv_row.get("stockout_probability", 0) >= 0.65:
                action = "EXPEDITE_ORDER"
                reason = "IMMINENT_STOCKOUT"
            else:
                action = "REPLENISH"
                reason = "BELOW_REORDER_POINT"
        elif status in ("OVERSTOCKED", "SLOW_MOVING", "OBSOLETE_RISK"):
            if on_order > 0 and on_hand + on_order > rop * 2:
                action = "STOP_OR_DELAY_ORDER"
                reason = "EXCESS_ON_ORDER"
                order_qty = 0.0

        weeks_to_stockout = (
            on_hand / avg_demand if avg_demand > 0 else 999
        )

        return {
            "reorder_point": rop,
            "safety_stock": safety,
            "net_requirement": net_req,
            "recommended_order_quantity": order_qty,
            "expected_stockout_weeks": round(weeks_to_stockout, 2),
            "replenishment_action": action,
            "replenishment_reason": reason,
        }

    def evaluate_all(self, inv_metrics: pd.DataFrame) -> pd.DataFrame:
        """批量评估。"""
        results = [self.evaluate(row) for _, row in inv_metrics.iterrows()]
        return pd.DataFrame(results)
