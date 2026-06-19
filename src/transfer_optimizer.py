"""地区间调拨优化器。"""

from __future__ import annotations

import pandas as pd

from src.config import INVENTORY_POLICY, REGIONS
from src.inventory_metrics import round_to_case_pack
from src.inventory_policy import classify_inventory_status


class TransferOptimizer:
    """跨地区库存调拨优化。"""

    def __init__(self, override: dict | None = None):
        self.override = {**INVENTORY_POLICY, **(override or {})}

    def find_transfers(self, inv_metrics: pd.DataFrame) -> pd.DataFrame:
        """识别调拨机会，greedy 分配且不重复消耗 source 库存。"""
        if inv_metrics.empty:
            return pd.DataFrame()

        work = inv_metrics.copy()
        work["inventory_status"] = work.apply(
            lambda r: classify_inventory_status(r, self.override), axis=1
        )

        source_statuses = {"OVERSTOCKED", "SLOW_MOVING", "OBSOLETE_RISK"}
        dest_statuses = {"STOCKOUT_RISK", "UNDERSTOCKED", "STOCKOUT"}

        transfers = []
        source_remaining: dict[tuple, float] = {}

        for sku_id in work["sku_id"].unique():
            sku_data = work[work["sku_id"] == sku_id]
            if len(sku_data) < 2:
                continue

            sources = sku_data[sku_data["inventory_status"].isin(source_statuses)]
            dests = sku_data[sku_data["inventory_status"].isin(dest_statuses)]

            for _, src in sources.iterrows():
                src_key = (sku_id, src["region"])
                safety = src.get("safety_stock", 0)
                excess = max(0, src["on_hand_inventory"] - safety - src["average_weekly_demand"] * 2)
                if excess <= 0:
                    continue
                source_remaining[src_key] = source_remaining.get(src_key, excess)

                for _, dst in dests.iterrows():
                    if src["region"] == dst["region"]:
                        continue

                    avail = source_remaining.get(src_key, 0)
                    if avail <= 0:
                        break

                    shortage = max(0, dst.get("reorder_point", 0) - dst["on_hand_inventory"] - dst.get("on_order_inventory", 0))
                    if shortage <= 0:
                        shortage = dst["average_weekly_demand"] * 2

                    case_pack = int(src.get("case_pack", 1))
                    qty = min(avail, shortage, self.override["maximum_transfer_units"])
                    qty = round_to_case_pack(min(qty, avail), case_pack)
                    if qty <= 0:
                        continue

                    cost_per = self.override["transfer_cost_per_unit"]
                    total_cost = qty * cost_per
                    unit_price = src.get("unit_cost", 10) * 1.5  # 近似售价
                    margin = unit_price - src.get("unit_cost", 10)
                    gp_recovered = qty * margin * 0.7  # 调拨后预计售出比例
                    rev_recovered = qty * unit_price * 0.7
                    net_value = gp_recovered - total_cost

                    if net_value <= self.override["minimum_transfer_value"]:
                        continue
                    if gp_recovered <= total_cost:
                        continue

                    src_woc_after = (src["on_hand_inventory"] - qty) / max(src["average_weekly_demand"], 0.01)
                    dst_woc_after = (dst["on_hand_inventory"] + qty) / max(dst["average_weekly_demand"], 0.01)

                    transfers.append({
                        "sku_id": sku_id,
                        "category": src.get("category", ""),
                        "source_region": src["region"],
                        "destination_region": dst["region"],
                        "transfer_quantity": qty,
                        "transfer_cost": round(total_cost, 2),
                        "transfer_cost_per_unit": cost_per,
                        "expected_revenue_recovered": round(rev_recovered, 2),
                        "expected_gp_recovered": round(gp_recovered, 2),
                        "net_transfer_value": round(net_value, 2),
                        "source_status_before": src["inventory_status"],
                        "source_status_after": classify_inventory_status(
                            {**src, "on_hand_inventory": src["on_hand_inventory"] - qty,
                             "available_weeks_of_cover": src_woc_after}, self.override
                        ),
                        "destination_status_before": dst["inventory_status"],
                        "destination_status_after": classify_inventory_status(
                            {**dst, "on_hand_inventory": dst["on_hand_inventory"] + qty,
                             "available_weeks_of_cover": dst_woc_after}, self.override
                        ),
                        "transfer_reason": "REGIONAL_IMBALANCE",
                    })

                    source_remaining[src_key] -= qty

        return pd.DataFrame(transfers)
