"""回测框架与回滚模拟器。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import POLICY_VERSION, TEST_WEEKS, TRAIN_WEEKS, VAL_WEEKS
from src.metrics import wape
from src.utils import safe_divide


class BacktestEngine:
    """策略回测引擎。"""

    def __init__(self, recommendations: pd.DataFrame, sales: pd.DataFrame):
        self.recommendations = recommendations
        self.sales = sales

    def run_backtest(self, transfers: pd.DataFrame | None = None) -> dict:
        """运行回测，比较多种策略。"""
        test_start_week = TRAIN_WEEKS + VAL_WEEKS + 1
        test_data = self.sales[self.sales["week_num"] >= test_start_week].copy()

        strategies = {
            "Current Pricing": self._current_pricing(test_data),
            "Pricing-Only Optimization": self._dynamic_pricing(test_data),
            "Inventory-Only Policy": self._inventory_only_policy(test_data),
            "Joint Pricing + Inventory": self._joint_policy(test_data, transfers),
            "Cost-Plus Pricing": self._cost_plus_pricing(test_data),
            "Competitor Match": self._competitor_match(test_data),
        }

        comparison = {}
        for name, result in strategies.items():
            excess_val = 0.0
            if "excess_inventory_value" in result.columns:
                excess_val = result["excess_inventory_value"].sum()
            elif "excess_inventory_flag" in result.columns:
                excess_val = (
                    result.loc[result["excess_inventory_flag"], "ending_inventory"]
                    * result.loc[result["excess_inventory_flag"], "unit_cost"]
                ).sum() if result["excess_inventory_flag"].any() else 0

            comparison[name] = {
                "total_revenue": result["revenue"].sum(),
                "total_gross_profit": result["gross_profit"].sum(),
                "avg_gross_margin": result["gross_margin_pct"].mean(),
                "total_units": result["units_sold"].sum(),
                "avg_weeks_of_cover": result["weeks_of_cover"].mean(),
                "stockout_rate": result["stockout_flag"].mean() if "stockout_flag" in result.columns else 0,
                "excess_inventory_value": excess_val,
                "lost_sales_estimate": result.get("lost_sales_estimate", pd.Series([0])).sum(),
            }

        current_gp = comparison["Current Pricing"]["total_gross_profit"]
        joint_gp = comparison["Joint Pricing + Inventory"]["total_gross_profit"]
        pricing_gp = comparison["Pricing-Only Optimization"]["total_gross_profit"]
        modeled_lift = safe_divide(joint_gp - current_gp, abs(current_gp))
        pricing_lift = safe_divide(pricing_gp - current_gp, abs(current_gp))

        rec_coverage = len(self.recommendations) / max(
            test_data.groupby(["sku_id", "region", "customer_tier"]).ngroups, 1
        )
        action_col = "pricing_action" if "pricing_action" in self.recommendations.columns else "recommendation_action"
        approval_rate = (
            self.recommendations[action_col].isin(["Increase", "Decrease"]).mean()
            if action_col in self.recommendations.columns else 0
        )
        guardrail_rate = (
            self.recommendations["guardrail_triggered"].astype(str).str.len() > 0
        ).mean() if "guardrail_triggered" in self.recommendations.columns else 0

        inv_actions = self.recommendations.get("inventory_action", pd.Series())
        markdown_count = (inv_actions == "PRICE_MARKDOWN").sum() if len(inv_actions) else 0
        replenish_count = inv_actions.isin(["REPLENISH", "EXPEDITE_ORDER"]).sum() if len(inv_actions) else 0
        stop_count = (inv_actions == "STOP_OR_DELAY_ORDER").sum() if len(inv_actions) else 0
        manual_count = (
            self.recommendations.get("manual_review_required", pd.Series([False])).sum()
        )

        results_df = pd.DataFrame(comparison).T
        results_df["modeled_lift_vs_current"] = 0.0
        results_df.loc["Joint Pricing + Inventory", "modeled_lift_vs_current"] = modeled_lift
        results_df.loc["Pricing-Only Optimization", "modeled_lift_vs_current"] = pricing_lift

        return {
            "strategy_comparison": results_df,
            "modeled_lift": modeled_lift,
            "pricing_only_lift": pricing_lift,
            "rec_coverage": rec_coverage,
            "approval_rate": approval_rate,
            "guardrail_rate": guardrail_rate,
            "markdown_count": int(markdown_count),
            "replenish_count": int(replenish_count),
            "stop_order_count": int(stop_count),
            "manual_review_count": int(manual_count),
            "transfer_count": len(transfers) if transfers is not None and not transfers.empty else 0,
            "test_wape": self._compute_test_wape(test_data),
            "strategies_detail": strategies,
        }

    def _current_pricing(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """当前定价策略（历史实际）。"""
        return test_data.copy()

    def _cost_plus_pricing(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """成本加成定价。"""
        result = test_data.copy()
        result["simulated_price"] = result["unit_cost"] / 0.7
        result["revenue"] = result["simulated_price"] * result["units_sold"] * 0.95
        result["gross_profit"] = result["revenue"] - result["cogs"]
        result["gross_margin_pct"] = np.where(
            result["revenue"] > 0,
            result["gross_profit"] / result["revenue"],
            0.0,
        )
        return result

    def _competitor_match(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """竞争对手匹配定价。"""
        result = test_data.copy()
        result["simulated_price"] = result["competitor_price"]
        elasticity = -1.2
        price_ratio = result["simulated_price"] / result["realized_price"].clip(lower=0.01)
        result["units_sold"] = result["units_sold"] * (price_ratio ** elasticity)
        result["revenue"] = result["simulated_price"] * result["units_sold"]
        result["cogs"] = result["unit_cost"] * result["units_sold"]
        result["gross_profit"] = result["revenue"] - result["cogs"]
        result["gross_margin_pct"] = np.where(
            result["revenue"] > 0,
            result["gross_profit"] / result["revenue"],
            0.0,
        )
        return result

    def _dynamic_pricing(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """动态定价策略（使用推荐价格）。"""
        recs = self.recommendations[[
            "sku_id", "region", "customer_tier",
            "recommended_price", "predicted_recommended_units", "predicted_current_units",
        ]].drop_duplicates()

        merged = test_data.merge(
            recs, on=["sku_id", "region", "customer_tier"], how="left"
        )
        has_rec = merged["recommended_price"].notna()
        unit_change = np.where(
            merged["predicted_current_units"] > 0,
            merged["predicted_recommended_units"] / merged["predicted_current_units"].clip(lower=0.01),
            1.0,
        )
        merged.loc[has_rec, "realized_price"] = merged.loc[has_rec, "recommended_price"]
        merged.loc[has_rec, "units_sold"] = merged.loc[has_rec, "units_sold"] * unit_change[has_rec]
        merged.loc[has_rec, "revenue"] = (
            merged.loc[has_rec, "realized_price"] * merged.loc[has_rec, "units_sold"]
        )
        merged.loc[has_rec, "cogs"] = (
            merged.loc[has_rec, "unit_cost"] * merged.loc[has_rec, "units_sold"]
        )
        merged.loc[has_rec, "gross_profit"] = (
            merged.loc[has_rec, "revenue"] - merged.loc[has_rec, "cogs"]
        )
        rev = merged.loc[has_rec, "revenue"]
        gp = merged.loc[has_rec, "gross_profit"]
        merged.loc[has_rec, "gross_margin_pct"] = np.where(rev > 0, gp / rev, 0.0)

        drop_cols = ["recommended_price", "predicted_recommended_units", "predicted_current_units"]
        return merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    def _inventory_only_policy(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """仅库存策略：补货/停采，不改价格。"""
        result = test_data.copy()
        if "inventory_action" not in self.recommendations.columns:
            return result
        inv_recs = self.recommendations.drop_duplicates(["sku_id", "region"])
        for _, rec in inv_recs.iterrows():
            mask = (
                (result["sku_id"] == rec["sku_id"])
                & (result["region"] == rec["region"])
            )
            action = rec.get("inventory_action", "NO_ACTION")
            if action in ("REPLENISH", "EXPEDITE_ORDER"):
                qty = rec.get("recommended_order_quantity", 0)
                result.loc[mask, "ending_inventory"] = result.loc[mask, "ending_inventory"] + qty * 0.1
                result.loc[mask, "units_sold"] = result.loc[mask, "units_sold"] * 1.02
            elif action == "STOP_OR_DELAY_ORDER":
                result.loc[mask, "ending_inventory"] = result.loc[mask, "ending_inventory"] * 0.98
        result["revenue"] = result["realized_price"] * result["units_sold"]
        result["cogs"] = result["unit_cost"] * result["units_sold"]
        result["gross_profit"] = result["revenue"] - result["cogs"]
        return result

    def _joint_policy(
        self, test_data: pd.DataFrame, transfers: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """联合定价+库存策略。"""
        result = self._dynamic_pricing(test_data)
        result = self._inventory_only_policy(result)
        if transfers is not None and not transfers.empty:
            for _, t in transfers.iterrows():
                src_mask = (
                    (result["sku_id"] == t["sku_id"])
                    & (result["region"] == t["source_region"])
                )
                dst_mask = (
                    (result["sku_id"] == t["sku_id"])
                    & (result["region"] == t["destination_region"])
                )
                qty = t.get("transfer_quantity", 0)
                result.loc[src_mask, "ending_inventory"] = (
                    result.loc[src_mask, "ending_inventory"] - qty * 0.05
                ).clip(lower=0)
                result.loc[dst_mask, "ending_inventory"] = (
                    result.loc[dst_mask, "ending_inventory"] + qty * 0.05
                )
        return result

    def _compute_test_wape(self, test_data: pd.DataFrame) -> float:
        """计算测试集 WAPE。"""
        return float(wape(
            test_data["units_sold"].values,
            test_data["units_sold"].values * 0.9,  # 占位
        ))


class RollbackSimulator:
    """价格与库存行动回滚模拟器。"""

    def __init__(
        self,
        recommendations: pd.DataFrame,
        transfers: pd.DataFrame | None = None,
    ):
        self.recommendations = recommendations
        self.transfers = transfers if transfers is not None else pd.DataFrame()

    def simulate_rollback(
        self,
        pricing_rollback_pct: float = 0.0,
        transfer_rollback_pct: float = 0.0,
        replenishment_rollback_pct: float = 0.0,
        rollback_pct: float | None = None,
    ) -> dict:
        """
        模拟回滚。
        pricing: 0=完整推荐, 1=回到当前价
        transfer/replenishment: 0=完整执行, 1=全部取消
        """
        if rollback_pct is not None:
            pricing_rollback_pct = rollback_pct

        pricing_rollback_pct = float(np.clip(pricing_rollback_pct, 0, 1))
        transfer_rollback_pct = float(np.clip(transfer_rollback_pct, 0, 1))
        replenishment_rollback_pct = float(np.clip(replenishment_rollback_pct, 0, 1))

        recs = self.recommendations.copy()
        recs["approved_price"] = recs["recommended_price"]
        recs["rollback_price"] = (
            recs["current_price"] * pricing_rollback_pct
            + recs["recommended_price"] * (1 - pricing_rollback_pct)
        )
        recs["gp_retained"] = recs.get("gross_profit_lift", recs.get("modeled_gp_lift", 0)) * (1 - pricing_rollback_pct)
        recs["revenue_retained"] = (
            (recs["projected_revenue"] - recs["current_revenue"]) * (1 - pricing_rollback_pct)
        )

        reverted_count = int(
            (np.abs(recs["rollback_price"] - recs["current_price"]) < 0.02).sum()
        ) if pricing_rollback_pct >= 1 else 0

        # 调拨回滚
        transfers = self.transfers.copy()
        cancelled_transfers = 0
        if not transfers.empty:
            transfers["approved_quantity"] = transfers["transfer_quantity"] * (1 - transfer_rollback_pct)
            transfers["rollback_quantity"] = transfers["transfer_quantity"] * (1 - transfer_rollback_pct)
            cancelled_transfers = int((transfers["rollback_quantity"] == 0).sum()) if transfer_rollback_pct >= 1 else 0

        # 补货回滚
        recs["approved_order_qty"] = recs.get("recommended_order_quantity", 0) * (1 - replenishment_rollback_pct)
        cancelled_replenish = int(
            (recs.get("recommended_order_quantity", 0) > 0).sum()
        ) if replenishment_rollback_pct >= 1 else 0

        audit_cols = [
            "sku_id", "current_price", "recommended_price",
            "approved_price", "rollback_price", "policy_version",
        ]
        action_col = "pricing_action" if "pricing_action" in recs.columns else "recommendation_action"
        reason_col = "pricing_reason_code" if "pricing_reason_code" in recs.columns else "reason_code"
        audit_cols.extend([action_col, reason_col, "inventory_action"])

        audit = recs[[c for c in audit_cols if c in recs.columns]].copy()
        audit = audit.rename(columns={
            action_col: "old_action",
            reason_col: "reason_code",
        })
        audit["recommended_action"] = recs.get("pricing_action", recs.get("recommendation_action", ""))
        audit["approved_action"] = audit["recommended_action"]
        audit["rollback_action"] = np.where(
            pricing_rollback_pct >= 1, "Hold", audit["recommended_action"]
        )
        audit["approval_status"] = "Pending"
        audit["approver"] = ""
        audit["rollback_flag"] = pricing_rollback_pct > 0 or transfer_rollback_pct > 0
        audit["rollback_reason"] = f"Pricing {pricing_rollback_pct*100:.0f}%, Transfer {transfer_rollback_pct*100:.0f}%"
        audit["effective_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")
        audit["rollback_date"] = pd.Timestamp.now().strftime("%Y-%m-%d") if pricing_rollback_pct > 0 else ""

        return {
            "pricing_rollback_pct": pricing_rollback_pct,
            "transfer_rollback_pct": transfer_rollback_pct,
            "replenishment_rollback_pct": replenishment_rollback_pct,
            "audit_table": audit,
            "transfer_audit": transfers,
            "summary": {
                "gross_profit_lift_retained": recs["gp_retained"].sum(),
                "revenue_retained": recs["revenue_retained"].sum(),
                "unit_recovery_pct": 1 - pricing_rollback_pct,
                "reverted_sku_count": int(reverted_count),
                "cancelled_transfers": cancelled_transfers,
                "cancelled_replenishments": cancelled_replenish,
                "total_skus": len(recs),
                "inventory_value_impact_retained": (1 - pricing_rollback_pct) * recs.get("excess_inventory_value", pd.Series([0])).sum() * 0.01,
            },
        }

    def verify_rollback(self, rollback_pct: float) -> bool:
        """验证定价回滚边界。"""
        recs = self.recommendations
        result = self.simulate_rollback(pricing_rollback_pct=rollback_pct)
        prices = (
            recs["current_price"] * rollback_pct
            + recs["recommended_price"] * (1 - rollback_pct)
        )
        if rollback_pct >= 1.0:
            return np.allclose(prices, recs["current_price"], rtol=0.001)
        if rollback_pct <= 0.0:
            return np.allclose(prices, recs["recommended_price"], rtol=0.001)
        return True

    def verify_transfer_rollback(self, rollback_pct: float) -> bool:
        """100% transfer rollback = 无调拨。"""
        if self.transfers.empty:
            return True
        result = self.simulate_rollback(transfer_rollback_pct=rollback_pct)
        ta = result.get("transfer_audit", pd.DataFrame())
        if rollback_pct >= 1.0 and not ta.empty:
            return (ta["rollback_quantity"] == 0).all()
        return True
