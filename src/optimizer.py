"""动态定价优化器：候选价格模拟与目标函数优化。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    BALANCED_WEIGHTS,
    CANDIDATE_PRICE_MAX_PCT,
    CANDIDATE_PRICE_MIN_PCT,
    CANDIDATE_PRICE_STEP_PCT,
    EXCESS_INVENTORY_MAX_MOVE_PCT,
    LOW_CONFIDENCE_MAX_MOVE_PCT,
    MAX_PRICE_MOVE_PCT,
    OBJECTIVES,
    POLICY_VERSION,
    WEEKS_OF_COVER_EXCESS,
    WEEKS_OF_COVER_STOCKOUT_RISK,
)
from src.inventory import simulate_inventory_impact
from src.utils import normalize_series, round_price, safe_divide


REASON_CODES = [
    "INELASTIC_MARGIN_OPPORTUNITY",
    "COMPETITOR_PRICE_GAP",
    "EXCESS_INVENTORY_MARKDOWN",
    "LOW_MARGIN_PROTECTION",
    "STOCKOUT_PROTECTION",
    "CUSTOMER_TIER_LEAKAGE",
    "LOW_CONFIDENCE_TEST",
    "MAP_CONSTRAINT",
    "PRICE_LADDER_CONSTRAINT",
    "NO_MATERIAL_OPPORTUNITY",
]


class PriceOptimizer:
    """动态定价优化器。"""

    def __init__(
        self,
        demand_model=None,
        elasticity_estimator=None,
        objective: str = "balanced",
        max_move_pct: float = MAX_PRICE_MOVE_PCT,
        margin_floor: float = 0.15,
        price_rounding: str = "ending_99",
        scenario_config: dict | None = None,
    ):
        self.demand_model = demand_model
        self.elasticity_estimator = elasticity_estimator
        self.objective = objective
        self.max_move_pct = max_move_pct
        self.margin_floor = margin_floor
        self.price_rounding = price_rounding
        self.scenario_config = scenario_config or {}

    def generate_candidate_prices(self, current_price: float) -> list[float]:
        """生成候选价格列表。"""
        prices = []
        pct = CANDIDATE_PRICE_MIN_PCT
        while pct <= CANDIDATE_PRICE_MAX_PCT + 1e-9:
            price = current_price * (1 + pct)
            prices.append(round(price, 2))
            pct += CANDIDATE_PRICE_STEP_PCT
        return prices

    def simulate_candidate(
        self,
        row: pd.Series,
        candidate_price: float,
        elasticity_info: dict,
    ) -> dict:
        """模拟单个候选价格的结果。"""
        current_price = row["realized_price"]
        unit_cost = row["unit_cost"]
        current_inventory = row.get("ending_inventory", row.get("prior_week_inventory", 50))

        # 需求预测：批量弹性路径（快速），ML 模型用于训练评估
        if self.demand_model is not None:
            el = elasticity_info.get("estimated_elasticity", -1.0)
            predicted_units = self.demand_model.predict_at_price(
                row, candidate_price, elasticity=el
            )
        else:
            # 弹性回退
            el = elasticity_info.get("estimated_elasticity", -1.0)
            price_ratio = candidate_price / current_price if current_price > 0 else 1
            base_units = row.get("units_sold", row.get("adjusted_units", 10))
            predicted_units = base_units * (price_ratio ** el)

        predicted_units = max(0, predicted_units)

        revenue = candidate_price * predicted_units
        cogs = unit_cost * predicted_units
        gross_profit = revenue - cogs
        gross_margin = safe_divide(gross_profit, revenue)

        inv_impact = simulate_inventory_impact(current_inventory, predicted_units)

        return {
            "candidate_price": candidate_price,
            "predicted_units": predicted_units,
            "revenue": revenue,
            "gross_profit": gross_profit,
            "gross_margin_pct": gross_margin,
            **inv_impact,
        }

    def _score_candidate(
        self,
        sim: dict,
        current_sim: dict,
        price_change_pct: float,
        confidence: float,
        is_excess: bool,
        is_stockout_risk: bool,
    ) -> float:
        """计算候选价格得分。"""
        obj = self.objective

        if obj == "maximize_gross_profit":
            return sim["gross_profit"]
        if obj == "maximize_revenue":
            return sim["revenue"]
        if obj == "reduce_excess_inventory":
            inv_reduction = current_sim.get("expected_ending_inventory", 0) - sim.get("expected_ending_inventory", 0)
            return inv_reduction * 10 + sim["gross_profit"] * 0.3

        # Balanced
        w = BALANCED_WEIGHTS
        gp_norm = sim["gross_profit"]
        inv_benefit = 0
        if is_excess:
            inv_benefit = max(0, current_sim["expected_ending_inventory"] - sim["expected_ending_inventory"])

        stockout_pen = sim.get("stockout_probability", 0) * 100
        price_change_pen = abs(price_change_pct) * 50
        low_conf_pen = (1 - confidence) * 20 if confidence < 0.4 else 0

        return (
            w["gross_profit"] * gp_norm
            + w["inventory_reduction"] * inv_benefit
            - w["stockout_penalty"] * stockout_pen
            - w["price_change_penalty"] * price_change_pen
            - w["low_confidence_penalty"] * low_conf_pen
        )

    def apply_guardrails(
        self,
        recommended_price: float,
        row: pd.Series,
        product: pd.Series,
        price_change_pct: float,
        confidence: float,
        tier_prices: dict | None = None,
    ) -> tuple[float, list[str], str]:
        """应用定价护栏，返回调整后价格、触发的护栏、原因代码。"""
        guardrails = []
        reason = "NO_MATERIAL_OPPORTUNITY"
        current_price = row["realized_price"]
        unit_cost = row["unit_cost"]

        # 1. Maximum Move
        max_move = self.max_move_pct
        if row.get("excess_inventory_flag", False) or row.get("weeks_of_cover", 0) > WEEKS_OF_COVER_EXCESS:
            max_move = EXCESS_INVENTORY_MAX_MOVE_PCT
        if confidence < 0.4:
            max_move = min(max_move, LOW_CONFIDENCE_MAX_MOVE_PCT)

        actual_change = (recommended_price - current_price) / current_price if current_price > 0 else 0
        if abs(actual_change) > max_move:
            sign = 1 if actual_change > 0 else -1
            recommended_price = current_price * (1 + sign * max_move)
            guardrails.append("MAX_MOVE")
            price_change_pct = sign * max_move

        # 2. Margin Floor
        min_margin = product.get("minimum_margin_pct", self.margin_floor)
        margin_floor_price = unit_cost / (1 - min_margin) if min_margin < 1 else unit_cost * 1.5
        if recommended_price < margin_floor_price:
            recommended_price = margin_floor_price
            guardrails.append("MARGIN_FLOOR")
            reason = "LOW_MARGIN_PROTECTION"

        # 3. MAP
        map_price = product.get("minimum_advertised_price", 0)
        if map_price > 0 and recommended_price < map_price:
            recommended_price = map_price
            guardrails.append("MAP")
            reason = "MAP_CONSTRAINT"

        # 4. Tier Ladder
        if tier_prices:
            tier = row.get("customer_tier", "Retail")
            if tier == "Trade" and "Retail" in tier_prices:
                if recommended_price > tier_prices["Retail"]:
                    recommended_price = tier_prices["Retail"]
                    guardrails.append("TIER_LADDER")
                    reason = "PRICE_LADDER_CONSTRAINT"
            if tier == "Fleet":
                if "Trade" in tier_prices and recommended_price > tier_prices["Trade"]:
                    recommended_price = tier_prices["Trade"]
                    guardrails.append("TIER_LADDER")
                    reason = "PRICE_LADDER_CONSTRAINT"
                if "Retail" in tier_prices and recommended_price > tier_prices["Retail"]:
                    recommended_price = min(recommended_price, tier_prices["Retail"])
                    guardrails.append("TIER_LADDER")

        # 5. Stockout Protection
        woc = row.get("weeks_of_cover", 10)
        if woc < WEEKS_OF_COVER_STOCKOUT_RISK:
            if recommended_price < current_price:
                recommended_price = current_price
                guardrails.append("STOCKOUT_PROTECTION")
                reason = "STOCKOUT_PROTECTION"
            elif recommended_price > current_price * 1.03:
                recommended_price = current_price * 1.03
                guardrails.append("STOCKOUT_PROTECTION")

        # 6. Low Confidence
        if confidence < 0.4 and abs(price_change_pct) > LOW_CONFIDENCE_MAX_MOVE_PCT:
            if price_change_pct > 0:
                recommended_price = current_price * (1 + LOW_CONFIDENCE_MAX_MOVE_PCT)
            else:
                recommended_price = current_price * (1 - LOW_CONFIDENCE_MAX_MOVE_PCT)
            guardrails.append("LOW_CONFIDENCE")
            reason = "LOW_CONFIDENCE_TEST"

        # 7. Price Rounding
        recommended_price = round_price(recommended_price, self.price_rounding)

        # 舍入后重新检查 MAP 和 margin floor
        if recommended_price < margin_floor_price:
            recommended_price = round_price(margin_floor_price, self.price_rounding)
            guardrails.append("MARGIN_FLOOR")
        if map_price > 0 and recommended_price < map_price:
            recommended_price = round_price(map_price, self.price_rounding)
            guardrails.append("MAP")

        # 确定原因代码
        if not guardrails:
            if row.get("excess_inventory_flag"):
                reason = "EXCESS_INVENTORY_MARKDOWN"
            elif row.get("competitor_price_index", 1) > 1.05:
                reason = "COMPETITOR_PRICE_GAP"
            elif confidence > 0.6 and price_change_pct > 0.02:
                reason = "INELASTIC_MARGIN_OPPORTUNITY"

        return recommended_price, guardrails, reason

    def optimize_sku(
        self,
        row: pd.Series,
        product: pd.Series,
        elasticity_info: dict,
        tier_prices: dict | None = None,
    ) -> dict:
        """为单个 SKU 生成定价推荐。"""
        current_price = row["realized_price"]
        candidates = self.generate_candidate_prices(current_price)

        current_sim = self.simulate_candidate(row, current_price, elasticity_info)
        is_excess = row.get("excess_inventory_flag", False) or row.get("weeks_of_cover", 0) > WEEKS_OF_COVER_EXCESS
        is_stockout = row.get("weeks_of_cover", 10) < WEEKS_OF_COVER_STOCKOUT_RISK
        confidence = elasticity_info.get("confidence_score", 0.5)

        best_score = -np.inf
        best_sim = current_sim
        best_price = current_price

        all_sims = []
        el = elasticity_info.get("estimated_elasticity", -1.0)
        if self.demand_model is not None:
            batch_units = self.demand_model.predict_batch_at_prices(row, candidates, el)
        else:
            batch_units = [None] * len(candidates)

        for i, cp in enumerate(candidates):
            sim = self.simulate_candidate(row, cp, elasticity_info)
            if batch_units[i] is not None:
                sim["predicted_units"] = batch_units[i]
                sim["revenue"] = cp * batch_units[i]
                sim["gross_profit"] = sim["revenue"] - row["unit_cost"] * batch_units[i]
                sim["gross_margin_pct"] = safe_divide(sim["gross_profit"], sim["revenue"])
            change_pct = (cp - current_price) / current_price if current_price > 0 else 0
            score = self._score_candidate(
                sim, current_sim, change_pct, confidence, is_excess, is_stockout
            )
            sim["score"] = score
            all_sims.append(sim)
            if score > best_score:
                best_score = score
                best_sim = sim
                best_price = cp

        price_change_pct = (best_price - current_price) / current_price if current_price > 0 else 0

        # 应用护栏
        final_price, guardrails, reason = self.apply_guardrails(
            best_price, row, product, price_change_pct, confidence, tier_prices
        )
        final_change_pct = (final_price - current_price) / current_price if current_price > 0 else 0

        # 最终模拟
        final_sim = self.simulate_candidate(row, final_price, elasticity_info)

        # 推荐动作
        action = self._determine_action(final_change_pct, confidence, guardrails)

        return {
            "sku_id": row["sku_id"],
            "category": row["category"],
            "region": row["region"],
            "customer_tier": row["customer_tier"],
            "current_price": current_price,
            "recommended_price": final_price,
            "price_change_pct": round(final_change_pct, 4),
            "predicted_current_units": current_sim["predicted_units"],
            "predicted_recommended_units": final_sim["predicted_units"],
            "current_revenue": current_sim["revenue"],
            "projected_revenue": final_sim["revenue"],
            "current_gross_profit": current_sim["gross_profit"],
            "projected_gross_profit": final_sim["gross_profit"],
            "gross_profit_lift": final_sim["gross_profit"] - current_sim["gross_profit"],
            "gross_margin_pct": final_sim["gross_margin_pct"],
            "expected_ending_inventory": final_sim["expected_ending_inventory"],
            "expected_weeks_of_cover": final_sim["expected_weeks_of_cover"],
            "elasticity": elasticity_info.get("estimated_elasticity", -1.0),
            "elasticity_confidence": confidence,
            "recommendation_action": action,
            "reason_code": reason,
            "guardrail_triggered": ",".join(guardrails) if guardrails else "",
            "approval_status": "Pending",
            "policy_version": POLICY_VERSION,
            "all_candidate_sims": all_sims,
        }

    def _determine_action(
        self,
        change_pct: float,
        confidence: float,
        guardrails: list,
    ) -> str:
        """确定推荐动作。"""
        if "LOW_CONFIDENCE" in guardrails or confidence < 0.3:
            if abs(change_pct) < 0.01:
                return "Hold"
            return "Test" if abs(change_pct) < 0.05 else "Manual Review"
        if abs(change_pct) < 0.005:
            return "Hold"
        if change_pct > 0.01:
            return "Increase"
        if change_pct < -0.01:
            return "Decrease"
        return "Hold"

    def generate_recommendations(
        self,
        sales: pd.DataFrame,
        products: pd.DataFrame,
        elasticity_estimator,
    ) -> pd.DataFrame:
        """为所有 SKU 生成推荐。"""
        latest = sales.sort_values("week_num").groupby(
            ["sku_id", "region", "customer_tier"]
        ).last().reset_index()

        latest = latest.merge(
            products[[c for c in products.columns if c not in latest.columns or c == "sku_id"]],
            on="sku_id", how="left",
        )

        recs = []
        total = len(latest)
        for idx, (_, row) in enumerate(latest.iterrows()):
            el_info = elasticity_estimator.get_elasticity(
                row["category"], row["region"], row["customer_tier"]
            )

            tier_prices = {}
            same_sku = latest[
                (latest["sku_id"] == row["sku_id"]) & (latest["region"] == row["region"])
            ]
            for _, tr in same_sku.iterrows():
                tier_prices[tr["customer_tier"]] = tr["realized_price"]

            rec = self.optimize_sku(row, row, el_info, tier_prices)
            recs.append({k: v for k, v in rec.items() if k != "all_candidate_sims"})

            if (idx + 1) % 5000 == 0:
                print(f"  推荐进度: {idx+1}/{total}")

        return pd.DataFrame(recs)
