"""定价优化器与护栏测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.config import MAX_PRICE_MOVE_PCT, LOW_CONFIDENCE_MAX_MOVE_PCT
from src.data_generator import generate_all_data
from src.demand_model import DemandModelTrainer
from src.elasticity import ElasticityEstimator
from src.optimizer import PriceOptimizer


class TestOptimizer:
    @pytest.fixture(scope="class")
    def setup(self):
        products, sales = generate_all_data(n_skus=30, n_weeks=20, seed=42)
        trainer = DemandModelTrainer("ridge")
        trainer.train(sales, products)
        elasticity = ElasticityEstimator()
        elasticity.fit(sales, products)
        optimizer = PriceOptimizer(
            demand_model=trainer,
            elasticity_estimator=elasticity,
        )
        latest = sales.sort_values("week_num").groupby(
            ["sku_id", "region", "customer_tier"]
        ).last().reset_index()
        latest = latest.merge(products, on="sku_id", suffixes=("", "_prod"))
        return optimizer, latest, products, elasticity

    def test_margin_floor(self, setup):
        """推荐价格不低于 margin floor。"""
        optimizer, latest, products, elasticity = setup
        row = latest.iloc[0]
        product = row
        el_info = elasticity.get_elasticity(row["category"], row["region"], row["customer_tier"])
        rec = optimizer.optimize_sku(row, product, el_info)
        min_price = row["unit_cost"] / (1 - product["minimum_margin_pct"])
        assert rec["recommended_price"] >= min_price - 0.01

    def test_map_constraint(self, setup):
        """推荐价格不低于 MAP。"""
        optimizer, latest, products, elasticity = setup
        for _, row in latest.head(5).iterrows():
            product = row
            el_info = elasticity.get_elasticity(row["category"], row["region"], row["customer_tier"])
            rec = optimizer.optimize_sku(row, product, el_info)
            map_price = product.get("minimum_advertised_price", 0)
            if map_price > 0:
                assert rec["recommended_price"] >= map_price - 0.05

    def test_price_move_guardrail(self, setup):
        """价格变动不超过护栏。"""
        optimizer, latest, products, elasticity = setup
        row = latest.iloc[0]
        product = row
        el_info = elasticity.get_elasticity(row["category"], row["region"], row["customer_tier"])
        rec = optimizer.optimize_sku(row, product, el_info)
        change = abs(rec["price_change_pct"])
        assert change <= MAX_PRICE_MOVE_PCT + 0.01 or rec["recommendation_action"] in ("Hold", "Manual Review", "Test")

    def test_reason_code_not_empty(self, setup):
        """推荐原因代码非空。"""
        optimizer, latest, products, elasticity = setup
        row = latest.iloc[0]
        el_info = elasticity.get_elasticity(row["category"], row["region"], row["customer_tier"])
        rec = optimizer.optimize_sku(row, row, el_info)
        assert rec["reason_code"] is not None and len(rec["reason_code"]) > 0

    def test_stockout_no_aggressive_decrease(self, setup):
        """缺货风险 SKU 不被激进降价。"""
        optimizer, latest, products, elasticity = setup
        stockout_rows = latest[latest["weeks_of_cover"] < 2]
        if len(stockout_rows) == 0:
            pytest.skip("No stockout rows in test data")
        for _, row in stockout_rows.head(3).iterrows():
            el_info = elasticity.get_elasticity(row["category"], row["region"], row["customer_tier"])
            rec = optimizer.optimize_sku(row, row, el_info)
            if rec["price_change_pct"] < 0:
                assert rec["price_change_pct"] > -0.01

    def test_low_confidence_not_aggressive(self, setup):
        """低置信度推荐不激进。"""
        optimizer, latest, products, elasticity = setup
        el_info = {"estimated_elasticity": -1.0, "confidence_score": 0.1}
        row = latest.iloc[0]
        rec = optimizer.optimize_sku(row, row, el_info)
        assert abs(rec["price_change_pct"]) <= LOW_CONFIDENCE_MAX_MOVE_PCT + 0.01 or rec["recommendation_action"] in ("Hold", "Test", "Manual Review")

    def test_tier_ladder(self, setup):
        """Retail ≥ Trade ≥ Fleet。"""
        optimizer, latest, products, elasticity = setup
        recs = optimizer.generate_recommendations(
            pd.concat([latest] * 1), products, elasticity
        )
        for sku_id in recs["sku_id"].unique()[:5]:
            sku_recs = recs[recs["sku_id"] == sku_id]
            prices = {}
            for _, r in sku_recs.iterrows():
                prices[r["customer_tier"]] = r["recommended_price"]
            if "Retail" in prices and "Trade" in prices:
                assert prices["Retail"] >= prices["Trade"] - 0.01
            if "Trade" in prices and "Fleet" in prices:
                assert prices["Trade"] >= prices["Fleet"] - 0.01
