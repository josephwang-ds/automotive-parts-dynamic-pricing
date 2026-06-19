"""回测与回滚测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.backtest import BacktestEngine, RollbackSimulator
from src.data_generator import generate_all_data
from src.demand_model import DemandModelTrainer
from src.elasticity import ElasticityEstimator
from src.optimizer import PriceOptimizer


class TestBacktest:
    @pytest.fixture(scope="class")
    def recommendations(self):
        products, sales = generate_all_data(n_skus=30, n_weeks=20, seed=42)
        trainer = DemandModelTrainer("ridge")
        trainer.train(sales, products)
        elasticity = ElasticityEstimator()
        elasticity.fit(sales, products)
        optimizer = PriceOptimizer(demand_model=trainer, elasticity_estimator=elasticity)
        return optimizer.generate_recommendations(sales, products, elasticity), sales

    def test_backtest_runs(self, recommendations):
        """回测可以运行。"""
        recs, sales = recommendations
        engine = BacktestEngine(recs, sales)
        result = engine.run_backtest()
        assert "strategy_comparison" in result
        assert "modeled_lift" in result

    def test_rollback_100_equals_baseline(self, recommendations):
        """100% 回滚等于 baseline 价格。"""
        recs, _ = recommendations
        sim = RollbackSimulator(recs)
        assert sim.verify_rollback(1.0)

    def test_rollback_0_equals_recommendation(self, recommendations):
        """0% 回滚等于完整推荐。"""
        recs, _ = recommendations
        sim = RollbackSimulator(recs)
        assert sim.verify_rollback(0.0)

    def test_rollback_intermediate(self, recommendations):
        """中间回滚值合理。"""
        recs, _ = recommendations
        sim = RollbackSimulator(recs)
        result = sim.simulate_rollback(0.5)
        audit = result["audit_table"]
        for _, row in audit.head(5).iterrows():
            expected = row["current_price"] * 0.5 + row["recommended_price"] * 0.5
            assert abs(row["rollback_price"] - expected) < 0.02

    def test_modeled_lift_not_proven(self, recommendations):
        """回测结果称为 modeled lift。"""
        recs, sales = recommendations
        engine = BacktestEngine(recs, sales)
        result = engine.run_backtest()
        assert isinstance(result["modeled_lift"], float)
