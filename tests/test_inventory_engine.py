"""Inventory Decision Engine 测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src.config import INVENTORY_POLICY
from src.data_generator import generate_all_data
from src.demand_model import DemandModelTrainer
from src.elasticity import ElasticityEstimator
from src.inventory_metrics import (
    build_sku_region_metrics,
    compute_excess_units,
    compute_inventory_turns,
    compute_reorder_point,
    compute_safety_stock,
    compute_weeks_of_cover,
    round_to_case_pack,
)
from src.inventory_policy import (
    classify_inventory_status,
    determine_inventory_action,
    evaluate_markdown_economics,
    resolve_pricing_inventory_conflict,
)
from src.joint_optimizer import JointOptimizer, compute_joint_confidence
from src.replenishment import ReplenishmentEngine
from src.transfer_optimizer import TransferOptimizer
from src.backtest import RollbackSimulator


@pytest.fixture(scope="module")
def bundle():
    products, sales = generate_all_data(n_skus=40, n_weeks=30, seed=42)
    trainer = DemandModelTrainer("ridge")
    trainer.train(sales, products)
    elasticity = ElasticityEstimator()
    elasticity.fit(sales, products)
    joint = JointOptimizer(trainer, elasticity, "Recommended")
    result = joint.generate_joint_recommendations(sales, products, elasticity)
    inv = build_sku_region_metrics(sales, products)
    return {
        "products": products, "sales": sales, "inv": inv,
        "recs": result["recommendations"], "transfers": result["transfers"],
        "elasticity": elasticity,
    }


class TestInventoryMetrics:
    def test_weeks_of_cover(self):
        assert abs(compute_weeks_of_cover(100, 10) - 10.0) < 0.01

    def test_inventory_turns_zero_denom(self):
        assert compute_inventory_turns(1000, 0) == 0.0

    def test_safety_stock_non_negative(self):
        assert compute_safety_stock(5, 2) >= 0

    def test_reorder_point(self):
        assert compute_reorder_point(10, 2, 5) == 25.0

    def test_excess_units_non_negative(self):
        assert compute_excess_units(100, 5) >= 0

    def test_case_pack_rounding(self):
        assert round_to_case_pack(7, 6) == 12


class TestReplenishment:
    def test_replenishment_qty_non_negative(self, bundle):
        engine = ReplenishmentEngine()
        for _, row in bundle["inv"].head(5).iterrows():
            assert engine.evaluate(row)["recommended_order_quantity"] >= 0


class TestTransferOptimizer:
    def test_transfer_net_value_positive(self, bundle):
        transfers = TransferOptimizer().find_transfers(bundle["inv"])
        if not transfers.empty:
            assert (transfers["net_transfer_value"] > 0).all()


class TestInventoryPolicy:
    def test_stockout_no_markdown(self):
        row = pd.Series({
            "on_hand_inventory": 0, "available_weeks_of_cover": 0,
            "stockout_probability": 0.9, "lead_time_weeks": 2,
            "on_order_inventory": 0, "average_weekly_demand": 10,
        })
        row["inventory_status"] = classify_inventory_status(row)
        action, _ = determine_inventory_action(row)
        assert action != "PRICE_MARKDOWN"

    def test_inventory_reason_non_empty(self, bundle):
        for _, r in bundle["recs"].head(10).iterrows():
            assert r.get("inventory_reason_code", "") != ""

    def test_pricing_inventory_conflict(self):
        pa, _, _ = resolve_pricing_inventory_conflict("Decrease", "REPLENISH", "STOCKOUT_RISK", -0.05)
        assert pa == "Hold"


class TestJointOptimizer:
    def test_joint_fields(self, bundle):
        for col in ["inventory_status", "inventory_action", "joint_confidence"]:
            assert col in bundle["recs"].columns

    def test_low_confidence(self):
        level, _ = compute_joint_confidence(0.1, 3, 0.5)
        assert level == "LOW"


class TestRollbackExtended:
    def test_pricing_rollback_bounds(self, bundle):
        sim = RollbackSimulator(bundle["recs"], bundle["transfers"])
        assert sim.verify_rollback(1.0)
        assert sim.verify_rollback(0.0)

    def test_transfer_rollback_100(self, bundle):
        sim = RollbackSimulator(bundle["recs"], bundle["transfers"])
        assert sim.verify_transfer_rollback(1.0)


class TestMarkdownEconomics:
    def test_negative_markdown(self):
        result = evaluate_markdown_economics(100, 80, 1000, 0.5)
        assert isinstance(result["net_markdown_value"], float)
