"""合成数据生成器测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.config import RANDOM_SEED
from src.data_generator import generate_all_data, generate_product_master, generate_weekly_sales


class TestDataGenerator:
    def test_reproducibility(self):
        """相同 seed 可复现。"""
        p1, s1 = generate_all_data(n_skus=100, n_weeks=10, seed=RANDOM_SEED)
        p2, s2 = generate_all_data(n_skus=100, n_weeks=10, seed=RANDOM_SEED)
        pd.testing.assert_frame_equal(p1, p2)
        pd.testing.assert_frame_equal(s1, s2)

    def test_all_elasticity_negative(self):
        """所有 true elasticity 为负。"""
        products, _ = generate_all_data(n_skus=200, n_weeks=5, seed=42)
        assert (products["true_price_elasticity"] < 0).all()

    def test_observed_sales_lte_inventory(self):
        """观测销量不超过可用库存。"""
        products = generate_product_master(n_skus=50, seed=42)
        sales = generate_weekly_sales(products, n_weeks=10, seed=42)
        available = sales["beginning_inventory"] + sales["receipts"]
        assert (sales["units_sold"] <= available + 0.02).all()

    def test_revenue_formula(self):
        """Revenue = realized_price × units_sold。"""
        products = generate_product_master(n_skus=50, seed=42)
        sales = generate_weekly_sales(products, n_weeks=10, seed=42)
        expected = (sales["realized_price"] * sales["units_sold"]).round(2)
        np.testing.assert_allclose(sales["revenue"], expected, rtol=0.05, atol=0.02)

    def test_gross_profit_formula(self):
        """Gross profit = revenue - COGS。"""
        products = generate_product_master(n_skus=50, seed=42)
        sales = generate_weekly_sales(products, n_weeks=10, seed=42)
        expected = sales["revenue"] - sales["cogs"]
        np.testing.assert_allclose(sales["gross_profit"], expected, rtol=0.01)

    def test_sku_count_in_range(self):
        """SKU 数量在合理范围。"""
        products, _ = generate_all_data(n_skus=3000, n_weeks=5, seed=42)
        assert 2000 <= len(products) <= 5000 or len(products) == 3000

    def test_required_columns(self):
        """销售数据包含必要字段。"""
        products = generate_product_master(n_skus=10, seed=42)
        sales = generate_weekly_sales(products, n_weeks=5, seed=42)
        required = [
            "week_start", "sku_id", "category", "region", "customer_tier",
            "unit_cost", "realized_price", "units_sold", "revenue",
            "gross_profit", "stockout_flag", "price_change_reason",
            "price_test_flag", "policy_version",
        ]
        for col in required:
            assert col in sales.columns, f"Missing column: {col}"
