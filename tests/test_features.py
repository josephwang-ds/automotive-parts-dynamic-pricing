"""特征工程测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src.config import TRAIN_WEEKS, VAL_WEEKS, TEST_WEEKS, N_WEEKS
from src.data_generator import generate_all_data
from src.features import build_features, get_time_split_masks, verify_no_future_leakage


class TestFeatures:
    @pytest.fixture(scope="class")
    def feature_df(self):
        products, sales = generate_all_data(n_skus=50, n_weeks=20, seed=42)
        return build_features(sales, products)

    def test_no_future_leakage(self, feature_df):
        """Lag 特征不使用未来数据。"""
        assert verify_no_future_leakage(feature_df) is True

    def test_lag_features_exist(self, feature_df):
        """滞后特征存在。"""
        for lag in [1, 2, 4, 13]:
            assert f"lag_{lag}_units" in feature_df.columns

    def test_rolling_features_exist(self, feature_df):
        """滚动特征存在。"""
        for window in [4, 8, 13]:
            assert f"rolling_{window}_units" in feature_df.columns

    def test_time_split_order(self, feature_df):
        """时间切分顺序正确。"""
        masks = get_time_split_masks(feature_df)
        train = feature_df[masks["train"]]
        val = feature_df[masks["val"]]
        test = feature_df[masks["test"]]

        assert len(train) > 0, "训练集不应为空"
        if len(val) > 0 and len(test) > 0:
            assert train["week_num"].max() <= val["week_num"].min()
            assert val["week_num"].max() <= test["week_num"].min()
        elif len(test) > 0:
            assert train["week_num"].max() < test["week_num"].min()

    def test_adjusted_units_column(self, feature_df):
        """调整销量列存在。"""
        assert "adjusted_units" in feature_df.columns
        assert (feature_df["adjusted_units"] >= 0).all()

    def test_no_true_elasticity_in_features(self, feature_df):
        """true_price_elasticity 不作为特征。"""
        from src.features import get_feature_columns
        cols = get_feature_columns()
        assert "true_price_elasticity" not in cols
