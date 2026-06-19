"""Streamlit 部署用轻量运行时对象（不依赖 scikit-learn）。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import DATA_DIR, MODELS_DIR, OUTPUTS_DIR
from src.inventory import inventory_analysis_from_metrics


class DeployDemandModel:
    """仅含 metadata 的需求模型占位，定价模拟走弹性快速路径。"""

    def __init__(self, metadata: dict):
        self.metrics = metadata.get("metrics", {})
        self.name = metadata.get("model_name", "HistGradientBoosting")
        self.model_type = metadata.get("model_type", "hgb")

    def predict_at_price(
        self,
        row: pd.Series,
        candidate_price: float,
        feature_template: pd.DataFrame | None = None,
        elasticity: float = -1.0,
    ) -> float:
        current_price = row.get("realized_price", candidate_price)
        base_units = row.get("units_sold", row.get("adjusted_units", 10))
        if current_price <= 0:
            return max(0.0, float(base_units))
        price_ratio = candidate_price / current_price
        return max(0.0, float(base_units * (price_ratio ** elasticity)))

    def predict_batch_at_prices(
        self,
        row: pd.Series,
        candidate_prices: list[float],
        elasticity: float = -1.0,
    ) -> list[float]:
        current_price = row.get("realized_price", 1)
        base_units = row.get("units_sold", row.get("adjusted_units", 10))
        if current_price <= 0:
            return [base_units] * len(candidate_prices)
        ratios = np.array(candidate_prices) / current_price
        return [max(0.0, float(base_units * (r ** elasticity))) for r in ratios]


class DeployElasticity:
    """从预计算 CSV 提供弹性查询，无需 sklearn。"""

    def __init__(self, estimates: pd.DataFrame, global_elasticity: float = -1.0):
        self.estimates = estimates
        self.global_elasticity = global_elasticity

    def get_elasticity(self, category: str, region: str, tier: str) -> dict:
        match = self.estimates[
            (self.estimates["category"] == category)
            & (self.estimates["region"] == region)
            & (self.estimates["customer_tier"] == tier)
            & (self.estimates.get("estimation_level", "") == "category_region_tier")
        ]
        if len(match) > 0:
            return match.iloc[0].to_dict()

        match = self.estimates[
            (self.estimates["category"] == category)
            & (self.estimates["region"] == "All")
            & (self.estimates["customer_tier"] == tier)
        ]
        if len(match) > 0:
            return match.iloc[0].to_dict()

        return {
            "estimated_elasticity": self.global_elasticity,
            "confidence_score": 0.3,
            "sample_size": 0,
            "price_variation": 0,
            "elasticity_class": "Low confidence",
            "estimation_level": "global_prior",
        }


class AppState:
    """与 PipelineState 字段兼容的应用状态容器。"""

    def __init__(self):
        self.products = pd.DataFrame()
        self.sales = pd.DataFrame()
        self.features = pd.DataFrame()
        self.demand_model = None
        self.model_results: dict = {}
        self.elasticity = None
        self.elasticity_df = pd.DataFrame()
        self.recommendations = pd.DataFrame()
        self.transfers = pd.DataFrame()
        self.inventory_metrics = pd.DataFrame()
        self.backtest_results: dict = {}
        self.inventory_analysis: dict = {}


def deploy_artifacts_ready() -> bool:
    return (
        (OUTPUTS_DIR / "recommendations.csv").exists()
        and (MODELS_DIR / "demand_model_metadata.json").exists()
    )


def load_deploy_state() -> AppState:
    """加载预计算部署产物（无 sklearn / 无训练流水线）。"""
    state = AppState()

    state.products = pd.read_csv(DATA_DIR / "synthetic_products.csv")

    sales_path = DATA_DIR / "synthetic_sales_latest.csv"
    if not sales_path.exists():
        sales_path = DATA_DIR / "synthetic_sales_app.csv"
    state.sales = pd.read_csv(sales_path, parse_dates=["week_start"])

    meta_path = MODELS_DIR / "demand_model_metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    state.demand_model = DeployDemandModel(metadata)
    state.model_results = {"hgb": {"metrics": state.demand_model.metrics}}

    state.recommendations = pd.read_csv(OUTPUTS_DIR / "recommendations.csv")
    tr_path = OUTPUTS_DIR / "transfer_recommendations.csv"
    inv_path = OUTPUTS_DIR / "inventory_metrics.csv"
    el_path = OUTPUTS_DIR / "elasticity_estimates.csv"
    state.transfers = pd.read_csv(tr_path) if tr_path.exists() else pd.DataFrame()
    state.inventory_metrics = pd.read_csv(inv_path) if inv_path.exists() else pd.DataFrame()

    if el_path.exists():
        el_df = pd.read_csv(el_path)
        global_elasticity = float(el_df["estimated_elasticity"].median())
        state.elasticity = DeployElasticity(el_df, global_elasticity)
        state.elasticity_df = el_df

    state.inventory_analysis = inventory_analysis_from_metrics(state.inventory_metrics)

    bt_path = OUTPUTS_DIR / "backtest_results.csv"
    if bt_path.exists():
        state.backtest_results = {
            "strategy_comparison": pd.read_csv(bt_path, index_col=0),
        }

    return state
