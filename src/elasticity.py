"""价格弹性估算：log-log 回归与层级收缩。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.config import CATEGORIES, CUSTOMER_TIERS, REGIONS
from src.utils import classify_elasticity


class ElasticityEstimator:
    """价格弹性估算器。"""

    def __init__(self):
        self.estimates: pd.DataFrame = pd.DataFrame()
        self.global_elasticity: float = -1.0

    def fit(self, sales: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
        """估算各层级弹性。"""
        df = sales.copy()

        # 筛选高质量样本
        quality_mask = (
            (df["price_test_flag"] == True)  # noqa: E712
            | (df["price_change_reason"].isin([
                "random_regional_test", "scheduled_review", "supplier_cost_change"
            ]))
        ) & (~df["stockout_flag"]) & (df["weeks_of_cover"] > 2)

        est_df = df[quality_mask].copy()
        if len(est_df) < 100:
            est_df = df[~df["stockout_flag"]].copy()

        est_df["log_units"] = np.log1p(
            est_df["units_sold"] + est_df.get("lost_sales_estimate", 0)
        )
        est_df["log_price"] = np.log(est_df["realized_price"].clip(lower=0.01))

        records = []

        # 全局弹性
        global_est = self._fit_group(est_df, "global")
        self.global_elasticity = global_est["elasticity"]

        # Category × tier
        for cat in CATEGORIES:
            for tier in CUSTOMER_TIERS:
                mask = (est_df["category"] == cat) & (est_df["customer_tier"] == tier)
                group_data = est_df[mask]
                est = self._fit_group(group_data, f"{cat}|{tier}")
                records.append({
                    "elasticity_segment": f"{cat} × {tier}",
                    "category": cat,
                    "region": "All",
                    "customer_tier": tier,
                    "estimated_elasticity": est["elasticity"],
                    "sample_size": est["sample_size"],
                    "price_variation": est["price_variation"],
                    "standard_error": est["std_error"],
                    "confidence_score": est["confidence"],
                    "estimation_level": "category_tier",
                })

        # Category × region × tier（样本足够时）
        for cat in CATEGORIES:
            for region in REGIONS:
                for tier in CUSTOMER_TIERS:
                    mask = (
                        (est_df["category"] == cat)
                        & (est_df["region"] == region)
                        & (est_df["customer_tier"] == tier)
                    )
                    group_data = est_df[mask]
                    if len(group_data) < 30:
                        continue
                    est = self._fit_group(group_data, f"{cat}|{region}|{tier}")
                    records.append({
                        "elasticity_segment": f"{cat} × {region} × {tier}",
                        "category": cat,
                        "region": region,
                        "customer_tier": tier,
                        "estimated_elasticity": est["elasticity"],
                        "sample_size": est["sample_size"],
                        "price_variation": est["price_variation"],
                        "standard_error": est["std_error"],
                        "confidence_score": est["confidence"],
                        "estimation_level": "category_region_tier",
                    })

        self.estimates = pd.DataFrame(records)

        # 分类标签
        self.estimates["elasticity_class"] = self.estimates.apply(
            lambda r: classify_elasticity(r["estimated_elasticity"], r["confidence_score"]),
            axis=1,
        )

        # 收缩低样本估计
        self._shrink_low_confidence()

        return self.estimates

    def _fit_group(self, data: pd.DataFrame, label: str) -> dict:
        """对单个分组拟合 log-log 模型。"""
        if len(data) < 10:
            cat_el = self._get_category_prior(data)
            return {
                "elasticity": cat_el,
                "sample_size": len(data),
                "price_variation": 0.0,
                "std_error": 1.0,
                "confidence": 0.2,
            }

        price_var = data["log_price"].std()
        if price_var < 0.01:
            return {
                "elasticity": self.global_elasticity,
                "sample_size": len(data),
                "price_variation": price_var,
                "std_error": 0.5,
                "confidence": 0.3,
            }

        try:
            X = data[["log_price"]].values
            # 添加控制变量
            for col in ["competitor_price_index", "promotion_depth", "seasonality_index"]:
                if col in data.columns:
                    X = np.column_stack([X, data[col].values])

            y = data["log_units"].values
            model = Ridge(alpha=1.0)
            model.fit(X, y)

            elasticity = float(np.clip(model.coef_[0], -3.0, -0.1))
            residuals = y - model.predict(X)
            std_error = float(np.std(residuals) / np.sqrt(len(data)))
            confidence = min(1.0, len(data) / 200) * min(1.0, price_var / 0.1)

            return {
                "elasticity": elasticity,
                "sample_size": len(data),
                "price_variation": float(price_var),
                "std_error": std_error,
                "confidence": confidence,
            }
        except Exception:
            return {
                "elasticity": self.global_elasticity,
                "sample_size": len(data),
                "price_variation": 0.0,
                "std_error": 1.0,
                "confidence": 0.2,
            }

    def _get_category_prior(self, data: pd.DataFrame) -> float:
        """获取品类先验弹性。"""
        if self.estimates.empty:
            return self.global_elasticity
        if len(data) > 0 and "category" in data.columns:
            cat = data["category"].iloc[0]
            if "category" in self.estimates.columns:
                cat_est = self.estimates[
                    (self.estimates["category"] == cat) & (self.estimates["region"] == "All")
                ]
                if len(cat_est) > 0:
                    return cat_est["estimated_elasticity"].median()
        return self.global_elasticity

    def _shrink_low_confidence(self):
        """低置信度估计向全局先验收缩。"""
        for idx, row in self.estimates.iterrows():
            if row["confidence_score"] < 0.4:
                shrunk = (
                    row["confidence_score"] * row["estimated_elasticity"]
                    + (1 - row["confidence_score"]) * self.global_elasticity
                )
                self.estimates.at[idx, "estimated_elasticity"] = np.clip(shrunk, -3.0, -0.1)
                self.estimates.at[idx, "elasticity_class"] = "Low confidence"

    def get_elasticity(
        self,
        category: str,
        region: str,
        tier: str,
    ) -> dict:
        """获取特定组合的弹性估计。"""
        # 优先 category × region × tier
        match = self.estimates[
            (self.estimates["category"] == category)
            & (self.estimates["region"] == region)
            & (self.estimates["customer_tier"] == tier)
            & (self.estimates["estimation_level"] == "category_region_tier")
        ]
        if len(match) > 0:
            row = match.iloc[0]
            return row.to_dict()

        # 回退到 category × tier
        match = self.estimates[
            (self.estimates["category"] == category)
            & (self.estimates["region"] == "All")
            & (self.estimates["customer_tier"] == tier)
        ]
        if len(match) > 0:
            row = match.iloc[0]
            return row.to_dict()

        return {
            "estimated_elasticity": self.global_elasticity,
            "confidence_score": 0.3,
            "sample_size": 0,
            "price_variation": 0,
            "elasticity_class": "Low confidence",
            "estimation_level": "global_prior",
        }
