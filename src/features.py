"""特征工程流水线：无未来泄漏的滞后与滚动特征。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TRAIN_WEEKS, VAL_WEEKS, TEST_WEEKS, N_WEEKS

# 数值特征列
NUMERIC_FEATURES = [
    "lag_1_units",
    "lag_2_units",
    "lag_4_units",
    "lag_13_units",
    "rolling_4_units",
    "rolling_8_units",
    "rolling_13_units",
    "rolling_4_revenue",
    "rolling_13_margin",
    "prior_week_inventory",
    "price_change_pct",
    "price_index_vs_competitor",
    "tier_discount_pct",
    "promotion_flag",
    "seasonality_index",
    "week_of_year",
    "month",
    "lead_time_days",
    "weeks_of_cover",
    "realized_price",
    "unit_cost",
    "competitor_price_index",
    "economic_demand_index",
]

CATEGORICAL_FEATURES = [
    "category",
    "region",
    "customer_tier",
    "lifecycle_stage",
]

TARGET_COL = "adjusted_units"
GROUP_COLS = ["sku_id", "region", "customer_tier"]


def _compute_adjusted_units(df: pd.DataFrame) -> pd.Series:
    """调整销量：库存截断时加上丢失销量估计。"""
    adjusted = df["units_sold"].copy()
    # 缺货周：用潜在需求或 units + lost_sales
    mask = df["stockout_flag"] | (df["lost_sales_estimate"] > 0)
    if "latent_demand" in df.columns:
        adjusted[mask] = df.loc[mask, "latent_demand"]
    else:
        adjusted[mask] = df.loc[mask, "units_sold"] + df.loc[mask, "lost_sales_estimate"]
    return adjusted


def build_features(sales: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """构建特征数据集，确保无未来泄漏。"""
    df = sales.copy()
    df = df.sort_values(["sku_id", "region", "customer_tier", "week_num"]).reset_index(drop=True)

    # 合并产品属性
    prod_cols = ["sku_id", "lifecycle_stage", "lead_time_days", "product_age_weeks"]
    df = df.merge(products[prod_cols], on="sku_id", how="left", suffixes=("", "_prod"))

    # 调整目标
    df[TARGET_COL] = _compute_adjusted_units(df)

    # 时间特征
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month

    # 价格变化
    df["price_change_pct"] = (
        df.groupby(GROUP_COLS)["realized_price"].pct_change().fillna(0)
    )
    df["price_index_vs_competitor"] = df["competitor_price_index"]
    df["tier_discount_pct"] = df["customer_tier_discount_pct"]

    # 滞后特征（仅使用过去数据）
    for lag in [1, 2, 4, 13]:
        df[f"lag_{lag}_units"] = (
            df.groupby(GROUP_COLS)["units_sold"].shift(lag)
        )

    # 滚动特征（shift 1 后 rolling，避免包含当前周）
    for window in [4, 8, 13]:
        shifted = df.groupby(GROUP_COLS)["units_sold"].shift(1)
        df[f"rolling_{window}_units"] = (
            shifted.groupby([df["sku_id"], df["region"], df["customer_tier"]])
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=[0, 1, 2], drop=True)
        )

    shifted_rev = df.groupby(GROUP_COLS)["revenue"].shift(1)
    df["rolling_4_revenue"] = (
        shifted_rev.groupby([df["sku_id"], df["region"], df["customer_tier"]])
        .rolling(4, min_periods=1)
        .mean()
        .reset_index(level=[0, 1, 2], drop=True)
    )

    shifted_margin = df.groupby(GROUP_COLS)["gross_margin_pct"].shift(1)
    df["rolling_13_margin"] = (
        shifted_margin.groupby([df["sku_id"], df["region"], df["customer_tier"]])
        .rolling(13, min_periods=1)
        .mean()
        .reset_index(level=[0, 1, 2], drop=True)
    )

    df["prior_week_inventory"] = df.groupby(GROUP_COLS)["ending_inventory"].shift(1)

    # 填充缺失
    numeric_fill = {
        "lag_1_units": 0, "lag_2_units": 0, "lag_4_units": 0, "lag_13_units": 0,
        "rolling_4_units": 0, "rolling_8_units": 0, "rolling_13_units": 0,
        "rolling_4_revenue": 0, "rolling_13_margin": 0,
        "prior_week_inventory": 0, "price_change_pct": 0,
        "weeks_of_cover": 0,
    }
    df = df.fillna(numeric_fill)

    return df


def get_feature_columns() -> list[str]:
    """返回模型特征列名。"""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def get_time_split_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """时间切分掩码（自适应数据长度）。"""
    max_week = df["week_num"].max()
    if max_week <= TRAIN_WEEKS + VAL_WEEKS + TEST_WEEKS:
        # 小数据集：按比例切分 75/12.5/12.5
        train_end = int(max_week * 0.75)
        val_end = int(max_week * 0.875)
    else:
        train_end = TRAIN_WEEKS
        val_end = TRAIN_WEEKS + VAL_WEEKS

    return {
        "train": df["week_num"] <= train_end,
        "val": (df["week_num"] > train_end) & (df["week_num"] <= val_end),
        "test": df["week_num"] > val_end,
    }


def prepare_model_matrix(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """准备模型输入矩阵，编码分类变量。"""
    if feature_cols is None:
        feature_cols = get_feature_columns()

    work = df.copy()
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # One-hot 编码
    if cat_cols:
        dummies = pd.get_dummies(work[cat_cols], prefix=cat_cols, drop_first=True)
        work = pd.concat([work[num_cols], dummies], axis=1)
        final_cols = list(work.columns)
    else:
        work = work[num_cols]
        final_cols = num_cols

    work = work.fillna(0)
    y = df[TARGET_COL]
    return work, y, final_cols


def verify_no_future_leakage(df: pd.DataFrame) -> bool:
    """验证滞后特征不使用未来数据。"""
    for lag in [1, 2, 4, 13]:
        col = f"lag_{lag}_units"
        if col not in df.columns:
            continue
        grouped = df.sort_values("week_num").groupby(GROUP_COLS)
        for _, grp in grouped:
            if len(grp) < lag + 1:
                continue
            for i in range(lag, len(grp)):
                expected = grp.iloc[i - lag]["units_sold"]
                actual = grp.iloc[i][col]
                if not (pd.isna(actual) or abs(actual - expected) < 0.01):
                    return False
    return True
