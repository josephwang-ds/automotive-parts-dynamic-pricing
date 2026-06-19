"""通用工具函数。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: Path) -> None:
    """保存 JSON 文件。"""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> dict:
    """加载 JSON 文件。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_currency(value: float) -> str:
    """格式化货币，带千分位。"""
    if pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"


def format_pct(value: float, decimals: int = 1) -> str:
    """格式化百分比。"""
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def round_price(price: float, method: str = "ending_99") -> float:
    """商业价格舍入。"""
    if method == "nearest_dollar":
        return round(price)
    if method == "ending_99":
        return int(price) + 0.99 if price >= 1 else round(price, 2)
    if method == "ending_95":
        return int(price) + 0.95 if price >= 1 else round(price, 2)
    return round(price, 2)


def normalize_series(s: pd.Series) -> pd.Series:
    """Min-max 归一化到 [0, 1]。"""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)


def get_week_splits(n_weeks: int, train: int, val: int, test: int) -> dict:
    """返回时间切分边界。"""
    assert train + val + test == n_weeks
    return {
        "train_end": train,
        "val_end": train + val,
        "test_end": n_weeks,
        "train_range": (0, train),
        "val_range": (train, train + val),
        "test_range": (train + val, n_weeks),
    }


def classify_elasticity(elasticity: float, confidence: float) -> str:
    """弹性分类。"""
    from src.config import ELASTICITY_SEGMENTS

    if confidence < 0.4 or pd.isna(elasticity):
        return "Low confidence"
    for segment, (lo, hi) in ELASTICITY_SEGMENTS.items():
        if segment == "Low confidence":
            continue
        if lo <= elasticity < hi:
            return segment
    return "Moderate"


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法。"""
    if b == 0 or pd.isna(b) or pd.isna(a):
        return default
    return a / b


def compute_inventory_turns(units_sold: float, avg_inventory: float) -> float:
    """计算库存周转率。"""
    return safe_divide(units_sold, avg_inventory, 0.0)


def compute_weeks_of_cover(inventory: float, weekly_demand: float) -> float:
    """计算库存覆盖周数。"""
    if weekly_demand <= 0:
        return 999.0
    return inventory / weekly_demand
