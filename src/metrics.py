"""评估指标计算。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差。"""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差。"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """加权绝对百分比误差。"""
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根对数误差。"""
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """预测偏差。"""
    return float(np.mean(y_pred - y_true))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算全部指标。"""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "RMSLE": rmsle(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
    }


def baseline_improvement(baseline_metric: float, model_metric: float) -> float:
    """相对基线的改进百分比。"""
    if baseline_metric == 0:
        return 0.0
    return (baseline_metric - model_metric) / baseline_metric
