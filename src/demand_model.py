"""需求预测模型：基线、线性模型、梯度提升。"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, RANDOM_SEED, TRAIN_WEEKS, VAL_WEEKS
from src.features import (
    GROUP_COLS,
    TARGET_COL,
    build_features,
    get_time_split_masks,
    prepare_model_matrix,
)
from src.metrics import compute_all_metrics, baseline_improvement


class SeasonalNaiveBaseline:
    """季节性朴素基线：使用去年同期或滞后销量。"""

    def __init__(self):
        self.name = "Seasonal Naive"
        self._lookup: pd.DataFrame = pd.DataFrame()

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "SeasonalNaiveBaseline":
        self._lookup = df[GROUP_COLS + ["week_num", TARGET_COL]].copy()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        lookup = self._lookup.rename(columns={TARGET_COL: "pred_val"})
        # lag-52
        lag52 = df[GROUP_COLS + ["week_num"]].copy()
        lag52["week_num"] = lag52["week_num"] - 52
        m52 = lag52.merge(lookup, on=GROUP_COLS + ["week_num"], how="left")
        preds = m52["pred_val"].fillna(0).values
        # lag-1 回退
        lag1 = df[GROUP_COLS + ["week_num"]].copy()
        lag1["week_num"] = lag1["week_num"] - 1
        m1 = lag1.merge(lookup, on=GROUP_COLS + ["week_num"], how="left")
        lag1_vals = m1["pred_val"].fillna(0).values
        return np.where(preds > 0, preds, lag1_vals)


class DemandModelTrainer:
    """需求模型训练器。"""

    def __init__(self, model_type: str = "hgb"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []
        self.metrics: dict = {}
        self.name = model_type

    def _create_model(self):
        if self.model_type == "ridge":
            self.name = "Ridge Regression"
            return Ridge(alpha=1.0, random_state=RANDOM_SEED)
        if self.model_type == "elasticnet":
            self.name = "ElasticNet"
            return ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=2000)
        if self.model_type == "rf":
            self.name = "Random Forest"
            return RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1
            )
        self.name = "HistGradientBoosting"
        return HistGradientBoostingRegressor(
            max_iter=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED
        )

    def train(
        self,
        feature_df: pd.DataFrame,
        products: pd.DataFrame,
    ) -> dict:
        """训练模型并返回评估指标。"""
        df = build_features(feature_df, products) if TARGET_COL not in feature_df.columns else feature_df
        masks = get_time_split_masks(df)

        X_all, y_all, self.feature_cols = prepare_model_matrix(df)

        train_idx = masks["train"]
        val_idx = masks["val"]
        test_idx = masks["test"]

        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_val = X_all[val_idx]
        y_val = y_all[val_idx]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]
        df_test = df[test_idx]

        # 基线
        baseline = SeasonalNaiveBaseline()
        baseline.fit(df[train_idx], y_train)
        baseline_pred = baseline.predict(df[test_idx])
        baseline_metrics = compute_all_metrics(y_test.values, baseline_pred)

        # 训练模型
        self.model = self._create_model()
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

        # 评估
        splits = [
            ("val", X_val, y_val, df[val_idx]),
            ("test", X_test, y_test, df_test),
        ]
        for split_name, X_split, y_split, df_split in splits:
            if len(X_split) == 0:
                continue
            X_scaled = self.scaler.transform(X_split)
            preds = np.maximum(self.model.predict(X_scaled), 0)
            metrics = compute_all_metrics(y_split.values, preds)
            metrics["baseline_WAPE"] = baseline_metrics["WAPE"]
            metrics["WAPE_improvement"] = baseline_improvement(
                baseline_metrics["WAPE"], metrics["WAPE"]
            )
            self.metrics[split_name] = metrics

        # 测试集预测
        X_test_scaled = self.scaler.transform(X_test)
        test_preds = np.maximum(self.model.predict(X_test_scaled), 0)

        self.metrics["baseline"] = baseline_metrics
        self.metrics["model_name"] = self.name
        self.metrics["train_weeks"] = TRAIN_WEEKS
        self.metrics["val_weeks"] = VAL_WEEKS

        return {
            "metrics": self.metrics,
            "test_predictions": pd.DataFrame({
                "week_num": df_test["week_num"].values,
                "sku_id": df_test["sku_id"].values,
                "category": df_test["category"].values,
                "region": df_test["region"].values,
                "customer_tier": df_test["customer_tier"].values,
                "actual": y_test.values,
                "predicted": test_preds,
            }),
            "feature_importance": self._get_feature_importance(),
        }

    def _get_feature_importance(self) -> pd.DataFrame | None:
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
            return pd.DataFrame({
                "feature": self.feature_cols[:len(imp)],
                "importance": imp,
            }).sort_values("importance", ascending=False)
        if hasattr(self.model, "coef_"):
            return pd.DataFrame({
                "feature": self.feature_cols,
                "importance": np.abs(self.model.coef_),
            }).sort_values("importance", ascending=False)
        return None

    def predict_at_price(
        self,
        row: pd.Series,
        candidate_price: float,
        feature_template: pd.DataFrame | None = None,
        elasticity: float = -1.0,
    ) -> float:
        """在候选价格下预测需求。"""
        current_price = row.get("realized_price", candidate_price)
        base_units = row.get("units_sold", row.get("adjusted_units", 10))

        # 快速路径：弹性公式（批量推荐时使用）
        if feature_template is None and current_price > 0:
            price_ratio = candidate_price / current_price
            return max(0.0, float(base_units * (price_ratio ** elasticity)))

        feat_row = row.copy()
        feat_row["realized_price"] = candidate_price
        if row.get("regular_price", 0) > 0:
            feat_row["price_change_pct"] = candidate_price / row["regular_price"] - 1

        X, _, _ = prepare_model_matrix(pd.DataFrame([feat_row]))
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        return max(0, float(self.model.predict(X_scaled)[0]))

    def predict_batch_at_prices(
        self,
        row: pd.Series,
        candidate_prices: list[float],
        elasticity: float = -1.0,
    ) -> list[float]:
        """批量预测多个候选价格（弹性快速路径）。"""
        current_price = row.get("realized_price", 1)
        base_units = row.get("units_sold", row.get("adjusted_units", 10))
        if current_price <= 0:
            return [base_units] * len(candidate_prices)
        ratios = np.array(candidate_prices) / current_price
        return [max(0.0, float(base_units * (r ** elasticity))) for r in ratios]

    def save(self, path: Path | None = None) -> Path:
        """保存模型工件。"""
        path = path or MODELS_DIR / "demand_model.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "model_type": self.model_type,
            "name": self.name,
            "metrics": self.metrics,
        }
        joblib.dump(artifact, path)

        meta_path = MODELS_DIR / "demand_model_metadata.json"
        with open(meta_path, "w") as f:
            json.dump({
                "model_name": self.name,
                "model_type": self.model_type,
                "feature_count": len(self.feature_cols),
                "metrics": self.metrics,
            }, f, indent=2, default=str)

        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "DemandModelTrainer":
        """加载模型。"""
        path = path or MODELS_DIR / "demand_model.joblib"
        artifact = joblib.load(path)
        trainer = cls(model_type=artifact["model_type"])
        trainer.model = artifact["model"]
        trainer.scaler = artifact["scaler"]
        trainer.feature_cols = artifact["feature_cols"]
        trainer.name = artifact["name"]
        trainer.metrics = artifact.get("metrics", {})
        return trainer


def train_all_models(sales: pd.DataFrame, products: pd.DataFrame) -> dict:
    """训练全部模型并比较。"""
    results = {}
    for model_type in ["ridge", "hgb"]:
        print(f"  训练模型: {model_type}...")
        trainer = DemandModelTrainer(model_type)
        result = trainer.train(sales, products)
        results[model_type] = result
        if model_type == "hgb":
            trainer.save()
            print(f"  模型已保存")
    return results
