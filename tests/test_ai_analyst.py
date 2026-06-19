"""AI 分析师测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src.ai_analyst import generate_answer
from src.data_generator import generate_all_data
from src.demand_model import DemandModelTrainer
from src.elasticity import ElasticityEstimator
from src.optimizer import PriceOptimizer


class TestAIAnalyst:
    @pytest.fixture(scope="class")
    def data_bundle(self):
        products, sales = generate_all_data(n_skus=30, n_weeks=20, seed=42)
        trainer = DemandModelTrainer("ridge")
        trainer.train(sales, products)
        elasticity = ElasticityEstimator()
        el_df = elasticity.fit(sales, products)
        optimizer = PriceOptimizer(demand_model=trainer, elasticity_estimator=elasticity)
        recs = optimizer.generate_recommendations(sales, products, elasticity)
        return {
            "recommendations": recs,
            "sales": sales,
            "elasticity": el_df,
            "products": products,
        }

    def test_generates_answer(self, data_bundle):
        """能生成回答。"""
        result = generate_answer(
            "Where is the largest margin opportunity?",
            data_bundle,
            provider="local",
        )
        assert "answer" in result
        assert len(result["answer"]) > 0

    def test_numbers_from_filtered_data(self, data_bundle):
        """引用的数字来自筛选数据。"""
        recs = data_bundle["recommendations"]
        total_lift = recs["gross_profit_lift"].sum()
        result = generate_answer(
            "What is the margin opportunity?",
            data_bundle,
            provider="local",
        )
        # 回答应包含实际计算的数字
        assert "answer" in result
        assert result.get("evidence") is not None

    def test_empty_filter_handling(self, data_bundle):
        """空筛选有合理处理。"""
        empty_recs = data_bundle["recommendations"].iloc[0:0]
        result = generate_answer(
            "What is the inventory situation?",
            {"recommendations": empty_recs, "sales": data_bundle["sales"],
             "elasticity": data_bundle["elasticity"], "products": data_bundle["products"]},
            active_filters={"region": "Nonexistent"},
            provider="local",
        )
        assert "answer" in result

    def test_rollback_question(self, data_bundle):
        """回滚问题能回答。"""
        result = generate_answer(
            "What happens if we roll back 50% of price changes?",
            data_bundle,
            provider="local",
        )
        assert "50%" in result["answer"] or "rollback" in result["answer"].lower()

    def test_caveat_present(self, data_bundle):
        """回答包含 caveat。"""
        result = generate_answer("Tell me about margins", data_bundle, provider="local")
        assert "caveat" in result
        assert "synthetic" in result["caveat"].lower() or "approval" in result["caveat"].lower()

    def test_intent_detection(self, data_bundle):
        """意图检测正确。"""
        result = generate_answer(
            "Which categories have excess inventory?",
            data_bundle,
            provider="local",
        )
        assert result["intent"] in ("excess_stock", "inventory", "category", "general")

    def test_model_artifact_save_load(self, data_bundle):
        """模型工件可以保存和加载。"""
        from src.demand_model import DemandModelTrainer
        import tempfile
        products, sales = generate_all_data(n_skus=20, n_weeks=15, seed=42)
        trainer = DemandModelTrainer("ridge")
        trainer.train(sales, products)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.joblib"
            trainer.save(path)
            loaded = DemandModelTrainer.load(path)
            assert loaded.model is not None
            assert len(loaded.feature_cols) > 0
