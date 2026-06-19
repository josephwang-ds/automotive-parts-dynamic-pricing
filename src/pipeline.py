"""端到端数据流水线：加载、训练、优化、回测。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest import BacktestEngine, RollbackSimulator
from src.config import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, SCENARIOS
from src.demand_model import DemandModelTrainer, train_all_models
from src.elasticity import ElasticityEstimator
from src.features import build_features
from src.inventory import analyze_inventory
from src.joint_optimizer import JointOptimizer


class PipelineState:
    """流水线状态容器。"""

    def __init__(self):
        self.products: pd.DataFrame = pd.DataFrame()
        self.sales: pd.DataFrame = pd.DataFrame()
        self.features: pd.DataFrame = pd.DataFrame()
        self.demand_model: DemandModelTrainer | None = None
        self.model_results: dict = {}
        self.elasticity: ElasticityEstimator | None = None
        self.elasticity_df: pd.DataFrame = pd.DataFrame()
        self.recommendations: pd.DataFrame = pd.DataFrame()
        self.transfers: pd.DataFrame = pd.DataFrame()
        self.inventory_metrics: pd.DataFrame = pd.DataFrame()
        self.backtest_results: dict = {}
        self.inventory_analysis: dict = {}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载合成数据。"""
    products = pd.read_csv(DATA_DIR / "synthetic_products.csv")
    sales = pd.read_csv(DATA_DIR / "synthetic_sales.csv", parse_dates=["week_start"])
    return products, sales


def run_full_pipeline(scenario: str = "Recommended") -> PipelineState:
    """运行完整流水线。"""
    state = PipelineState()

    # 1. 加载数据
    state.products, state.sales = load_data()

    # 2. 特征工程
    state.features = build_features(state.sales, state.products)

    # 3. 训练需求模型
    print("训练需求模型...")
    state.model_results = train_all_models(state.sales, state.products)
    state.demand_model = DemandModelTrainer.load()

    # 4. 弹性估算
    print("估算价格弹性...")
    state.elasticity = ElasticityEstimator()
    state.elasticity_df = state.elasticity.fit(state.sales, state.products)

    # 5. 库存分析
    print("分析库存...")
    scenario_cfg = SCENARIOS.get(scenario, SCENARIOS["Recommended"])

    # 6. 联合定价+库存优化
    print("生成联合推荐（定价+库存）...")
    joint = JointOptimizer(
        demand_model=state.demand_model,
        elasticity_estimator=state.elasticity,
        scenario=scenario,
    )
    joint_result = joint.generate_joint_recommendations(
        state.sales, state.products, state.elasticity
    )
    state.recommendations = joint_result["recommendations"]
    state.transfers = joint_result["transfers"]
    state.inventory_metrics = joint_result["inventory_metrics"]
    state.inventory_analysis = analyze_inventory(state.sales, state.products, scenario_cfg)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    state.recommendations.to_csv(OUTPUTS_DIR / "recommendations.csv", index=False)
    if not state.transfers.empty:
        state.transfers.to_csv(OUTPUTS_DIR / "transfer_recommendations.csv", index=False)
    state.inventory_metrics.to_csv(OUTPUTS_DIR / "inventory_metrics.csv", index=False)

    # 7. 回测
    backtest = BacktestEngine(state.recommendations, state.sales)
    state.backtest_results = backtest.run_backtest(state.transfers)
    state.backtest_results["strategy_comparison"].to_csv(
        OUTPUTS_DIR / "backtest_results.csv"
    )

    return state


def get_or_run_pipeline(scenario: str = "Recommended") -> PipelineState:
    """获取或运行流水线（带缓存）。"""
    rec_path = OUTPUTS_DIR / "recommendations.csv"
    model_path = MODELS_DIR / "demand_model.joblib"

    if rec_path.exists() and model_path.exists():
        state = PipelineState()
        state.products, state.sales = load_data()
        state.features = build_features(state.sales, state.products)
        state.demand_model = DemandModelTrainer.load()
        state.elasticity = ElasticityEstimator()
        state.elasticity_df = state.elasticity.fit(state.sales, state.products)
        state.recommendations = pd.read_csv(rec_path)
        tr_path = OUTPUTS_DIR / "transfer_recommendations.csv"
        inv_path = OUTPUTS_DIR / "inventory_metrics.csv"
        state.transfers = pd.read_csv(tr_path) if tr_path.exists() else pd.DataFrame()
        state.inventory_metrics = pd.read_csv(inv_path) if inv_path.exists() else pd.DataFrame()
        scenario_cfg = SCENARIOS.get(scenario, SCENARIOS["Recommended"])
        state.inventory_analysis = analyze_inventory(state.sales, state.products, scenario_cfg)

        bt_path = OUTPUTS_DIR / "backtest_results.csv"
        if bt_path.exists():
            state.backtest_results = {
                "strategy_comparison": pd.read_csv(bt_path, index_col=0),
            }
        else:
            backtest = BacktestEngine(state.recommendations, state.sales)
            state.backtest_results = backtest.run_backtest(state.transfers)

        # 加载模型指标
        meta_path = MODELS_DIR / "demand_model_metadata.json"
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            state.model_results = {"hgb": {"metrics": meta.get("metrics", {})}}

        return state

    return run_full_pipeline(scenario)
