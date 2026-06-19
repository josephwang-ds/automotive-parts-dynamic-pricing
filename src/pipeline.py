"""端到端数据流水线：加载、训练、优化、回测。"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.backtest import BacktestEngine, RollbackSimulator
from src.config import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, SCENARIOS
from src.demand_model import DemandModelTrainer, train_all_models
from src.elasticity import ElasticityEstimator
from src.features import build_features
from src.inventory import analyze_inventory, inventory_analysis_from_metrics
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


def _is_streamlit_cloud() -> bool:
    """检测是否在 Streamlit Cloud 容器内运行。"""
    return Path("/mount/src").exists() or os.getenv("STREAMLIT_RUNTIME_ENV") == "cloud"


def load_data(deploy: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载合成数据。"""
    products = pd.read_csv(DATA_DIR / "synthetic_products.csv")
    if deploy:
        for name in (
            "synthetic_sales_latest.csv",
            "synthetic_sales_app.csv",
            "synthetic_sales.csv",
        ):
            candidate = DATA_DIR / name
            if candidate.exists():
                sales_path = candidate
                break
        else:
            raise FileNotFoundError("未找到部署用销售数据文件。")
    else:
        sales_path = DATA_DIR / "synthetic_sales.csv"
    sales = pd.read_csv(sales_path, parse_dates=["week_start"])
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
    state.elasticity_df.to_csv(OUTPUTS_DIR / "elasticity_estimates.csv", index=False)

    # 7. 回测
    backtest = BacktestEngine(state.recommendations, state.sales)
    state.backtest_results = backtest.run_backtest(state.transfers)
    state.backtest_results["strategy_comparison"].to_csv(
        OUTPUTS_DIR / "backtest_results.csv"
    )

    return state


def _load_deploy_pipeline(scenario: str) -> PipelineState:
    """部署快速路径：只加载预计算产物，避免在 Streamlit Cloud 上重算特征/弹性。"""
    state = PipelineState()
    meta_path = MODELS_DIR / "demand_model_metadata.json"

    state.products, state.sales = load_data(deploy=True)

    state.demand_model = DemandModelTrainer.load_metadata(meta_path)
    state.recommendations = pd.read_csv(OUTPUTS_DIR / "recommendations.csv")
    tr_path = OUTPUTS_DIR / "transfer_recommendations.csv"
    inv_path = OUTPUTS_DIR / "inventory_metrics.csv"
    el_path = OUTPUTS_DIR / "elasticity_estimates.csv"
    state.transfers = pd.read_csv(tr_path) if tr_path.exists() else pd.DataFrame()
    state.inventory_metrics = pd.read_csv(inv_path) if inv_path.exists() else pd.DataFrame()

    if el_path.exists():
        state.elasticity = ElasticityEstimator.from_estimates_csv(el_path)
        state.elasticity_df = state.elasticity.estimates
    else:
        state.elasticity = ElasticityEstimator()
        state.elasticity_df = state.elasticity.fit(state.sales, state.products)

    state.inventory_analysis = inventory_analysis_from_metrics(state.inventory_metrics)

    bt_path = OUTPUTS_DIR / "backtest_results.csv"
    if bt_path.exists():
        state.backtest_results = {
            "strategy_comparison": pd.read_csv(bt_path, index_col=0),
        }

    state.model_results = {"hgb": {"metrics": state.demand_model.metrics}}
    return state


def get_or_run_pipeline(scenario: str = "Recommended") -> PipelineState:
    """获取或运行流水线（带缓存）。"""
    rec_path = OUTPUTS_DIR / "recommendations.csv"
    meta_path = MODELS_DIR / "demand_model_metadata.json"

    # Deployment path: use portable metadata plus precomputed outputs. A
    # scikit-learn joblib created under another Python/NumPy version is not a
    # stable cross-environment deployment format and is unnecessary here.
    if rec_path.exists() and meta_path.exists():
        return _load_deploy_pipeline(scenario)

    if _is_streamlit_cloud():
        raise FileNotFoundError(
            "Streamlit Cloud 缺少部署产物。请确认仓库已提交 "
            "outputs/recommendations.csv 与 models/demand_model_metadata.json。"
        )

    return run_full_pipeline(scenario)
