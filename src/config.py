"""全局配置：业务参数、模型参数、优化权重、场景预设。"""

from pathlib import Path

# 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# 业务维度
CATEGORIES = [
    "Brakes",
    "Filters",
    "Electrical",
    "Suspension",
    "Fluids",
    "Tools",
    "Batteries",
    "Engine Components",
]

REGIONS = [
    "Lower Mainland",
    "Vancouver Island",
    "Interior",
    "Northern BC",
]

CUSTOMER_TIERS = ["Retail", "Trade", "Fleet"]

# 数据规模
N_SKUS = 3000
N_WEEKS = 104
RANDOM_SEED = 42

# 时间切分（周）
TRAIN_WEEKS = 78
VAL_WEEKS = 13
TEST_WEEKS = 13

# 品类弹性先验范围
CATEGORY_ELASTICITY_PRIORS = {
    "Brakes": (-1.2, -0.5),
    "Filters": (-2.5, -1.0),
    "Electrical": (-1.5, -0.4),
    "Suspension": (-1.3, -0.6),
    "Fluids": (-2.8, -1.2),
    "Tools": (-0.8, -0.3),
    "Batteries": (-1.8, -0.7),
    "Engine Components": (-1.6, -0.8),
}

# 客户层级价格敏感度乘数
TIER_SENSITIVITY = {"Retail": 0.85, "Trade": 1.0, "Fleet": 1.15}

# 地区需求乘数
REGION_DEMAND_FACTORS = {
    "Lower Mainland": 1.35,
    "Vancouver Island": 0.85,
    "Interior": 0.95,
    "Northern BC": 0.75,
}

# 价格优化
CANDIDATE_PRICE_MIN_PCT = -0.15
CANDIDATE_PRICE_MAX_PCT = 0.15
CANDIDATE_PRICE_STEP_PCT = 0.01

# Guardrails
MAX_PRICE_MOVE_PCT = 0.10
EXCESS_INVENTORY_MAX_MOVE_PCT = 0.15
LOW_CONFIDENCE_MAX_MOVE_PCT = 0.03
MARGIN_FLOOR_DEFAULT = 0.15
PRICE_ROUNDING_OPTIONS = ["nearest_dollar", "ending_99", "ending_95"]

# 库存阈值（向后兼容别名）
WEEKS_OF_COVER_EXCESS = 16
WEEKS_OF_COVER_STOCKOUT_RISK = 2
SLOW_MOVING_WEEKS = 20

# ── Inventory Decision Engine 参数 ──
INVENTORY_POLICY = {
    "demand_window_weeks": 8,
    "demand_window_long_weeks": 13,
    "target_service_level": 0.95,
    "service_level_z": 1.65,
    "minimum_weeks_of_cover": 4,
    "target_weeks_of_cover": 8,
    "maximum_weeks_of_cover": 16,
    "stockout_probability_threshold": 0.60,
    "slow_moving_turns_threshold": 2.0,
    "obsolescence_threshold": 0.65,
    "annual_holding_cost_rate": 0.25,
    "transfer_cost_per_unit": 2.50,
    "minimum_transfer_value": 50.0,
    "maximum_transfer_units": 500,
    "transfer_lead_time_weeks": 1,
    "replenishment_review_period_weeks": 1,
    "minimum_order_quantity": 1,
    "maximum_markdown_pct": 0.15,
    "clearance_markdown_pct": 0.20,
    "minimum_markdown_value": 10.0,
    "low_confidence_price_cap": 0.03,
    "low_confidence_transfer_cap": 100,
    "lost_sales_risk_threshold": 0.30,
    "demand_epsilon": 0.01,
}

# 库存健康状态优先级（高→低）
INVENTORY_STATUS_PRIORITY = [
    "STOCKOUT",
    "STOCKOUT_RISK",
    "OBSOLETE_RISK",
    "UNDERSTOCKED",
    "SLOW_MOVING",
    "OVERSTOCKED",
    "HEALTHY",
]

INVENTORY_ACTIONS = [
    "REPLENISH",
    "EXPEDITE_ORDER",
    "INTER_REGION_TRANSFER",
    "PRICE_MARKDOWN",
    "HOLD_PRICE",
    "STOP_OR_DELAY_ORDER",
    "LIQUIDATION_REVIEW",
    "MANUAL_REVIEW",
    "NO_ACTION",
]

JOINT_CONFIDENCE_LEVELS = ["HIGH", "MEDIUM", "LOW"]

# 优化目标权重（Balanced Objective）
BALANCED_WEIGHTS = {
    "gross_profit": 0.40,
    "inventory_reduction": 0.25,
    "stockout_penalty": 0.20,
    "price_change_penalty": 0.10,
    "low_confidence_penalty": 0.05,
}

# 场景预设
SCENARIOS = {
    "Conservative": {
        "max_price_move_pct": 0.05,
        "objective": "balanced",
        "confidence_threshold": 0.6,
        "stockout_protection": True,
        "service_level_z": 1.96,
        "minimum_transfer_value": 100.0,
        "target_weeks_of_cover": 10,
    },
    "Recommended": {
        "max_price_move_pct": 0.10,
        "objective": "balanced",
        "confidence_threshold": 0.4,
        "stockout_protection": True,
        "service_level_z": 1.65,
        "minimum_transfer_value": 50.0,
        "target_weeks_of_cover": 8,
    },
    "Margin Push": {
        "max_price_move_pct": 0.10,
        "objective": "maximize_gross_profit",
        "confidence_threshold": 0.3,
        "stockout_protection": True,
        "service_level_z": 1.65,
        "minimum_transfer_value": 75.0,
        "target_weeks_of_cover": 8,
    },
    "Inventory Clearance": {
        "max_price_move_pct": 0.15,
        "objective": "reduce_excess_inventory",
        "confidence_threshold": 0.3,
        "stockout_protection": False,
        "service_level_z": 1.28,
        "minimum_transfer_value": 30.0,
        "target_weeks_of_cover": 6,
        "maximum_markdown_pct": 0.20,
    },
}

# 目标函数映射
OBJECTIVES = {
    "maximize_gross_profit": "Maximize Gross Profit",
    "maximize_revenue": "Maximize Revenue",
    "reduce_excess_inventory": "Reduce Excess Inventory",
    "balanced": "Balanced Objective",
}

# 弹性分类阈值
ELASTICITY_SEGMENTS = {
    "Highly elastic": (-3.0, -1.5),
    "Elastic": (-1.5, -1.0),
    "Moderate": (-1.0, -0.5),
    "Inelastic": (-0.5, -0.1),
    "Low confidence": (-0.1, 0.0),
}

# UI 设计系统
UI_COLORS = {
    "background": "#F6F8FC",
    "primary_navy": "#0F172A",
    "electric_blue": "#2563EB",
    "emerald": "#059669",
    "warning_amber": "#D97706",
    "risk_red": "#DC2626",
    "border": "#E2E8F0",
    "muted_text": "#64748B",
}

# 策略版本
POLICY_VERSION = "v2.0.0-inventory-engine"
