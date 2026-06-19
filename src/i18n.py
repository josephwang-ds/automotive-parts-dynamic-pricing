"""轻量 i18n：中英文切换。

用法：
    from src import i18n
    i18n.set_language("zh")        # 或 "en"
    i18n.t("Current Revenue")      # 简单标签：英文做 key，返回对应语言文案
    i18n.tf("exec_summary", n=10)  # 模板：语义 key + format 参数

设计说明：
- 英文是源语言，也是简单标签的 key；缺失时回退英文（永不报 KeyError）。
- 过滤值（区域 / 品类 / SKU 等）刻意保持英文，因为要和数据列里的值匹配，
  翻译它们会破坏筛选逻辑。仅翻译界面文案。
"""

from __future__ import annotations

LANGUAGES = {"en": "English", "zh": "中文"}

_LANG = "en"


def set_language(lang: str) -> None:
    global _LANG
    _LANG = lang if lang in LANGUAGES else "en"


def get_language() -> str:
    return _LANG


def t(key: str) -> str:
    """简单标签翻译：英文 key -> 当前语言文案。"""
    if _LANG == "en":
        return key
    return _ZH.get(key, key)


def tf(key: str, **kwargs) -> str:
    """模板翻译：语义 key -> 当前语言模板，再 format。"""
    template = (_EN_TPL if _LANG == "en" else _ZH_TPL).get(key, _EN_TPL.get(key, ""))
    try:
        return template.format(**kwargs)
    except Exception:
        return template


# ── 简单标签：英文 -> 中文 ──
_ZH = {
    # 全局 / 侧边栏
    "Parts Dynamic Pricing & Inventory AI": "汽车零配件动态定价与库存优化 AI",
    "Loading data...": "正在加载数据…",
    "App failed to start. Ensure the repo includes the outputs/ and data/ deploy files.":
        "应用启动失败。请确认仓库已包含 outputs/ 与 data/ 部署文件。",
    "Language": "语言",
    "Filters": "筛选",
    "Navigation": "导航",
    "Region": "区域",
    "Category": "品类",
    "Customer Tier": "客户层级",
    "Pricing Objective": "定价目标",
    "Scenario": "情景",
    "Margin Floor": "毛利下限",
    "Max Price Move %": "最大调价幅度 %",
    "Confidence Threshold": "置信度阈值",
    "All": "全部",
    # 页面名（导航 + 标题共用）
    "Executive Command Center": "经营指挥中心",
    "Demand Model": "需求模型",
    "Elasticity Explorer": "价格弹性浏览",
    "SKU Decision Workbench": "SKU 决策工作台",
    "Inventory Control Tower": "库存控制塔",
    "Backtest & Rollback": "回测与回滚",
    "AI Analyst": "AI 分析助手",
    "Data & Governance": "数据与治理",
    # 经营指挥中心
    "SKUs in Scope": "范围内 SKU 数",
    "Current Revenue": "当前营收",
    "Current Gross Profit": "当前毛利",
    "Modeled GP Lift": "模型毛利提升",
    "Simulated estimate": "模拟估计",
    "Inventory Value": "库存价值",
    "Avg Inventory Turns": "平均库存周转",
    "Excess Inventory": "过剩库存",
    "Stockout-Risk SKUs": "缺货风险 SKU",
    "Modeled Opportunity by Category": "各品类模型机会",
    "Recommendation Action Distribution": "建议动作分布",
    "Opportunity by Region": "各区域机会",
    "Top Approval Candidates": "重点审批候选",
    # 需求模型
    "Demand Forecasting Model": "需求预测模型",
    "Baseline Improvement": "相对基线提升",
    "Model Comparison": "模型对比",
    "Actual vs Predicted": "实际 vs 预测",
    "Residual Distribution": "残差分布",
    "Error by Category": "各品类误差",
    # 弹性浏览
    "Price Elasticity Explorer": "价格弹性浏览",
    "Elasticity estimates not available.": "暂无弹性估计。",
    "Elasticity Heatmap: Category × Tier": "弹性热力图：品类 × 层级",
    "Elasticity": "弹性",
    "Confidence": "置信度",
    "Sample Size": "样本量",
    "Price Variation": "价格波动",
    "Segment": "区间",
    # SKU 决策工作台
    "No recommendations for current filters.": "当前筛选下没有建议。",
    "Select SKU": "选择 SKU",
    "Current Price": "当前价格",
    "Recommended Price": "建议价格",
    "Price Change": "价格变动",
    "Action": "动作",
    "Predicted Units (Current)": "预测销量（当前）",
    "Predicted Units (Recommended)": "预测销量（建议）",
    "GP Lift": "毛利提升",
    "Margin": "毛利率",
    "Inventory Decision": "库存决策",
    "Inventory Status": "库存状态",
    "Inventory Action": "库存动作",
    "Weeks of Cover": "可售周数",
    "Joint Confidence": "联合置信度",
    "On-Hand": "在手库存",
    "On-Order": "在途库存",
    "Stockout Prob.": "缺货概率",
    "Reorder Qty": "补货数量",
    "Price vs Predicted Units": "价格 vs 预测销量",
    "Price vs Gross Profit": "价格 vs 毛利",
    # 库存控制塔
    "Total Inventory Value": "库存总价值",
    "Est. Lost Sales": "预计损失销量",
    "Avg Weeks of Cover": "平均可售周数",
    "Transfer Opportunities": "调拨机会",
    "Replenishment Candidates": "补货候选",
    "Inventory Health Distribution": "库存健康度分布",
    "Inventory Value by Status": "各状态库存价值",
    "Weeks-of-Cover Distribution": "可售周数分布",
    "Margin vs Inventory Turns": "毛利 vs 库存周转",
    "Inventory Action Distribution": "库存动作分布",
    "Pricing vs Inventory Action Matrix": "定价 vs 库存动作矩阵",
    "Top Excess Inventory Candidates": "过剩库存重点候选",
    "Transfer Recommendations": "调拨建议",
    "Manual Review Queue": "人工复核队列",
    # 回测与回滚
    "Backtest & Rollback Simulator": "回测与回滚模拟",
    "Gross Profit Comparison by Strategy": "各策略毛利对比",
    "Revenue Comparison by Strategy": "各策略营收对比",
    "Rollback Simulator": "回滚模拟器",
    "Pricing Rollback %": "定价回滚 %",
    "Transfer Rollback %": "调拨回滚 %",
    "Replenishment Rollback %": "补货回滚 %",
    "GP Lift Retained": "保留毛利提升",
    "Revenue Retained": "保留营收",
    "Cancelled Transfers": "取消调拨",
    "Unit Recovery": "销量恢复",
    "Strategy": "策略",
    # AI 分析助手
    "AI Analyst (Local Deterministic)": "AI 分析助手（本地确定式）",
    "Suggested Questions:": "推荐问题：",
    "Ask a question": "提问",
    "e.g., Where is the largest modeled margin opportunity?": "例如：模型显示最大的毛利机会在哪里？",
    "Analyze": "分析",
    "Answer:": "回答：",
    "Intent:": "意图：",
    "Provider:": "来源：",
    "Active Filters:": "当前筛选：",
    "Evidence Used:": "使用的证据：",
    "Caveat:": "注意事项：",
    "Metric Definitions:": "指标定义：",
    # 数据与治理
    "Star Schema": "星型模型",
    "Metric Definitions": "指标定义",
    "Approval Workflow": "审批流程",
    "Production Monitoring Checklist": "生产监控清单",
    # —— 星型模型表说明 ——
    "Weekly sales transactions": "每周销售流水",
    "Weekly inventory positions and weeks of cover": "每周库存水位与可售周数",
    "Open and planned purchase orders": "未结与计划中的采购订单",
    "Inter-region transfer recommendations": "跨区域调拨建议",
    "Replenishment, stop-order, markdown actions": "补货、停单、降价等动作",
    "Price changes with reason codes": "带原因码的价格变动",
    "Joint pricing + inventory recommendations": "定价 + 库存联合建议",
    "SKU master with cost, margin, lead time": "SKU 主数据：成本、毛利、提前期",
    "Supplier lead time and MOQ": "供应商提前期与最小起订量",
    "4 BC regions": "4 个 BC 区域",
    "Retail, Trade, Fleet": "零售、批发、车队",
    "104-week calendar": "104 周日历",
    # —— 指标定义名 ——
    "Revenue": "营收",
    "Gross Profit": "毛利",
    "Gross Margin": "毛利率",
    "Inventory Turns": "库存周转",
    "Stockout Rate": "缺货率",
    "Excess Inventory Value": "过剩库存价值",
    "Modeled Lift": "模型提升",
    # —— 监控清单 ——
    "Demand forecast error": "需求预测误差",
    "Stockout-rate change": "缺货率变化",
    "Excess-inventory change": "过剩库存变化",
    "Transfer success rate": "调拨成功率",
    "Replenishment service level": "补货服务水平",
    "Inventory-turn change": "库存周转变化",
    "Holding-cost change": "持有成本变化",
    "Realized vs expected sell-through": "实际 vs 预期动销",
    "Analyst override rate": "分析师推翻率",
    "Pricing rollback rate": "定价回滚率",
    "Inventory-action rollback rate": "库存动作回滚率",
}


# ── 模板（含占位符 / HTML 内文）──
_EN_TPL = {
    "disclosure": (
        "<strong>Synthetic Data Disclosure:</strong> "
        "The public demo uses a representative synthetic sample. "
        "The workflow is designed for production catalogs containing millions of SKUs. "
        "All data is independently generated — no former-employer or external project data is used."
    ),
    "exec_summary": (
        "<strong>Executive Summary:</strong> "
        "Across {n} SKUs, the dynamic pricing model identifies {gp} in modeled gross profit lift. "
        "The largest opportunity is in <strong>{cat}</strong>. "
        "{inc} SKUs recommended for price increase, {dec} for decrease. "
        "All recommendations require human approval before rollout."
    ),
    "model_info": (
        "<strong>Model:</strong> {name} | "
        "<strong>Split:</strong> Train {train}w / Val {val}w / Test 13w (time-based, no random split) | "
        "<strong>Stockout adjustment:</strong> Applied | "
        "<strong>Leakage protection:</strong> Lag features use prior-week data only"
    ),
    "limitations": (
        "<strong>Limitations:</strong> "
        "Model predicts unconstrained demand adjusted for stockout censoring. "
        "Predictive importance is not causal elasticity. "
        "Production deployment requires controlled price experiments."
    ),
    "elasticity_distinction": (
        "<strong>Important Distinction:</strong> "
        "Predictive demand response (from ML model) ≠ Estimated causal elasticity (from log-log regression). "
        "Low-confidence estimates are shrunk toward category/global priors. "
        "Price endogeneity is mitigated by using price-test observations."
    ),
    "why_price": (
        "<strong>Why this price?</strong> {reason}. "
        "Elasticity: {elasticity:.2f} (confidence: {conf:.2f}). "
        "Guardrails: {guardrail}. "
        "<strong>Human approval required.</strong>"
    ),
    "decision_path": (
        "<strong>Decision Path:</strong> "
        "Demand → <em>{status}</em> → Pricing: <em>{pricing}</em> → "
        "Operational: <em>{inv_action}</em> → Approval: {approval}"
    ),
    "backtest_methodology": (
        "<strong>Methodology:</strong> "
        "Comparing four pricing policies over the final 13-week test period. "
        "Results are <em>modeled/simulated estimates</em>, not proven business impact. "
        "Observational backtest — causal lift requires controlled experiments."
    ),
    "approval_workflow": (
        "Data refresh → Demand forecast → Inventory classification → "
        "Pricing optimization → Transfer/replenishment evaluation → "
        "Guardrail validation → Analyst review → Manager approval → "
        "Controlled execution → Monitoring → Rollback"
    ),
    "Required": "Required",
    "Standard": "Standard",
}

_ZH_TPL = {
    "disclosure": (
        "<strong>合成数据声明：</strong>"
        "公开演示使用的是有代表性的合成样本。"
        "该工作流是为包含数百万 SKU 的生产目录设计的。"
        "所有数据均独立生成——不使用任何前雇主或外部项目数据。"
    ),
    "exec_summary": (
        "<strong>经营摘要：</strong>"
        "在 {n} 个 SKU 上，动态定价模型识别出 {gp} 的模型毛利提升空间。"
        "最大的机会出现在 <strong>{cat}</strong>。"
        "建议 {inc} 个 SKU 提价、{dec} 个降价。"
        "所有建议在上线前都需人工审批。"
    ),
    "model_info": (
        "<strong>模型：</strong>{name} | "
        "<strong>划分：</strong>训练 {train} 周 / 验证 {val} 周 / 测试 13 周（按时间划分，非随机）| "
        "<strong>缺货校正：</strong>已应用 | "
        "<strong>防泄漏：</strong>滞后特征仅使用前一周数据"
    ),
    "limitations": (
        "<strong>局限：</strong>"
        "模型预测的是经缺货删失校正后的无约束需求。"
        "预测重要性不等于因果弹性。"
        "生产上线需要受控的价格实验。"
    ),
    "elasticity_distinction": (
        "<strong>重要区分：</strong>"
        "预测性需求响应（来自 ML 模型）≠ 估计的因果弹性（来自 log-log 回归）。"
        "低置信度的估计会向品类/全局先验收缩。"
        "通过使用价格测试观测来缓解价格内生性。"
    ),
    "why_price": (
        "<strong>为何是这个价格？</strong>{reason}。"
        "弹性：{elasticity:.2f}（置信度：{conf:.2f}）。"
        "护栏：{guardrail}。"
        "<strong>需人工审批。</strong>"
    ),
    "decision_path": (
        "<strong>决策路径：</strong>"
        "需求 → <em>{status}</em> → 定价：<em>{pricing}</em> → "
        "操作：<em>{inv_action}</em> → 审批：{approval}"
    ),
    "backtest_methodology": (
        "<strong>方法：</strong>"
        "在最后 13 周测试期内对比四种定价策略。"
        "结果为<em>模型/模拟估计</em>，并非已验证的业务影响。"
        "观测性回测——因果提升需要受控实验。"
    ),
    "approval_workflow": (
        "数据刷新 → 需求预测 → 库存分类 → "
        "定价优化 → 调拨/补货评估 → "
        "护栏校验 → 分析师复核 → 经理审批 → "
        "受控执行 → 监控 → 回滚"
    ),
    "Required": "需要",
    "Standard": "标准",
}
