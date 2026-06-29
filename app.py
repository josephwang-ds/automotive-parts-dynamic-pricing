"""汽车零配件动态定价与库存优化 — Streamlit 应用。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import i18n
from src.backtest import RollbackSimulator
from src.config import (
    CATEGORIES,
    CUSTOMER_TIERS,
    OBJECTIVES,
    REGIONS,
    SCENARIOS,
    UI_COLORS,
)
from src.utils import format_currency, format_pct

t = i18n.t
tf = i18n.tf


def t_opt(x):
    """数据值下拉：只翻译 'All'，其余保持英文（品类/区域等需与数据匹配）。"""
    return t("All") if x == "All" else x

# 页面配置
st.set_page_config(
    page_title="Parts Dynamic Pricing AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义 CSS
st.markdown(f"""
<style>
    .stApp {{ background-color: {UI_COLORS['background']}; }}
    .metric-card {{
        background: white;
        border: 1px solid {UI_COLORS['border']};
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }}
    .metric-label {{ color: {UI_COLORS['muted_text']}; font-size: 0.85rem; margin-bottom: 4px; }}
    .metric-value {{ color: {UI_COLORS['primary_navy']}; font-size: 1.5rem; font-weight: 600; }}
    .disclosure {{
        background: #EFF6FF;
        border-left: 4px solid {UI_COLORS['electric_blue']};
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 20px;
        font-size: 0.9rem;
        color: {UI_COLORS['primary_navy']};
    }}
    .callout {{
        background: white;
        border: 1px solid {UI_COLORS['border']};
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
    }}
    .status-pill {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
    }}
    .pill-green {{ background: #D1FAE5; color: {UI_COLORS['emerald']}; }}
    .pill-amber {{ background: #FEF3C7; color: {UI_COLORS['warning_amber']}; }}
    .pill-red {{ background: #FEE2E2; color: {UI_COLORS['risk_red']}; }}
    .section-title {{
        color: {UI_COLORS['primary_navy']};
        font-size: 1.1rem;
        font-weight: 600;
        margin: 20px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid {UI_COLORS['electric_blue']};
    }}
</style>
""", unsafe_allow_html=True)


def section_title(key: str):
    """渲染带翻译的小节标题。"""
    st.markdown(f'<div class="section-title">{t(key)}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: str = ""):
    """渲染指标卡片（label 自动翻译）。"""
    delta_html = f'<div style="color:{UI_COLORS["emerald"]};font-size:0.8rem;">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{t(label)}</div>'
        f'<div class="metric-value">{value}</div>{delta_html}</div>',
        unsafe_allow_html=True,
    )


def status_pill(text: str, level: str = "green"):
    """渲染状态标签。"""
    cls = {"green": "pill-green", "amber": "pill-amber", "red": "pill-red"}.get(level, "pill-green")
    return f'<span class="status-pill {cls}">{text}</span>'


def synthetic_disclosure():
    """合成数据声明。"""
    st.markdown(f'<div class="disclosure">{tf("disclosure")}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_app_state():
    """加载应用状态（部署路径优先，无 sklearn）。"""
    from src.runtime_stubs import deploy_artifacts_ready, load_deploy_state

    if deploy_artifacts_ready():
        return load_deploy_state()

    from src.pipeline import get_or_run_pipeline

    return get_or_run_pipeline()


# 每个定价目标对应的排序字段（推荐如何排名）
_OBJECTIVE_SORT_COL = {
    "maximize_gross_profit": "gross_profit_lift",
    "maximize_revenue": "projected_revenue",
    "reduce_excess_inventory": "excess_inventory_value",
    "balanced": "gross_profit_lift",
}


def apply_policy(recs, objective, margin_floor, max_move_frac, confidence):
    """把侧边栏的策略参数真正作用到推荐表：护栏过滤 + 目标排序。

    返回「符合当前策略」的可执行推荐子集，因此拖动任一滑块都会即时改变
    页面上的 SKU 数、汇总指标、图表和排行榜。
    """
    df = recs
    # 空值（无该项分数的行）不参与过滤——阈值=最小时即「不过滤」
    if "elasticity_confidence" in df.columns:
        col = df["elasticity_confidence"]
        df = df[(col >= confidence) | col.isna()]
    if "gross_margin_pct" in df.columns:
        col = df["gross_margin_pct"]
        df = df[(col >= margin_floor) | col.isna()]
    if "price_change_pct" in df.columns:
        col = df["price_change_pct"]
        df = df[(col.abs() <= max_move_frac + 1e-9) | col.isna()]
    sort_col = _OBJECTIVE_SORT_COL.get(objective, "gross_profit_lift")
    if sort_col in df.columns and len(df):
        df = df.sort_values(sort_col, ascending=False)
    return df


# 使用「数据筛选」的页面（依赖推荐表 / 销售明细）
DATA_FILTER_PAGES = {
    "Executive Command Center", "SKU Decision Workbench",
    "Inventory Control Tower", "Backtest & Rollback",
}
# 使用「定价策略」的页面（依赖按护栏过滤后的推荐表）
PRICING_POLICY_PAGES = DATA_FILTER_PAGES


def _default_filters():
    return {
        "region": "All", "category": "All", "customer_tier": "All",
        "objective": "balanced", "scenario": "Recommended",
        "margin_floor": 0.15, "max_move": 10, "confidence_threshold": 0.4,
    }


def apply_sidebar_filters(recs, sales, page):
    """侧边栏分层：仅在相关页面显示对应控件，避免在无关页面「调了没反应」。

    - 数据筛选 / 定价策略 都放在可折叠区块里。
    - 当前页面用不到的控件直接不渲染，并给出一句说明。
    """
    use_filters = page in DATA_FILTER_PAGES
    use_policy = page in PRICING_POLICY_PAGES

    if not use_filters and not use_policy:
        st.sidebar.caption(t("This page shows global model/data — sidebar filters don't apply here."))
        return recs, sales, _default_filters()

    # ── 数据筛选（可折叠）──
    filters = {"region": "All", "category": "All", "customer_tier": "All"}
    if use_filters:
        with st.sidebar.expander(t("Data Filters"), expanded=True):
            st.caption(t("Choose which slice of the catalog to view."))
            filters["region"] = st.selectbox(t("Region"), ["All"] + REGIONS, format_func=t_opt,
                                             help=t("Show only this sales region."))
            filters["category"] = st.selectbox(t("Category"), ["All"] + CATEGORIES, format_func=t_opt,
                                               help=t("Show only this product category."))
            filters["customer_tier"] = st.selectbox(t("Customer Tier"), ["All"] + CUSTOMER_TIERS,
                                                    format_func=t_opt,
                                                    help=t("Retail / Trade / Fleet pricing segment."))

    mask = pd.Series(True, index=recs.index)
    sales_mask = pd.Series(True, index=sales.index) if len(sales) else pd.Series(dtype=bool)
    for k, v in filters.items():
        if v != "All":
            if k in recs.columns:
                mask &= recs[k] == v
            if k in sales.columns and len(sales):
                sales_mask &= sales[k] == v
    data_recs = recs[mask]
    filtered_sales = sales[sales_mask] if len(sales) else sales

    # ── 定价策略（可折叠；情景 = 一键预设，下面可微调）──
    objective, scenario = "balanced", "Recommended"
    margin_floor, max_move, confidence_threshold = 0.15, 10, 0.4
    if use_policy:
        with st.sidebar.expander(t("Pricing Policy"), expanded=True):
            st.caption(t("Adjust guardrails and objective — the recommendation set updates live."))
            scenario_keys = list(SCENARIOS.keys())
            scenario_default = scenario_keys.index("Recommended") if "Recommended" in scenario_keys else 0
            scenario = st.selectbox(
                t("Scenario"), scenario_keys, index=scenario_default, format_func=t,
                help=t("Preset policy bundle — sets the objective, confidence and max-move below. "
                       "You can still fine-tune them."),
            )
            cfg = SCENARIOS[scenario]
            obj_keys = list(OBJECTIVES.keys())
            obj_default = obj_keys.index(cfg["objective"]) if cfg.get("objective") in obj_keys else len(obj_keys) - 1
            objective = st.selectbox(
                t("Pricing Objective"), obj_keys, index=obj_default,
                format_func=lambda x: t(OBJECTIVES[x]),
                help=t("How recommendations are ranked: by profit, revenue, or excess-inventory reduction."),
                key=f"obj_{scenario}",
            )
            confidence_threshold = st.slider(
                t("Confidence Threshold"), 0.0, 1.0, float(cfg.get("confidence_threshold", 0.4)), 0.05,
                help=t("Keep only recommendations whose elasticity confidence is at least this. "
                       "Higher = safer, fewer SKUs."),
                key=f"ct_{scenario}",
            )
            margin_floor = st.slider(
                t("Margin Floor"), 0.05, 0.35, 0.15, 0.01,
                help=t("Drop recommendations whose projected gross margin falls below this floor."),
                key=f"mf_{scenario}",
            )
            max_move = st.slider(
                t("Max Price Move %"), 1, 15, int(round(cfg.get("max_price_move_pct", 0.10) * 100)),
                help=t("Keep only recommendations within this price-change limit; "
                       "larger moves need manual review."),
                key=f"mm_{scenario}",
            )

        actionable = apply_policy(data_recs, objective, margin_floor, max_move / 100, confidence_threshold)
        st.sidebar.caption(tf("policy_pass", n=f"{len(actionable):,}", total=f"{len(data_recs):,}"))
    else:
        actionable = data_recs

    return actionable, filtered_sales, {
        **filters, "objective": objective, "scenario": scenario,
        "margin_floor": margin_floor, "max_move": max_move,
        "confidence_threshold": confidence_threshold,
    }


# ── 页面 ──

def page_executive(recs, sales, inv, bt, filters):
    """Executive Command Center。"""
    section_title("Executive Command Center")

    total_rev = recs["current_revenue"].sum()
    total_gp = recs["current_gross_profit"].sum()
    gp_lift = recs["gross_profit_lift"].sum()
    inv_val = inv.get("total_inventory_value", 0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("SKUs in Scope", f"{len(recs):,}")
    with c2:
        metric_card("Current Revenue", format_currency(total_rev))
    with c3:
        metric_card("Current Gross Profit", format_currency(total_gp))
    with c4:
        metric_card("Modeled GP Lift", format_currency(gp_lift), t("Simulated estimate"))

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        metric_card("Inventory Value", format_currency(inv_val))
    with c6:
        metric_card("Avg Inventory Turns", f"{inv.get('avg_inventory_turns', 0):.1f}")
    with c7:
        metric_card("Excess Inventory", format_currency(inv.get("excess_inventory_value", 0)))
    with c8:
        metric_card("Stockout-Risk SKUs", f"{inv.get('stockout_risk_skus', 0):,}")

    # Executive summary
    top_cat = recs.groupby("category")["gross_profit_lift"].sum().idxmax() if len(recs) > 0 else "N/A"
    action_dist = recs["recommendation_action"].value_counts().to_dict() if len(recs) > 0 else {}
    increases = action_dist.get("Increase", 0)
    decreases = action_dist.get("Decrease", 0)
    st.markdown(
        f'<div class="callout">'
        + tf("exec_summary", n=f"{len(recs):,}", gp=format_currency(gp_lift),
             cat=top_cat, inc=increases, dec=decreases)
        + '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if len(recs) > 0:
            fig = px.bar(
                recs.groupby("category")["gross_profit_lift"].sum().reset_index(),
                x="category", y="gross_profit_lift",
                title=t("Modeled Opportunity by Category"),
                color_discrete_sequence=[UI_COLORS["electric_blue"]],
            )
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, width='stretch')

    with col2:
        if len(recs) > 0:
            fig = px.pie(
                recs, names="recommendation_action",
                title=t("Recommendation Action Distribution"),
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, width='stretch')

    col3, col4 = st.columns(2)
    with col3:
        if len(recs) > 0:
            fig = px.bar(
                recs.groupby("region")["gross_profit_lift"].sum().reset_index(),
                x="region", y="gross_profit_lift",
                title=t("Opportunity by Region"),
                color_discrete_sequence=[UI_COLORS["emerald"]],
            )
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, width='stretch')

    with col4:
        top_candidates = recs.nlargest(10, "gross_profit_lift")[
            ["sku_id", "category", "gross_profit_lift", "recommendation_action"]
        ] if len(recs) > 0 else pd.DataFrame()
        st.markdown(f"**{t('Top Approval Candidates')}**")
        st.dataframe(top_candidates, width='stretch', hide_index=True)


def page_pricing_studio(recs, state, filters):
    """SKU Decision Workbench（定价 + 库存联合决策）。"""
    section_title("SKU Decision Workbench")

    if recs.empty:
        st.warning(t("No recommendations for current filters."))
        return

    sku_ids = recs["sku_id"].unique().tolist()
    selected_sku = st.selectbox(t("Select SKU"), sku_ids)

    rec = recs[recs["sku_id"] == selected_sku].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Current Price", format_currency(rec["current_price"]))
    with c2:
        metric_card("Recommended Price", format_currency(rec["recommended_price"]))
    with c3:
        metric_card("Price Change", format_pct(rec["price_change_pct"]))
    with c4:
        metric_card("Action", rec["recommendation_action"])

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        metric_card("Predicted Units (Current)", f"{rec['predicted_current_units']:.1f}")
    with c6:
        metric_card("Predicted Units (Recommended)", f"{rec['predicted_recommended_units']:.1f}")
    with c7:
        metric_card("GP Lift", format_currency(rec["gross_profit_lift"]))
    with c8:
        metric_card("Margin", format_pct(rec["gross_margin_pct"]))

    st.markdown(
        '<div class="callout">'
        + tf("why_price",
             reason=rec.get("pricing_reason_code", rec.get("reason_code", "")),
             elasticity=rec["elasticity"],
             conf=rec.get("elasticity_confidence", 0),
             guardrail=rec.get("guardrail_triggered") or "None")
        + '</div>',
        unsafe_allow_html=True,
    )

    # 库存决策区
    if "inventory_status" in rec.index:
        section_title("Inventory Decision")
        ic1, ic2, ic3, ic4 = st.columns(4)
        with ic1:
            metric_card("Inventory Status", rec.get("inventory_status", "N/A"))
        with ic2:
            metric_card("Inventory Action", rec.get("inventory_action", "N/A"))
        with ic3:
            metric_card("Weeks of Cover", f"{rec.get('available_weeks_of_cover', 0):.1f}")
        with ic4:
            metric_card("Joint Confidence", rec.get("joint_confidence", "N/A"))

        ic5, ic6, ic7, ic8 = st.columns(4)
        with ic5:
            metric_card("On-Hand", f"{rec.get('on_hand_inventory', 0):,.0f}")
        with ic6:
            metric_card("On-Order", f"{rec.get('on_order_inventory', 0):,.0f}")
        with ic7:
            metric_card("Stockout Prob.", format_pct(rec.get("stockout_probability", 0)))
        with ic8:
            metric_card("Reorder Qty", f"{rec.get('recommended_order_quantity', 0):,.0f}")

        st.markdown(
            '<div class="callout">'
            + tf("decision_path",
                 status=rec.get("inventory_status"),
                 pricing=rec.get("pricing_action", rec.get("recommendation_action")),
                 inv_action=rec.get("inventory_action"),
                 approval=tf("Required") if rec.get("manual_review_required") else tf("Standard"))
            + '</div>',
            unsafe_allow_html=True,
        )

    # 候选价格模拟图
    if state.demand_model and state.sales is not None:
        from src.optimizer import PriceOptimizer

        latest = state.sales.sort_values("week_num").groupby(
            ["sku_id", "region", "customer_tier"]
        ).last().reset_index()
        row_data = latest[
            (latest["sku_id"] == selected_sku)
            & (latest["region"] == rec["region"])
            & (latest["customer_tier"] == rec["customer_tier"])
        ]
        if len(row_data) > 0:
            opt = PriceOptimizer(demand_model=state.demand_model)
            el_info = state.elasticity.get_elasticity(
                rec["category"], rec["region"], rec["customer_tier"]
            ) if state.elasticity else {}
            candidates = opt.generate_candidate_prices(rec["current_price"])
            sims = [opt.simulate_candidate(row_data.iloc[0], cp, el_info) for cp in candidates]
            sim_df = pd.DataFrame(sims)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.line(sim_df, x="candidate_price", y="predicted_units",
                              title=t("Price vs Predicted Units"),
                              color_discrete_sequence=[UI_COLORS["electric_blue"]])
                fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig, width='stretch')
            with col2:
                fig = px.line(sim_df, x="candidate_price", y="gross_profit",
                              title=t("Price vs Gross Profit"),
                              color_discrete_sequence=[UI_COLORS["emerald"]])
                fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig, width='stretch')


def page_inventory_control_tower(state, inv, recs, filters):
    """Inventory Control Tower。"""
    section_title("Inventory Control Tower")

    transfers = getattr(state, "transfers", pd.DataFrame())
    inv_snap = inv.get("latest_snapshot", pd.DataFrame())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Inventory Value", format_currency(inv.get("total_inventory_value", 0)))
    with c2:
        metric_card("Excess Inventory", format_currency(inv.get("excess_inventory_value", 0)))
    with c3:
        metric_card("Stockout-Risk SKUs", f"{inv.get('stockout_risk_skus', 0):,}")
    with c4:
        metric_card("Est. Lost Sales", f"{inv.get('estimated_lost_sales', 0):,.0f}")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        metric_card("Avg Inventory Turns", f"{inv.get('avg_inventory_turns', 0):.1f}")
    with c6:
        metric_card("Avg Weeks of Cover", f"{inv.get('avg_weeks_of_cover', 0):.1f}")
    with c7:
        metric_card("Transfer Opportunities", f"{len(transfers):,}")
    with c8:
        rep_count = len(recs[recs.get("inventory_action", pd.Series()).isin(["REPLENISH", "EXPEDITE_ORDER"])]) if "inventory_action" in recs.columns else 0
        metric_card("Replenishment Candidates", f"{rep_count:,}")

    col1, col2 = st.columns(2)
    with col1:
        if not inv_snap.empty and "inventory_status" in inv_snap.columns:
            fig = px.bar(
                inv_snap["inventory_status"].value_counts().reset_index(),
                x="count", y="inventory_status", orientation="h",
                title=t("Inventory Health Distribution"),
                color_discrete_sequence=[UI_COLORS["electric_blue"]],
            )
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, width='stretch')
    with col2:
        if not inv_snap.empty and "inventory_status" in inv_snap.columns:
            val_by_status = inv_snap.groupby("inventory_status")["inventory_value"].sum().reset_index()
            fig = px.bar(val_by_status, x="inventory_status", y="inventory_value",
                         title=t("Inventory Value by Status"),
                         color_discrete_sequence=[UI_COLORS["warning_amber"]])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, width='stretch')

    col3, col4 = st.columns(2)
    with col3:
        if not inv_snap.empty and "available_weeks_of_cover" in inv_snap.columns:
            fig = px.histogram(inv_snap, x="available_weeks_of_cover", nbins=30,
                               title=t("Weeks-of-Cover Distribution"),
                               color_discrete_sequence=[UI_COLORS["electric_blue"]])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, width='stretch')
    with col4:
        if not inv_snap.empty and "inventory_turns" in inv_snap.columns:
            fig = px.scatter(inv_snap, x="inventory_turns", y="unit_cost",
                             size="excess_inventory_value", color="inventory_status",
                             title=t("Margin vs Inventory Turns"),
                             opacity=0.6)
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, width='stretch')

    if "inventory_action" in recs.columns:
        action_dist = recs.drop_duplicates(["sku_id", "region"])["inventory_action"].value_counts()
        fig = px.pie(values=action_dist.values, names=action_dist.index,
                     title=t("Inventory Action Distribution"))
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, width='stretch')

    # 决策矩阵
    if "inventory_status" in recs.columns and "pricing_action" in recs.columns:
        matrix = recs.drop_duplicates(["sku_id", "region"]).groupby(
            ["inventory_status", "pricing_action"]
        ).size().reset_index(name="count")
        if not matrix.empty:
            st.markdown(f"**{t('Pricing vs Inventory Action Matrix')}**")
            st.dataframe(matrix.head(50), width='stretch', hide_index=True)

    # 数据表
    st.markdown(f"**{t('Top Excess Inventory Candidates')}**")
    if not inv_snap.empty:
        excess = inv_snap[inv_snap.get("is_excess", inv_snap.get("inventory_status", "") == "OVERSTOCKED")]
        if len(excess) == 0 and "inventory_status" in inv_snap.columns:
            excess = inv_snap[inv_snap["inventory_status"].isin(["OVERSTOCKED", "SLOW_MOVING"])]
        cols = [c for c in ["sku_id", "category", "region", "inventory_status", "excess_inventory_value",
                            "available_weeks_of_cover"] if c in excess.columns]
        st.dataframe(excess.nlargest(50, "excess_inventory_value")[cols] if len(excess) > 0 else pd.DataFrame(),
                     width='stretch', hide_index=True)

    if not transfers.empty:
        st.markdown(f"**{t('Transfer Recommendations')}**")
        st.dataframe(transfers.head(50), width='stretch', hide_index=True)

    if "inventory_action" in recs.columns:
        manual = recs[recs.get("manual_review_required", False) == True]  # noqa: E712
        if len(manual) > 0:
            st.markdown(f"**{t('Manual Review Queue')}**")
            show_cols = [c for c in ["sku_id", "region", "customer_tier", "pricing_action",
                        "inventory_action", "joint_confidence", "inventory_reason_code"] if c in manual.columns]
            st.dataframe(manual[show_cols].head(50), width='stretch', hide_index=True)


def page_backtest(state, recs, filters):
    """Backtest & Rollback。"""
    section_title("Backtest & Rollback Simulator")

    bt = state.backtest_results
    comparison = bt.get("strategy_comparison", pd.DataFrame())

    if not comparison.empty:
        st.markdown(
            f'<div class="callout">{tf("backtest_methodology")}</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(comparison, width='stretch')

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                comparison.reset_index(), x="index", y="total_gross_profit",
                title=t("Gross Profit Comparison by Strategy"),
                labels={"index": t("Strategy")},
                color_discrete_sequence=[UI_COLORS["emerald"]],
            )
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, width='stretch')

        with col2:
            fig = px.bar(
                comparison.reset_index(), x="index", y="total_revenue",
                title=t("Revenue Comparison by Strategy"),
                labels={"index": t("Strategy")},
                color_discrete_sequence=[UI_COLORS["electric_blue"]],
            )
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, width='stretch')

    # Rollback
    st.markdown(f"### {t('Rollback Simulator')}")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        pricing_rb = st.slider(t("Pricing Rollback %"), 0, 100, 0, 5)
    with col_r2:
        transfer_rb = st.slider(t("Transfer Rollback %"), 0, 100, 0, 5)
    with col_r3:
        replenish_rb = st.slider(t("Replenishment Rollback %"), 0, 100, 0, 5)

    transfers = getattr(state, "transfers", pd.DataFrame())
    simulator = RollbackSimulator(recs, transfers)
    result = simulator.simulate_rollback(
        pricing_rollback_pct=pricing_rb / 100,
        transfer_rollback_pct=transfer_rb / 100,
        replenishment_rollback_pct=replenish_rb / 100,
    )

    summary = result["summary"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("GP Lift Retained", format_currency(summary["gross_profit_lift_retained"]))
    with c2:
        metric_card("Revenue Retained", format_currency(summary["revenue_retained"]))
    with c3:
        metric_card("Cancelled Transfers", f"{summary.get('cancelled_transfers', 0):,}")
    with c4:
        metric_card("Unit Recovery", f"{summary.get('unit_recovery_pct', 0)*100:.0f}%")

    st.dataframe(result["audit_table"].head(20), width='stretch', hide_index=True)


def render_model_quality_fold(state):
    """折叠：需求模型关键质量指标（从原 Demand Model 页精简而来）。"""
    metrics = {}
    if state.demand_model:
        metrics = state.demand_model.metrics
    elif state.model_results:
        metrics = state.model_results.get("hgb", {}).get("metrics", {})
    test_m = metrics.get("test", {})
    if not test_m:
        return
    with st.expander(t("Demand model quality")):
        st.caption(tf("model_info",
                      name=metrics.get("model_name", "HistGradientBoosting"),
                      train=metrics.get("train_weeks", 78),
                      val=metrics.get("val_weeks", 13)))
        m1, m2, m3 = st.columns(3)
        with m1:
            metric_card("MAE", f"{test_m.get('MAE', 0):.2f}")
        with m2:
            metric_card("WAPE", f"{test_m.get('WAPE', 0):.3f}")
        with m3:
            metric_card("Baseline Improvement", f"{test_m.get('WAPE_improvement', 0)*100:.1f}%")


def render_elasticity_fold(state):
    """折叠：价格弹性概览（从原 Elasticity 页精简为一张表，不再用热力图）。"""
    el_df = getattr(state, "elasticity_df", None)
    if el_df is None or el_df.empty:
        return
    cat_tier = el_df[el_df["estimation_level"] == "category_tier"]
    if cat_tier.empty:
        return
    with st.expander(t("Price elasticity context")):
        st.caption(t("Estimated price elasticity by category and customer tier — more negative means more price-sensitive."))
        pivot = cat_tier.pivot_table(index="category", columns="customer_tier",
                                     values="estimated_elasticity", aggfunc="first")
        st.dataframe(pivot.round(3), width='stretch')


def render_method_fold():
    """折叠：方法、指标定义与数据说明（从原 Governance 页精简而来）。"""
    with st.expander(t("Method, metrics & data")):
        metrics_def = {
            "Revenue": "realized_price × units_sold",
            "Gross Profit": "revenue − COGS (unit_cost × units_sold)",
            "Inventory Turns": "annual_units_sold / average_inventory",
            "Weeks of Cover": "ending_inventory / weekly_demand",
            "Modeled Lift": "simulated GP difference between dynamic and current pricing",
        }
        for m, d in metrics_def.items():
            st.markdown(f"- **{t(m)}:** {d}")
        st.caption(t("All data is synthetic and deterministic; figures are modeled estimates, not live results."))


def main():
    # 语言开关（最先读取，保证整页文案一致）
    i18n.set_language(st.session_state.get("lang", "en"))
    st.sidebar.selectbox(
        t("Language"), list(i18n.LANGUAGES.keys()),
        format_func=lambda k: i18n.LANGUAGES[k], key="lang",
    )
    i18n.set_language(st.session_state.get("lang", "en"))
    st.sidebar.markdown("---")

    st.title(t("Parts Dynamic Pricing & Inventory AI"))
    synthetic_disclosure()

    try:
        with st.spinner(t("Loading data...")):
            state = load_app_state()
    except Exception as exc:
        st.error(t("App failed to start. Ensure the repo includes the outputs/ and data/ deploy files."))
        st.exception(exc)
        st.stop()

    # 顶部 Tab 导航：四段式完整故事链，一键切换、滚动流畅（无需侧栏逐页点选）。
    recs, sales, filters = apply_sidebar_filters(
        state.recommendations, state.sales, "Executive Command Center"
    )

    tab_exec, tab_sku, tab_inv, tab_bt = st.tabs([
        t("Executive Command Center"),
        t("SKU Decision Workbench"),
        t("Inventory Control Tower"),
        t("Backtest & Rollback"),
    ])

    with tab_exec:
        page_executive(recs, sales, state.inventory_analysis,
                       state.backtest_results, filters)
        render_model_quality_fold(state)
        render_method_fold()

    with tab_sku:
        page_pricing_studio(recs, state, filters)
        render_elasticity_fold(state)

    with tab_inv:
        page_inventory_control_tower(state, state.inventory_analysis, recs, filters)

    with tab_bt:
        page_backtest(state, recs, filters)


# Streamlit 每次运行都会执行整个脚本，必须直接调用 main()
main()
