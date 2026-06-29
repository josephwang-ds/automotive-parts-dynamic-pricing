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

HERO_HEADLINE = "Automotive Parts Dynamic Pricing & Inventory Optimization"
HERO_SUB = ("Connect pricing, demand forecasting, inventory health, and replenishment "
            "decisions in one explainable decision-support workflow.")


def _kpi_row(cards):
    """渲染一行 KPI 卡片。cards = [(label, value, delta), ...]"""
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        label, value = card[0], card[1]
        delta = card[2] if len(card) > 2 else ""
        with col:
            metric_card(label, value, delta)


def _plot(fig):
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, width="stretch")


# ════════════════════════ PAGE 1 — EXECUTIVE OVERVIEW ════════════════════════
def page_overview(recs, sales, state, filters):
    inv = state.inventory_analysis
    st.markdown(f"### {t(HERO_HEADLINE)}")
    st.caption(t(HERO_SUB))

    total_rev = recs["current_revenue"].sum() if len(recs) else 0
    total_gp = recs["current_gross_profit"].sum() if len(recs) else 0
    gp_margin = (total_gp / total_rev) if total_rev else 0
    gp_lift = recs["gross_profit_lift"].sum() if len(recs) else 0
    inv_val = inv.get("total_inventory_value", 0)
    turns = inv.get("avg_inventory_turns", 0)
    excess = inv.get("excess_inventory_value", 0)
    snap = inv.get("latest_snapshot", pd.DataFrame())
    stockout_rate = 0.0
    if not snap.empty and "inventory_status" in snap.columns:
        stockout_rate = float(snap["inventory_status"].isin(["STOCKOUT", "STOCKOUT_RISK"]).mean())
    price_ops = int(recs["recommendation_action"].isin(["Increase", "Decrease"]).sum()) if len(recs) else 0
    high_risk = int(recs["inventory_status"].isin(["STOCKOUT", "STOCKOUT_RISK"]).sum()) if "inventory_status" in recs.columns else 0
    awaiting = int(recs.get("manual_review_required", pd.Series([], dtype=bool)).sum())

    section_title("Headline KPIs")
    _kpi_row([
        ("Total Revenue", format_currency(total_rev), t("Latest period")),
        ("Gross Margin $", format_currency(total_gp)),
        ("Gross Margin %", format_pct(gp_margin)),
        ("Inventory Value", format_currency(inv_val)),
        ("Inventory Turnover", f"{turns:.1f}"),
    ])
    _kpi_row([
        ("Stockout Rate", format_pct(stockout_rate)),
        ("Excess Inventory Value", format_currency(excess)),
        ("Expected Margin Opportunity", format_currency(gp_lift), t("Modeled")),
        ("Pricing Opportunities", f"{price_ops:,}"),
        ("Recommendations Awaiting Review", f"{awaiting:,}"),
    ])

    st.markdown('<div class="callout">' + t(
        "The system helps pricing, category, and inventory teams identify margin improvement "
        "opportunities, stockout risks, excess inventory, competitive price gaps, and "
        "replenishment and transfer needs — each with a reason code, confidence level, and a "
        "recommended operational action.") + '</div>', unsafe_allow_html=True)

    if recs.empty:
        st.warning(t("No recommendations for current filters."))
        return

    col1, col2 = st.columns(2)
    with col1:
        d = recs.groupby("category")["gross_profit_lift"].sum().reset_index()
        _plot(px.bar(d, x="category", y="gross_profit_lift",
                     title=t("Modeled Margin Opportunity by Category"),
                     color_discrete_sequence=[UI_COLORS["electric_blue"]]))
    with col2:
        _plot(px.pie(recs, names="recommendation_action",
                     title=t("Recommended Actions by Type"),
                     color_discrete_sequence=px.colors.qualitative.Set2))

    col3, col4 = st.columns(2)
    with col3:
        d = recs.groupby("region")["gross_profit_lift"].sum().reset_index()
        _plot(px.bar(d, x="region", y="gross_profit_lift",
                     title=t("Opportunity by Region"),
                     color_discrete_sequence=[UI_COLORS["emerald"]]))
    with col4:
        st.markdown(f"**{t('Top Margin Opportunities')}**")
        top = recs.nlargest(10, "gross_profit_lift")[
            ["sku_id", "category", "region", "gross_profit_lift", "recommendation_action"]]
        st.dataframe(top, width="stretch", hide_index=True)


# ════════════════════════ PAGE 2 — PRICING OPPORTUNITIES ═════════════════════
def page_pricing_opportunities(recs, sales, state, filters):
    section_title("Pricing Opportunities")
    if recs.empty:
        st.warning(t("No recommendations for current filters."))
        return

    movers = recs[recs["recommendation_action"].isin(["Increase", "Decrease"])]
    _kpi_row([
        ("Pricing Opportunities", f"{len(movers):,}"),
        ("Avg Recommended Move", format_pct(movers["price_change_pct"].mean() if len(movers) else 0)),
        ("Modeled Margin Lift", format_currency(recs["gross_profit_lift"].sum())),
        ("High-Confidence Share", format_pct((recs["joint_confidence"] == "HIGH").mean())),
    ])

    col1, col2 = st.columns(2)
    with col1:
        top = movers.reindex(movers["gross_profit_lift"].abs().sort_values(ascending=False).index).head(15)
        if not top.empty:
            d = top[["sku_id", "current_price", "recommended_price"]].melt(
                id_vars="sku_id", var_name="type", value_name="price")
            _plot(px.bar(d, x="sku_id", y="price", color="type", barmode="group",
                         title=t("Current vs Recommended Price (top movers)")))
    with col2:
        _plot(px.histogram(recs, x="price_change_pct", nbins=40,
                           title=t("Price-Change Distribution"),
                           color_discrete_sequence=[UI_COLORS["electric_blue"]]))

    col3, col4 = st.columns(2)
    with col3:
        if "competitor_price_index" in sales.columns and len(sales):
            comp = sales.groupby("category")["competitor_price_index"].mean().reset_index()
            fig = px.bar(comp, x="category", y="competitor_price_index",
                         title=t("Competitor Price Index by Category (1.0 = parity)"),
                         color_discrete_sequence=[UI_COLORS["warning_amber"]])
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
            _plot(fig)
    with col4:
        d = recs.copy()
        d["unit_impact"] = d["predicted_recommended_units"] - d["predicted_current_units"]
        _plot(px.scatter(d, x="unit_impact", y="gross_profit_lift", color="recommendation_action",
                         title=t("Margin Lift vs Unit Impact"), opacity=0.5))

    st.markdown(f"**{t('High-Confidence Pricing Opportunities')}**")
    hi = recs[(recs["joint_confidence"] == "HIGH")
              & (recs["recommendation_action"].isin(["Increase", "Decrease"]))].nlargest(50, "gross_profit_lift")
    cols = [c for c in ["sku_id", "category", "region", "customer_tier", "current_price",
            "recommended_price", "price_change_pct", "gross_profit_lift", "gross_margin_pct",
            "reason_code"] if c in hi.columns]
    st.dataframe(hi[cols], width="stretch", hide_index=True)


# ════════════════════════════ PAGE 3 — INVENTORY RISK ════════════════════════
def page_inventory_risk(recs, state, filters):
    section_title("Inventory Risk")
    inv = state.inventory_analysis
    snap = inv.get("latest_snapshot", pd.DataFrame())
    transfers = getattr(state, "transfers", pd.DataFrame())

    _kpi_row([
        ("Inventory Value", format_currency(inv.get("total_inventory_value", 0))),
        ("Excess Inventory Value", format_currency(inv.get("excess_inventory_value", 0))),
        ("Stockout-Risk SKUs", f"{inv.get('stockout_risk_skus', 0):,}"),
        ("Est. Lost Sales", f"{inv.get('estimated_lost_sales', 0):,.0f}"),
        ("Avg Weeks of Cover", f"{inv.get('avg_weeks_of_cover', 0):.1f}"),
    ])

    col1, col2 = st.columns(2)
    with col1:
        if not snap.empty and "inventory_status" in snap.columns:
            d = snap["inventory_status"].value_counts().reset_index()
            d.columns = ["inventory_status", "count"]
            _plot(px.bar(d, x="count", y="inventory_status", orientation="h",
                         title=t("Inventory Health Distribution"),
                         color_discrete_sequence=[UI_COLORS["electric_blue"]]))
    with col2:
        if not recs.empty and {"region", "category", "inventory_status"}.issubset(recs.columns):
            r = recs.drop_duplicates(["sku_id", "region"]).copy()
            r["at_risk"] = r["inventory_status"].isin(["STOCKOUT", "STOCKOUT_RISK"]).astype(int)
            piv = r.pivot_table(index="category", columns="region", values="at_risk", aggfunc="mean")
            if not piv.empty:
                _plot(px.imshow(piv, title=t("Stockout-Risk Heatmap (Category x Region)"),
                                color_continuous_scale="Reds", aspect="auto"))

    col3, col4 = st.columns(2)
    with col3:
        if not snap.empty and "available_weeks_of_cover" in snap.columns:
            _plot(px.histogram(snap, x="available_weeks_of_cover", nbins=30,
                               title=t("Weeks-of-Cover Distribution"),
                               color_discrete_sequence=[UI_COLORS["electric_blue"]]))
    with col4:
        if not recs.empty:
            d = recs.groupby("category")["excess_inventory_value"].sum().reset_index()
            _plot(px.bar(d, x="category", y="excess_inventory_value",
                         title=t("Excess Inventory by Category"),
                         color_discrete_sequence=[UI_COLORS["warning_amber"]]))

    if not recs.empty and "inventory_action" in recs.columns:
        d = recs["inventory_action"].value_counts().reset_index()
        d.columns = ["inventory_action", "count"]
        fig = px.bar(d, x="inventory_action", y="count",
                     title=t("Recommended Inventory Actions"),
                     color_discrete_sequence=[UI_COLORS["emerald"]])
        fig.update_layout(xaxis_tickangle=-30)
        _plot(fig)

    st.markdown(f"**{t('Warehouse-to-Store Transfer Recommendations')}**")
    if transfers is not None and len(transfers):
        cols = [c for c in ["sku_id", "category", "source_region", "destination_region",
                "transfer_quantity", "net_transfer_value", "transfer_reason"] if c in transfers.columns]
        st.dataframe(transfers.sort_values("net_transfer_value", ascending=False)[cols].head(50),
                     width="stretch", hide_index=True)
    else:
        st.caption(t("No transfer opportunities under current filters."))

    st.markdown(f"**{t('Top Excess / Aged Inventory')}**")
    if not recs.empty and "inventory_status" in recs.columns:
        aged = recs[recs["inventory_status"].isin(["SLOW_MOVING", "OBSOLETE_RISK"])]
        cols = [c for c in ["sku_id", "category", "region", "inventory_status",
                "excess_inventory_value", "available_weeks_of_cover", "obsolescence_risk_score"]
                if c in aged.columns]
        if len(aged):
            st.dataframe(aged.nlargest(50, "excess_inventory_value")[cols],
                         width="stretch", hide_index=True)


# ═════════════════ PAGE 4 — SKU EXPLORER + SCENARIO SIMULATOR ════════════════
def page_sku_explorer(recs, sales, state, filters):
    section_title("SKU Explorer")
    if recs.empty:
        st.warning(t("No recommendations for current filters."))
        return
    sku = st.selectbox(t("Select SKU"), recs["sku_id"].unique().tolist())
    rec = recs[recs["sku_id"] == sku].iloc[0]

    prod = state.products[state.products["sku_id"] == sku]
    pname = prod.iloc[0]["product_name"] if len(prod) else sku
    st.markdown(f"#### {pname} — {sku}")
    if len(prod):
        p = prod.iloc[0]
        cs = st.columns(4)
        with cs[0]:
            metric_card("Category", f"{p['category']} / {p.get('subcategory', '')}")
        with cs[1]:
            metric_card("Brand Tier", str(p.get("brand_tier", "")))
        with cs[2]:
            metric_card("Lead Time (days)", f"{p.get('lead_time_days', '')}")
        with cs[3]:
            metric_card("Lifecycle", str(p.get("lifecycle_stage", "")))

    _kpi_row([
        ("Current Price", format_currency(rec["current_price"])),
        ("Recommended", format_currency(rec["recommended_price"]), rec["recommendation_action"]),
        ("Forecast Demand (wk)", f"{rec['predicted_current_units']:.1f}"),
        ("Weeks of Cover", f"{rec.get('available_weeks_of_cover', 0):.1f}"),
        ("Stockout Prob.", format_pct(rec.get("stockout_probability", 0))),
    ])
    st.markdown('<div class="callout">' + tf("why_price",
                reason=rec.get("pricing_reason_code", rec.get("reason_code", "")),
                elasticity=rec["elasticity"], conf=rec.get("elasticity_confidence", 0),
                guardrail=rec.get("guardrail_triggered") or "None") + '</div>',
                unsafe_allow_html=True)

    st.divider()
    section_title("Scenario Simulator")
    st.caption(t("Adjust the candidate price and conditions; the engine re-simulates demand, "
                 "margin, and inventory outcomes."))

    base_row = None
    if state.sales is not None and len(state.sales):
        latest = state.sales.sort_values("week_num").groupby(
            ["sku_id", "region", "customer_tier"]).last().reset_index()
        match = latest[(latest["sku_id"] == sku)
                       & (latest["region"] == rec["region"])
                       & (latest["customer_tier"] == rec["customer_tier"])]
        if len(match):
            base_row = match.iloc[0]

    if base_row is None:
        st.info(t("Scenario data unavailable for this SKU / segment."))
        return

    from src.optimizer import PriceOptimizer
    cur = float(rec["current_price"])
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        cand = st.slider(t("Candidate Price"), float(round(cur * 0.8, 2)),
                         float(round(cur * 1.2, 2)), float(round(cur, 2)))
    with sc2:
        promo = st.checkbox(t("Promotion active"), value=bool(base_row.get("promotion_flag", 0)))
    with sc3:
        avail = st.number_input(t("Available Inventory"), min_value=0,
                                value=int(base_row.get("ending_inventory", 50)))

    row = base_row.copy()
    row["ending_inventory"] = avail
    row["promotion_flag"] = 1 if promo else 0
    el_info = state.elasticity.get_elasticity(rec["category"], rec["region"], rec["customer_tier"]) if state.elasticity else {}
    opt = PriceOptimizer(demand_model=state.demand_model)
    sim = opt.simulate_candidate(row, cand, el_info)

    _kpi_row([
        ("Forecast Demand", f"{sim['predicted_units']:.1f}"),
        ("Expected Revenue", format_currency(sim["revenue"])),
        ("Expected Gross Margin", format_pct(sim["gross_margin_pct"])),
        ("Ending Inventory", f"{sim['expected_ending_inventory']:.0f}"),
        ("Weeks of Supply", f"{sim['expected_weeks_of_cover']:.1f}"),
    ])
    _kpi_row([
        ("Stockout Probability", format_pct(sim["stockout_probability"])),
        ("Gross Profit", format_currency(sim["gross_profit"])),
        ("Competitor Index", f"{base_row.get('competitor_price_index', 1.0):.2f}"),
        ("Elasticity", f"{el_info.get('estimated_elasticity', rec['elasticity']):.2f}"),
        ("Confidence", str(rec.get("joint_confidence", ""))),
    ])

    candidates = opt.generate_candidate_prices(cur)
    sims = pd.DataFrame([opt.simulate_candidate(row, cp, el_info) for cp in candidates])
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(sims, x="candidate_price", y="predicted_units", markers=True,
                      title=t("Price vs Predicted Units"),
                      color_discrete_sequence=[UI_COLORS["electric_blue"]])
        fig.add_vline(x=cand, line_dash="dash", line_color="gray")
        _plot(fig)
    with c2:
        fig = px.line(sims, x="candidate_price", y="gross_profit", markers=True,
                      title=t("Price vs Gross Profit"),
                      color_discrete_sequence=[UI_COLORS["emerald"]])
        fig.add_vline(x=cand, line_dash="dash", line_color="gray")
        _plot(fig)


# ════════════════════════ PAGE 5 — RECOMMENDATION QUEUE ══════════════════════
def page_recommendation_queue(recs, state, filters):
    section_title("Recommendation Queue")
    st.caption(t("Each recommendation is a reviewable item: approve, override the price, request "
                 "replenish / transfer, pause, or add a note. Nothing is auto-applied."))
    if recs.empty:
        st.warning(t("No recommendations for current filters."))
        return

    decisions = st.session_state.setdefault("queue_decisions", {})

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        only_action = st.checkbox(t("Only items needing action"), value=True)
    with fc2:
        conf_filter = st.selectbox(t("Confidence"), ["All", "HIGH", "MEDIUM"], index=0)
    with fc3:
        review_only = st.checkbox(t("Only manual-review items"), value=False)

    q = recs.copy()
    if only_action:
        act = q["recommendation_action"].isin(["Increase", "Decrease", "Test"])
        if "inventory_action" in q.columns:
            act = act | q["inventory_action"].isin(
                ["EXPEDITE_ORDER", "REPLENISH", "INTER_REGION_TRANSFER", "STOP_OR_DELAY_ORDER"])
        q = q[act]
    if conf_filter != "All":
        q = q[q["joint_confidence"] == conf_filter]
    if review_only and "manual_review_required" in q.columns:
        q = q[q["manual_review_required"] == True]  # noqa: E712
    q = q.sort_values("gross_profit_lift", ascending=False).head(200)

    _kpi_row([
        ("Items in Queue", f"{len(q):,}"),
        ("Modeled GP Lift", format_currency(q["gross_profit_lift"].sum())),
        ("Decisions Logged", f"{len(decisions):,}"),
        ("Awaiting Review", f"{int(recs.get('manual_review_required', pd.Series([], dtype=bool)).sum()):,}"),
    ])

    show = [c for c in ["sku_id", "category", "region", "customer_tier", "current_price",
            "recommended_price", "price_change_pct", "recommendation_action", "inventory_action",
            "joint_confidence", "reason_code"] if c in q.columns]
    disp = q[show].copy()
    disp["decision"] = disp["sku_id"].map(lambda s: decisions.get(s, {}).get("decision", "—"))
    st.dataframe(disp, width="stretch", hide_index=True, height=360)

    st.divider()
    st.markdown(f"#### {t('Review an item')}")
    if len(q):
        sku = st.selectbox(t("SKU to review"), q["sku_id"].tolist())
        rec = q[q["sku_id"] == sku].iloc[0]
        a, b, c, d = st.columns(4)
        with a:
            metric_card("Current Price", format_currency(rec["current_price"]))
        with b:
            metric_card("Recommended", format_currency(rec["recommended_price"]), rec["recommendation_action"])
        with c:
            metric_card("Inventory Action", rec.get("inventory_action", "N/A"))
        with d:
            metric_card("Confidence", rec.get("joint_confidence", "N/A"))
        st.markdown('<div class="callout"><b>' + t("Reason") + ':</b> '
                    + f'{rec.get("reason_code", "")} · {rec.get("inventory_reason_code", "")}</div>',
                    unsafe_allow_html=True)

        dc1, dc2 = st.columns([2, 1])
        with dc1:
            choice = st.radio(t("Decision"), [t("Approve"), t("Override price"),
                              t("Request replenish"), t("Request transfer"), t("Pause"),
                              t("Hold for data")])
            override_price = None
            if choice == t("Override price"):
                override_price = st.number_input(t("New price"), min_value=0.0,
                                                 value=float(rec["recommended_price"]))
            note = st.text_input(t("Business note (optional)"))
        with dc2:
            st.markdown("&nbsp;")
            if st.button(t("Log decision"), type="primary"):
                decisions[sku] = {"decision": choice, "override_price": override_price, "note": note}
                st.success(f"{t('Decision logged')}: {sku}")
                st.rerun()

    if decisions:
        st.markdown(f"#### {t('Decision log')}")
        log = pd.DataFrame([{"sku_id": k, **v} for k, v in decisions.items()])
        st.dataframe(log, width="stretch", hide_index=True)
        st.download_button(t("Download decision log (CSV)"),
                           log.to_csv(index=False).encode("utf-8"),
                           file_name="recommendation_decisions.csv", mime="text/csv")


# ════════════════════ PAGE 6 — MODEL & METHODOLOGY ═══════════════════════════
ARCH_DIAGRAM = """POS Sales / Product / Cost / Inventory / Promotion / Competitor
            -> Data Validation & Feature Pipeline
            -> SKU x Region x Tier x Week Analytical Dataset
            -> SKU Segmentation
            -> Demand Forecasting (global tree model)
            -> Price Sensitivity Estimation
            -> Candidate Price Simulation
            -> Inventory & Business Constraints
            -> Recommendation Engine
            -> Price / Replenish / Transfer / Promote / Hold
            -> Dashboard & Human Approval
            -> Pilot Measurement & Model Monitoring"""

ASSUMPTIONS = [
    "Observed sales during a stockout do not equal true demand; censored periods are flagged.",
    "Sparse long-tail SKUs use product-family priors and conservative, rule-based recommendations.",
    "Historical price-sales relationships are decision-support evidence, not strict causality.",
    "Competitor prices are used only when product-match confidence is sufficient.",
    "Recommendations are a price range plus an operational action, kept under human review.",
]


def page_methodology(state, recs):
    section_title("Model Performance")
    metrics = {}
    if state.demand_model:
        metrics = state.demand_model.metrics
    elif state.model_results:
        metrics = state.model_results.get("hgb", {}).get("metrics", {})
    test_m = metrics.get("test", {})
    if test_m:
        _kpi_row([
            ("Model", metrics.get("model_name", "HistGradientBoosting")),
            ("MAE", f"{test_m.get('MAE', 0):.2f}"),
            ("WAPE", f"{test_m.get('WAPE', 0):.3f}"),
            ("Forecast Bias", f"{test_m.get('Bias', 0):.2f}"),
            ("Baseline Improvement", f"{test_m.get('WAPE_improvement', 0)*100:.1f}%"),
        ])
        st.caption(t("WAPE is preferred over MAPE because MAPE is unstable when demand is near zero."))

    bt = state.backtest_results.get("strategy_comparison") if state.backtest_results else None
    if bt is not None and len(bt):
        st.markdown(f"#### {t('Strategy Backtest (modeled)')}")
        st.dataframe(bt, width="stretch")
        d = bt.reset_index().rename(columns={"index": "strategy"})
        if "strategy" in d.columns and "total_gross_profit" in d.columns:
            fig = px.bar(d, x="strategy", y="total_gross_profit",
                         title=t("Gross Profit by Strategy"),
                         color_discrete_sequence=[UI_COLORS["emerald"]])
            fig.update_layout(xaxis_tickangle=-20)
            _plot(fig)

    section_title("Data & Methodology")
    st.markdown(t("**Data grain:** SKU x Region x Customer Tier x Week. Weekly aggregation reduces "
                  "long-tail daily noise and aligns with pricing and replenishment review cycles."))
    st.code(ARCH_DIAGRAM, language="text")

    section_title("Business Assumptions")
    for a in ASSUMPTIONS:
        st.markdown(f"- {t(a)}")

    section_title("Positioning")
    st.markdown('<div class="callout">' + t(
        "Positioned as an explainable pricing and inventory decision-support prototype — not a fully "
        "autonomous engine that changes prices without review. Dynamic pricing is about selecting the "
        "best business action under demand, inventory, margin, supply, and service-level constraints.")
        + '</div>', unsafe_allow_html=True)

    st.markdown('<div class="disclosure">' + t(
        "This demo uses synthetic sample data to demonstrate the analytical workflow, business logic, "
        "and user experience. It contains no confidential employer, supplier, customer, or transaction "
        "data. Model outputs are illustrative and should be validated through controlled business "
        "pilots before production use.") + '</div>', unsafe_allow_html=True)


def main():
    # 语言开关（最先读取，保证整页文案一致）
    i18n.set_language(st.session_state.get("lang", "en"))
    st.sidebar.selectbox(
        t("Language"), list(i18n.LANGUAGES.keys()),
        format_func=lambda k: i18n.LANGUAGES[k], key="lang",
    )
    i18n.set_language(st.session_state.get("lang", "en"))
    st.sidebar.markdown("---")

    st.title(t(HERO_HEADLINE))
    synthetic_disclosure()

    try:
        with st.spinner(t("Loading data...")):
            state = load_app_state()
    except Exception as exc:
        st.error(t("App failed to start. Ensure the repo includes the outputs/ and data/ deploy files."))
        st.exception(exc)
        st.stop()

    # 顶部 Tab 导航：完整故事链，一键切换、滚动流畅。
    recs, sales, filters = apply_sidebar_filters(
        state.recommendations, state.sales, "Executive Command Center"
    )

    tabs = st.tabs([
        t("Overview"),
        t("Pricing Opportunities"),
        t("Inventory Risk"),
        t("SKU Explorer"),
        t("Recommendation Queue"),
        t("Model & Methodology"),
    ])
    with tabs[0]:
        page_overview(recs, sales, state, filters)
    with tabs[1]:
        page_pricing_opportunities(recs, sales, state, filters)
    with tabs[2]:
        page_inventory_risk(recs, state, filters)
    with tabs[3]:
        page_sku_explorer(recs, sales, state, filters)
    with tabs[4]:
        page_recommendation_queue(recs, state, filters)
    with tabs[5]:
        page_methodology(state, recs)


# Streamlit 每次运行都会执行整个脚本，必须直接调用 main()
main()
