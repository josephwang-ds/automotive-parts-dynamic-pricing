"""汽车零配件动态定价与库存优化 — Streamlit 应用。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import i18n
from src.ai_analyst import SUGGESTED_QUESTIONS, generate_answer
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


def apply_sidebar_filters(recs, sales):
    """应用侧边栏筛选。"""
    st.sidebar.markdown(f"### {t('Filters')}")
    region = st.sidebar.selectbox(t("Region"), ["All"] + REGIONS, format_func=t)
    category = st.sidebar.selectbox(t("Category"), ["All"] + CATEGORIES, format_func=t)
    tier = st.sidebar.selectbox(t("Customer Tier"), ["All"] + CUSTOMER_TIERS, format_func=t)
    objective = st.sidebar.selectbox(t("Pricing Objective"), list(OBJECTIVES.keys()),
                                     format_func=lambda x: OBJECTIVES[x])
    scenario = st.sidebar.selectbox(t("Scenario"), list(SCENARIOS.keys()))
    margin_floor = st.sidebar.slider(t("Margin Floor"), 0.05, 0.35, 0.15, 0.01)
    max_move = st.sidebar.slider(t("Max Price Move %"), 1, 15, 10)
    confidence_threshold = st.sidebar.slider(t("Confidence Threshold"), 0.0, 1.0, 0.4, 0.05)

    filters = {"region": region, "category": category, "customer_tier": tier}
    mask = pd.Series(True, index=recs.index)
    sales_mask = pd.Series(True, index=sales.index) if len(sales) else pd.Series(dtype=bool)
    for k, v in filters.items():
        if v != "All":
            if k in recs.columns:
                mask &= recs[k] == v
            if k in sales.columns and len(sales):
                sales_mask &= sales[k] == v

    filtered_recs = recs[mask]
    filtered_sales = sales[sales_mask] if len(sales) else sales

    return filtered_recs, filtered_sales, {
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


def page_demand_model(state, filters):
    """Demand Model 页面。"""
    section_title("Demand Forecasting Model")

    metrics = {}
    if state.demand_model:
        metrics = state.demand_model.metrics
    elif state.model_results:
        hgb = state.model_results.get("hgb", {})
        metrics = hgb.get("metrics", {})

    test_m = metrics.get("test", {})
    st.markdown(
        '<div class="callout">'
        + tf("model_info",
             name=metrics.get("model_name", "HistGradientBoosting"),
             train=metrics.get("train_weeks", 78),
             val=metrics.get("val_weeks", 13))
        + '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        metric_card("MAE", f"{test_m.get('MAE', 0):.2f}")
    with c2:
        metric_card("RMSE", f"{test_m.get('RMSE', 0):.2f}")
    with c3:
        metric_card("WAPE", f"{test_m.get('WAPE', 0):.3f}")
    with c4:
        metric_card("RMSLE", f"{test_m.get('RMSLE', 0):.4f}")
    with c5:
        metric_card("Bias", f"{test_m.get('Bias', 0):.2f}")
    with c6:
        imp = test_m.get("WAPE_improvement", 0)
        metric_card("Baseline Improvement", f"{imp*100:.1f}%")

    # 模型比较
    if state.model_results:
        comparison = []
        for mt, result in state.model_results.items():
            m = result.get("metrics", {}).get("test", {})
            comparison.append({
                "Model": result.get("metrics", {}).get("model_name", mt),
                "MAE": m.get("MAE", 0),
                "RMSE": m.get("RMSE", 0),
                "WAPE": m.get("WAPE", 0),
                "WAPE Improvement": m.get("WAPE_improvement", 0),
            })
        if comparison:
            st.markdown(f"**{t('Model Comparison')}**")
            st.dataframe(pd.DataFrame(comparison), width='stretch', hide_index=True)

    # 测试集预测图
    if state.model_results and "hgb" in state.model_results:
        preds = state.model_results["hgb"].get("test_predictions")
        if preds is not None and len(preds) > 0:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(
                    preds, x="actual", y="predicted",
                    opacity=0.3, title=t("Actual vs Predicted"),
                    color_discrete_sequence=[UI_COLORS["electric_blue"]],
                )
                fig.add_trace(go.Scatter(
                    x=[0, preds["actual"].max()], y=[0, preds["actual"].max()],
                    mode="lines", line=dict(dash="dash", color="gray"),
                ))
                fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig, width='stretch')

            with col2:
                preds["residual"] = preds["actual"] - preds["predicted"]
                fig = px.histogram(
                    preds, x="residual", nbins=50,
                    title=t("Residual Distribution"),
                    color_discrete_sequence=[UI_COLORS["electric_blue"]],
                )
                fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig, width='stretch')

            # 按品类误差
            err_by_cat = preds.groupby("category").apply(
                lambda g: np.mean(np.abs(g["actual"] - g["predicted"]))
            ).reset_index(name="MAE")
            fig = px.bar(err_by_cat, x="category", y="MAE",
                         title=t("Error by Category"),
                         color_discrete_sequence=[UI_COLORS["warning_amber"]])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, width='stretch')

    st.markdown(
        f'<div class="callout">{tf("limitations")}</div>',
        unsafe_allow_html=True,
    )


def page_elasticity(state, filters):
    """Elasticity Explorer。"""
    section_title("Price Elasticity Explorer")

    el_df = state.elasticity_df
    if el_df.empty:
        st.warning(t("Elasticity estimates not available."))
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sel_cat = st.selectbox(t("Category"), ["All"] + CATEGORIES, key="el_cat", format_func=t)
    with col2:
        sel_region = st.selectbox(t("Region"), ["All"] + REGIONS, key="el_region", format_func=t)
    with col3:
        sel_tier = st.selectbox(t("Customer Tier"), ["All"] + CUSTOMER_TIERS, key="el_tier", format_func=t)
    with col4:
        sku_list = state.products["sku_id"].tolist()[:100]
        sel_sku = st.selectbox("SKU", ["All"] + sku_list, key="el_sku", format_func=t)

    filtered = el_df.copy()
    if sel_cat != "All":
        filtered = filtered[filtered["category"] == sel_cat]
    if sel_tier != "All":
        filtered = filtered[filtered["customer_tier"] == sel_tier]

    # 热力图
    heatmap_data = el_df[el_df["estimation_level"] == "category_tier"].pivot_table(
        index="category", columns="customer_tier",
        values="estimated_elasticity", aggfunc="first",
    )
    if not heatmap_data.empty:
        fig = px.imshow(
            heatmap_data, title=t("Elasticity Heatmap: Category × Tier"),
            color_continuous_scale="RdYlGn_r", aspect="auto",
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, width='stretch')

    if len(filtered) > 0:
        row = filtered.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            metric_card("Elasticity", f"{row['estimated_elasticity']:.3f}")
        with c2:
            metric_card("Confidence", f"{row.get('confidence_score', 0):.2f}")
        with c3:
            metric_card("Sample Size", f"{row.get('sample_size', 0):,}")
        with c4:
            metric_card("Price Variation", f"{row.get('price_variation', 0):.4f}")
        with c5:
            metric_card("Segment", row.get("elasticity_class", "N/A"))

    st.markdown(
        f'<div class="callout">{tf("elasticity_distinction")}</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        filtered[["elasticity_segment", "estimated_elasticity", "confidence_score",
                   "sample_size", "elasticity_class", "estimation_level"]].head(20),
        width='stretch', hide_index=True,
    )


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


def page_inventory(inv, recs, filters):
    """向后兼容别名。"""
    pass


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


def page_ai_analyst(state, recs, sales, filters):
    """AI Analyst。"""
    section_title("AI Analyst (Local Deterministic)")

    st.markdown(f"**{t('Suggested Questions:')}**")
    cols = st.columns(3)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 3]:
            if st.button(q, key=f"sq_{i}"):
                st.session_state["ai_question"] = q

    question = st.text_input(
        t("Ask a question"),
        value=st.session_state.get("ai_question", ""),
        placeholder=t("e.g., Where is the largest modeled margin opportunity?"),
    )

    if st.button(t("Analyze")) or st.session_state.get("ai_question"):
        if question:
            model_metrics = {}
            if state.demand_model:
                model_metrics = state.demand_model.metrics
            elif state.model_results:
                model_metrics = state.model_results.get("hgb", {}).get("metrics", {})

            result = generate_answer(
                question=question,
                filtered_data={
                    "recommendations": recs,
                    "sales": sales,
                    "elasticity": state.elasticity_df,
                    "products": state.products,
                    "transfers": getattr(state, "transfers", pd.DataFrame()),
                    "inventory_metrics": getattr(state, "inventory_metrics", pd.DataFrame()),
                },
                model_metrics=model_metrics,
                active_filters=filters,
                provider="local",
            )

            st.markdown(f'<div class="callout"><strong>{t("Answer:")}</strong> {result["answer"]}</div>',
                         unsafe_allow_html=True)
            st.markdown(f"**{t('Intent:')}** {result['intent']} | **{t('Provider:')}** {result['provider']}")
            st.markdown(f"**{t('Active Filters:')}** {result['active_filters']}")

            if result.get("evidence"):
                st.markdown(f"**{t('Evidence Used:')}**")
                for e in result["evidence"]:
                    st.markdown(f"- {e}")

            st.markdown(f"**{t('Caveat:')}** {result['caveat']}")

            if result.get("metric_definitions"):
                st.markdown(f"**{t('Metric Definitions:')}**")
                for k, v in result["metric_definitions"].items():
                    st.markdown(f"- **{k}:** {v}")


def page_governance():
    """Data & Governance。"""
    section_title("Data & Governance")

    st.markdown(f"### {t('Star Schema')}")
    schema = {
        "fact_sales_weekly": "Weekly sales transactions",
        "fact_inventory_weekly": "Weekly inventory positions and weeks of cover",
        "fact_purchase_orders": "Open and planned purchase orders",
        "fact_inventory_transfers": "Inter-region transfer recommendations",
        "fact_inventory_actions": "Replenishment, stop-order, markdown actions",
        "fact_price_history": "Price changes with reason codes",
        "fact_recommendations": "Joint pricing + inventory recommendations",
        "dim_product": "SKU master with cost, margin, lead time",
        "dim_supplier": "Supplier lead time and MOQ",
        "dim_region": "4 BC regions",
        "dim_customer_tier": "Retail, Trade, Fleet",
        "dim_calendar": "104-week calendar",
    }
    for table, desc in schema.items():
        st.markdown(f"- **{table}:** {t(desc)}")

    st.markdown(f"### {t('Metric Definitions')}")
    metrics_def = {
        "Revenue": "realized_price × units_sold",
        "Gross Profit": "revenue − COGS (unit_cost × units_sold)",
        "Gross Margin": "gross_profit / revenue",
        "Inventory Turns": "annual_units_sold / average_inventory",
        "Weeks of Cover": "ending_inventory / weekly_demand",
        "Stockout Rate": "% of SKU-weeks with stockout_flag = true",
        "Excess Inventory Value": "sum(inventory × unit_cost) where weeks_of_cover > 16",
        "Modeled Lift": "simulated GP difference between dynamic and current pricing",
    }
    for m, d in metrics_def.items():
        st.markdown(f"- **{t(m)}:** {d}")

    st.markdown(f"### {t('Approval Workflow')}")
    st.markdown(tf("approval_workflow"))

    st.markdown(f"### {t('Production Monitoring Checklist')}")
    monitors = [
        "Demand forecast error", "Stockout-rate change", "Excess-inventory change",
        "Transfer success rate", "Replenishment service level", "Inventory-turn change",
        "Holding-cost change", "Realized vs expected sell-through",
        "Analyst override rate", "Pricing rollback rate", "Inventory-action rollback rate",
    ]
    for m in monitors:
        st.markdown(f"- {t(m)}")


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

    st.sidebar.markdown(f"### {t('Navigation')}")
    pages = [
        "Executive Command Center",
        "Demand Model",
        "Elasticity Explorer",
        "SKU Decision Workbench",
        "Inventory Control Tower",
        "Backtest & Rollback",
        "AI Analyst",
        "Data & Governance",
    ]
    page = st.sidebar.radio(
        "Section", pages, format_func=t, label_visibility="collapsed",
    )
    st.sidebar.markdown("---")

    recs, sales, filters = apply_sidebar_filters(
        state.recommendations, state.sales
    )

    if page == "Executive Command Center":
        page_executive(recs, sales, state.inventory_analysis,
                       state.backtest_results, filters)
    elif page == "Demand Model":
        page_demand_model(state, filters)
    elif page == "Elasticity Explorer":
        page_elasticity(state, filters)
    elif page == "SKU Decision Workbench":
        page_pricing_studio(recs, state, filters)
    elif page == "Inventory Control Tower":
        page_inventory_control_tower(state, state.inventory_analysis, recs, filters)
    elif page == "Backtest & Rollback":
        page_backtest(state, recs, filters)
    elif page == "AI Analyst":
        page_ai_analyst(state, recs, sales, filters)
    elif page == "Data & Governance":
        page_governance()


# Streamlit 每次运行都会执行整个脚本，必须直接调用 main()
main()
