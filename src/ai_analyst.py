"""本地确定性 AI 分析师。"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.utils import format_currency, format_pct, safe_divide


INTENT_KEYWORDS = {
    "pricing_opportunity": ["opportunity", "margin opportunity", "largest", "pricing"],
    "margin": ["margin", "profit", "gross profit", "gp"],
    "inventory": ["inventory", "stock", "cover", "turns"],
    "inventory_health": ["inventory health", "health distribution", "status"],
    "weeks_of_cover": ["weeks of cover", "weeks-of-cover", "woc"],
    "inventory_turns": ["inventory turns", "turnover"],
    "stockout": ["stockout", "out of stock", "supply risk"],
    "lost_sales": ["lost sales", "lost-sales"],
    "excess_stock": ["excess", "slow-moving", "slow moving", "overstock"],
    "transfer_opportunity": ["transfer", "between regions", "inter-region"],
    "replenishment": ["replenishment", "replenish", "reorder"],
    "stop_order": ["stop order", "delay order", "stop or delay"],
    "markdown_economics": ["markdown economics", "net markdown", "markdown value"],
    "obsolescence": ["obsolete", "obsolescence", "end-of-life"],
    "pricing_vs_inventory": ["wrong problem", "pricing vs inventory", "pricing being used"],
    "category": ["category", "categories", "brakes", "filters"],
    "region": ["region", "lower mainland", "vancouver", "interior", "northern"],
    "customer_tier": ["tier", "retail", "trade", "fleet", "customer"],
    "elasticity": ["elasticity", "elastic", "inelastic", "price sensitivity"],
    "model_performance": ["model", "accuracy", "wape", "mae", "forecast"],
    "guardrail": ["guardrail", "constraint", "manual review"],
    "rollback": ["rollback", "roll back", "revert", "cancel"],
}


SUGGESTED_QUESTIONS = [
    "Which categories carry the most excess inventory value?",
    "Which SKUs face the highest stockout risk?",
    "Where can inventory be transferred between regions?",
    "Which markdowns have positive net economic value?",
    "Which overstocked items should not be discounted?",
    "Which SKUs need replenishment?",
    "Which open orders should be stopped or delayed?",
    "Where is pricing being used to solve the wrong problem?",
    "What happens if we cancel 50% of transfer recommendations?",
    "Which recommendations require manual review?",
]


def _detect_intent(question: str) -> str:
    """检测问题意图。"""
    q_lower = question.lower()
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in q_lower)
        if score > 0:
            scores[intent] = score
    if not scores:
        return "general"
    return max(scores, key=scores.get)


def _apply_filters(
    df: pd.DataFrame,
    active_filters: dict,
) -> pd.DataFrame:
    """应用筛选条件。"""
    filtered = df.copy()
    for key, value in active_filters.items():
        if value and value != "All" and key in filtered.columns:
            if isinstance(value, list):
                filtered = filtered[filtered[key].isin(value)]
            else:
                filtered = filtered[filtered[key] == value]
    return filtered


def generate_answer(
    question: str,
    filtered_data: dict[str, pd.DataFrame],
    model_metrics: dict | None = None,
    active_filters: dict | None = None,
    provider: str = "local",
) -> dict:
    """
    生成本地确定性分析回答。

    Parameters
    ----------
    question : str
        用户问题
    filtered_data : dict
        包含 sales, products, recommendations, elasticity 等 DataFrame
    model_metrics : dict
        模型评估指标
    active_filters : dict
        当前活跃筛选条件
    provider : str
        "local" 或 "future_llm"（占位）
    """
    if provider == "future_llm":
        return {
            "answer": "Future LLM provider is not implemented. Using local deterministic analyst.",
            "intent": "general",
            "provider": provider,
        }

    active_filters = active_filters or {}
    intent = _detect_intent(question)
    recs = filtered_data.get("recommendations", pd.DataFrame())
    sales = filtered_data.get("sales", pd.DataFrame())
    elasticity = filtered_data.get("elasticity", pd.DataFrame())
    products = filtered_data.get("products", pd.DataFrame())
    transfers = filtered_data.get("transfers", pd.DataFrame())
    inv_metrics = filtered_data.get("inventory_metrics", pd.DataFrame())

    if not recs.empty:
        recs = _apply_filters(recs, active_filters)

    evidence = []
    caveat = (
        "This analysis uses synthetic demo data. Results are modeled estimates, "
        "not proven business impact. All recommendations require human approval."
    )

    answer_parts = []
    metric_definitions = {}

    if intent == "pricing_opportunity":
        if not recs.empty:
            top = recs.nlargest(5, "gross_profit_lift")
            total_lift = recs["gross_profit_lift"].sum()
            answer_parts.append(
                f"The largest modeled margin opportunities total {format_currency(total_lift)} "
                f"in projected gross profit lift across {len(recs)} SKUs."
            )
            for _, row in top.iterrows():
                evidence.append(
                    f"{row['sku_id']} ({row['category']}, {row['region']}, {row['customer_tier']}): "
                    f"GP lift {format_currency(row['gross_profit_lift'])}, "
                    f"action: {row['recommendation_action']}"
                )
            metric_definitions["gross_profit_lift"] = (
                "Projected gross profit at recommended price minus current gross profit"
            )

    elif intent == "margin":
        if not recs.empty:
            avg_margin = recs["gross_margin_pct"].mean()
            total_gp = recs["projected_gross_profit"].sum()
            answer_parts.append(
                f"Average projected gross margin: {format_pct(avg_margin)}. "
                f"Total projected gross profit: {format_currency(total_gp)}."
            )
            evidence.append(f"Based on {len(recs)} recommendations in scope")

    elif intent == "transfer_opportunity":
        if not transfers.empty:
            answer_parts.append(f"{len(transfers)} inter-region transfer opportunities identified.")
            for _, row in transfers.nlargest(5, "net_transfer_value").iterrows():
                evidence.append(
                    f"{row['sku_id']}: {row['source_region']} → {row['destination_region']}, "
                    f"qty={row['transfer_quantity']:.0f}, net value={format_currency(row['net_transfer_value'])}"
                )
        elif "transfer_candidate_flag" in recs.columns:
            tc = recs[recs["transfer_candidate_flag"] == True]  # noqa: E712
            answer_parts.append(f"{len(tc)} SKUs flagged as transfer candidates.")
        metric_definitions["net_transfer_value"] = "Expected GP recovered minus transfer cost"

    elif intent == "replenishment":
        if "inventory_action" in recs.columns:
            rep = recs[recs["inventory_action"].isin(["REPLENISH", "EXPEDITE_ORDER"])]
            answer_parts.append(f"{len(rep)} SKUs need replenishment or expedited orders.")
            if not rep.empty and "recommended_order_quantity" in rep.columns:
                total_qty = rep["recommended_order_quantity"].sum()
                evidence.append(f"Total recommended order quantity: {total_qty:,.0f} units")

    elif intent == "stop_order":
        if "inventory_action" in recs.columns:
            stop = recs[recs["inventory_action"] == "STOP_OR_DELAY_ORDER"]
            answer_parts.append(f"{len(stop)} SKUs recommended for stop/delay order.")

    elif intent == "markdown_economics":
        if "inventory_action" in recs.columns:
            md = recs[recs["inventory_action"] == "PRICE_MARKDOWN"]
            answer_parts.append(
                f"{len(md)} markdown recommendations with positive net economic value in scope."
            )

    elif intent == "pricing_vs_inventory":
        if "inventory_action" in recs.columns and "pricing_action" in recs.columns:
            wrong = recs[
                (recs["pricing_action"] == "Decrease")
                & (recs["inventory_action"].isin(["REPLENISH", "EXPEDITE_ORDER", "INTER_REGION_TRANSFER"]))
            ]
            answer_parts.append(
                f"{len(wrong)} cases where pricing decrease conflicts with inventory needs (resolved by joint optimizer)."
            )

    elif intent == "inventory_health" or intent == "weeks_of_cover":
        if not inv_metrics.empty:
            avg_woc = inv_metrics["available_weeks_of_cover"].mean()
            answer_parts.append(f"Average available weeks of cover: {avg_woc:.1f}.")
            if "inventory_status" in inv_metrics.columns:
                dist = inv_metrics["inventory_status"].value_counts()
                for status, cnt in dist.head(5).items():
                    evidence.append(f"{status}: {cnt} SKU-regions")

    elif intent == "lost_sales":
        if "lost_sales_estimate" in recs.columns:
            total = recs["lost_sales_estimate"].sum()
            answer_parts.append(f"Estimated lost sales in scope: {total:,.0f} units.")
        elif not inv_metrics.empty and "lost_sales_estimate" in inv_metrics.columns:
            total = inv_metrics["lost_sales_estimate"].sum()
            answer_parts.append(f"Estimated lost sales: {total:,.0f} units.")

    elif intent == "obsolescence":
        if "obsolescence_risk_score" in recs.columns:
            high = recs[recs["obsolescence_risk_score"] > 0.65]
            answer_parts.append(f"{len(high)} SKUs with high obsolescence risk (>0.65).")

    elif intent in ("inventory", "excess_stock"):
        if not sales.empty:
            latest = sales.sort_values("week_num").groupby("sku_id").last()
            excess = latest[latest.get("excess_inventory_flag", False) == True]  # noqa: E712
            if len(excess) == 0 and "weeks_of_cover" in latest.columns:
                excess = latest[latest["weeks_of_cover"] > 16]
            answer_parts.append(
                f"{len(excess)} SKUs have excess inventory (weeks of cover > 16)."
            )
            if not excess.empty:
                by_cat = excess.groupby("category").size()
                for cat, count in by_cat.items():
                    evidence.append(f"{cat}: {count} excess SKUs")

    elif intent == "stockout":
        if not sales.empty:
            latest = sales.sort_values("week_num").groupby("sku_id").last()
            risk = latest[latest.get("stockout_flag", False) == True]  # noqa: E712
            if len(risk) == 0 and "weeks_of_cover" in latest.columns:
                risk = latest[latest["weeks_of_cover"] < 2]
            answer_parts.append(f"{len(risk)} SKUs are at stockout risk.")

    elif intent == "category":
        if not recs.empty:
            by_cat = recs.groupby("category")["gross_profit_lift"].sum().sort_values(ascending=False)
            answer_parts.append("Gross profit lift by category:")
            for cat, lift in by_cat.head(5).items():
                evidence.append(f"{cat}: {format_currency(lift)}")

    elif intent == "elasticity":
        if not elasticity.empty:
            low_conf = elasticity[elasticity.get("elasticity_class", "") == "Low confidence"]
            answer_parts.append(
                f"{len(low_conf)} elasticity estimates have low confidence out of {len(elasticity)} total."
            )
            if not low_conf.empty:
                for _, row in low_conf.head(5).iterrows():
                    evidence.append(
                        f"{row.get('elasticity_segment', 'N/A')}: "
                        f"elasticity={row.get('estimated_elasticity', 'N/A'):.2f}, "
                        f"confidence={row.get('confidence_score', 0):.2f}"
                    )

    elif intent == "guardrail":
        if not recs.empty:
            manual = recs[recs["recommendation_action"] == "Manual Review"]
            guarded = recs[recs["guardrail_triggered"].str.len() > 0]
            answer_parts.append(
                f"{len(manual)} recommendations require manual review. "
                f"{len(guarded)} recommendations triggered guardrails."
            )

    elif intent == "rollback":
        match = re.search(r"(\d+)%", question)
        pct = int(match.group(1)) / 100 if match else 0.5
        if "transfer" in question.lower() or "cancel" in question.lower():
            if not transfers.empty:
                retained = len(transfers) * (1 - pct)
                answer_parts.append(
                    f"A {pct*100:.0f}% transfer cancellation would retain {retained:.0f} of "
                    f"{len(transfers)} transfer recommendations."
                )
        else:
            if not recs.empty:
                col = "modeled_gp_lift" if "modeled_gp_lift" in recs.columns else "gross_profit_lift"
                retained_gp = recs[col].sum() * (1 - pct)
                answer_parts.append(
                    f"A {pct*100:.0f}% pricing rollback would retain {format_currency(retained_gp)} "
                    f"of the modeled gross profit lift."
                )
            evidence.append(f"Total SKUs affected: {len(recs)}")

    elif intent == "model_performance":
        if model_metrics:
            test_m = model_metrics.get("test", model_metrics)
            answer_parts.append(
                f"Demand model ({model_metrics.get('model_name', 'N/A')}): "
                f"Test WAPE={test_m.get('WAPE', 'N/A'):.3f}, "
                f"MAE={test_m.get('MAE', 'N/A'):.2f}, "
                f"RMSE={test_m.get('RMSE', 'N/A'):.2f}."
            )
            if "WAPE_improvement" in test_m:
                answer_parts.append(
                    f"WAPE improvement over baseline: {test_m['WAPE_improvement']*100:.1f}%."
                )

    else:
        if not recs.empty:
            answer_parts.append(
                f"Analysis scope: {len(recs)} SKU recommendations, "
                f"total modeled GP lift: {format_currency(recs['gross_profit_lift'].sum())}."
            )
        else:
            answer_parts.append("No data available for the current filter scope.")

    # 空数据处理
    if not answer_parts:
        answer_parts.append(
            "No matching data found for the current filters. "
            "Try broadening your region, category, or tier selection."
        )

    filter_str = ", ".join(f"{k}={v}" for k, v in active_filters.items() if v and v != "All")

    return {
        "answer": " ".join(answer_parts),
        "intent": intent,
        "active_filters": filter_str or "None",
        "metric_definitions": metric_definitions,
        "evidence": evidence[:10],
        "caveat": caveat,
        "provider": provider,
        "suggested_followups": SUGGESTED_QUESTIONS[:3],
    }
