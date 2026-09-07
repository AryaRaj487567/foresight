import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

# Set up project root in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

st.set_page_config(page_title="PROJECT FORESIGHT", page_icon="📊", layout="wide")

# Colors
COLOR_SCHEME = {
    "primary": "#1f77b4",
    "danger": "#d62728",
    "warning": "#ff7f0e",
    "success": "#2ca02c",
    "info": "#17becf"
}

RISK_COLORS = {
    "CRITICAL": "#d62728",
    "HIGH": "#ff7f0e",
    "MEDIUM": "#ffca28",
    "LOW": "#2ca02c",
    "NORMAL": "#2ca02c",
    "URGENT": "#d62728"
}

@st.cache_data
def load_dataframe(rel_path):
    path = project_root / rel_path
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"Error reading {rel_path}: {e}")
            return None
    return None

@st.cache_data
def load_json(rel_path):
    path = project_root / rel_path
    if path.exists():
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Error reading {rel_path}: {e}")
            return None
    return None

def format_currency(val):
    if pd.isna(val): return "₹0.00"
    return f"₹{val:,.2f}"

def format_pct(val):
    if pd.isna(val): return "0.0%"
    return f"{val:.1f}%" if val > 1 else f"{val*100:.1f}%"

# Load data
df_merged = load_dataframe("data/processed/merged_analysis_ready.csv")
df_sku = load_dataframe("data/processed/sku_master_cleaned.csv")
df_forecast = load_dataframe("outputs/forecasts/forecast_8week.csv")
df_metrics = load_dataframe("outputs/forecasts/evaluation_metrics.csv")
df_backtest = load_dataframe("outputs/forecasts/backtest_results.csv")
df_weekly = load_dataframe("outputs/forecasts/weekly_sales.csv")
df_stockout = load_dataframe("outputs/risk/stockout_risk.csv")
df_overstock = load_dataframe("outputs/risk/overstock_analysis.csv")
df_reorder = load_dataframe("outputs/risk/reorder_recommendations.csv")
df_markdown = load_dataframe("outputs/risk/markdown_candidates.csv")
df_actions = load_dataframe("outputs/risk/inventory_action_recommendations.csv")
risk_summary = load_json("outputs/risk/risk_summary.json")


# Sidebar
st.sidebar.title("🛋️ NorthBay Living")
st.sidebar.markdown("**PROJECT FORESIGHT**")

pages = ["Executive Overview", "Demand Forecasts", "Inventory Risk", "Reorder & Markdown", "Financial Impact"]
selection = st.sidebar.radio("Navigation", pages)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

# Extract categories for filter
categories = []
if df_sku is not None and "category" in df_sku.columns:
    categories = sorted(df_sku["category"].dropna().unique().tolist())
elif df_merged is not None and "category" in df_merged.columns:
    categories = sorted(df_merged["category"].dropna().unique().tolist())

selected_categories = st.sidebar.multiselect("Select Categories", options=categories, default=categories)

# Filter function
def filter_by_cat(df):
    if df is None: return None
    if not selected_categories: return df
    
    if "category" in df.columns:
        return df[df["category"].isin(selected_categories)]
    elif "sku_id" in df.columns and df_sku is not None and "category" in df_sku.columns:
        cat_map = df_sku.set_index("sku_id")["category"].to_dict()
        df_temp = df.copy()
        df_temp["category"] = df_temp["sku_id"].map(cat_map)
        return df_temp[df_temp["category"].isin(selected_categories)].drop(columns=["category"], errors='ignore')
    return df

# Main app pages
if selection == "Executive Overview":
    st.header("Executive Overview")
    
    # Calculate KPIs
    total_skus = 0
    total_revenue_30d = 0.0
    skus_at_risk = 0
    excess_cap = 0.0
    rev_at_risk = 0.0
    
    f_sku = filter_by_cat(df_sku)
    if f_sku is not None:
        total_skus = len(f_sku)
        
    f_merged = filter_by_cat(df_merged)
    if f_merged is not None and "date" in f_merged.columns and "revenue" in f_merged.columns:
        try:
            f_merged["date"] = pd.to_datetime(f_merged["date"])
            max_date = f_merged["date"].max()
            last_30d = f_merged[f_merged["date"] >= max_date - pd.Timedelta(days=30)]
            total_revenue_30d = last_30d["revenue"].sum()
        except:
            pass
            
    f_stockout = filter_by_cat(df_stockout)
    if f_stockout is not None and "risk_level" in f_stockout.columns:
        skus_at_risk = len(f_stockout[f_stockout["risk_level"].isin(["CRITICAL", "HIGH"])])
        
    f_overstock = filter_by_cat(df_overstock)
    if f_overstock is not None and "excess_value" in f_overstock.columns:
        excess_cap = f_overstock["excess_value"].sum()
        
    if risk_summary and "total_revenue_at_risk" in risk_summary:
        rev_at_risk = risk_summary["total_revenue_at_risk"]
        
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total SKUs", f"{total_skus:,}")
    c2.metric("Total Revenue (30d)", format_currency(total_revenue_30d))
    c3.metric("SKUs at Risk (CRIT/HIGH)", f"{skus_at_risk:,}")
    c4.metric("Excess Capital", format_currency(excess_cap))
    c5.metric("Revenue at Risk", format_currency(rev_at_risk))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Distribution")
        if f_stockout is not None and "risk_level" in f_stockout.columns:
            risk_counts = f_stockout["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            fig = px.pie(risk_counts, names="Risk Level", values="Count", 
                         color="Risk Level", color_discrete_map=RISK_COLORS, hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Stockout risk data not available.")
            
    with col2:
        st.subheader("Revenue by Category")
        if f_merged is not None and "category" in f_merged.columns and "revenue" in f_merged.columns:
            cat_rev = f_merged.groupby("category")["revenue"].sum().reset_index()
            fig = px.bar(cat_rev, x="category", y="revenue", color="category")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Revenue data not available.")
            
    st.subheader("Top 10 Revenue at Risk SKUs")
    if risk_summary and "top_10_skus" in risk_summary and "revenue_at_risk" in risk_summary["top_10_skus"]:
        top_skus = risk_summary["top_10_skus"]["revenue_at_risk"]
        if top_skus:
            df_top = pd.DataFrame(top_skus)
            if "value" in df_top.columns:
                df_top["value"] = df_top["value"].apply(format_currency)
            st.dataframe(df_top, use_container_width=True)
        else:
            st.info("No top SKUs available.")
    else:
        st.info("Risk summary data not available.")

elif selection == "Demand Forecasts":
    st.header("Demand Forecasts")
    
    f_weekly = filter_by_cat(df_weekly)
    f_forecast = filter_by_cat(df_forecast)
    f_metrics = filter_by_cat(df_metrics)
    
    if f_weekly is not None and f_forecast is not None and df_sku is not None:
        # Create SKU selector
        merged_sku = pd.merge(f_weekly[["sku_id"]].drop_duplicates(), df_sku[["sku_id", "product_name"]], on="sku_id", how="left")
        merged_sku["display"] = merged_sku["sku_id"].astype(str) + " - " + merged_sku["product_name"].fillna("Unknown")
        sku_list = merged_sku["display"].tolist()
        
        if sku_list:
            selected_sku_display = st.selectbox("Select SKU", sku_list)
            selected_sku = selected_sku_display.split(" - ")[0]
            
            # Forecast Chart
            sku_weekly = f_weekly[f_weekly["sku_id"].astype(str) == selected_sku].copy()
            sku_fcst = f_forecast[f_forecast["sku_id"].astype(str) == selected_sku].copy()
            
            if not sku_weekly.empty and not sku_fcst.empty:
                sku_weekly["week_start"] = pd.to_datetime(sku_weekly["week_start"])
                sku_fcst["week_start"] = pd.to_datetime(sku_fcst["week_start"])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sku_weekly["week_start"], y=sku_weekly["weekly_units"],
                                         mode='lines', name='Historical Sales', line=dict(color=COLOR_SCHEME["primary"])))
                fig.add_trace(go.Scatter(x=sku_fcst["week_start"], y=sku_fcst["forecast_units"],
                                         mode='lines', name='Forecast', line=dict(color=COLOR_SCHEME["warning"], dash='dash')))
                
                # Confidence interval
                fig.add_trace(go.Scatter(x=sku_fcst["week_start"].tolist() + sku_fcst["week_start"].tolist()[::-1],
                                         y=sku_fcst["upper_bound"].tolist() + sku_fcst["lower_bound"].tolist()[::-1],
                                         fill='toself', fillcolor='rgba(255, 127, 14, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                                         hoverinfo="skip", showlegend=True, name='Confidence Interval'))
                
                fig.update_layout(title=f"Demand Forecast for {selected_sku_display}", xaxis_title="Date", yaxis_title="Units")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data to plot chart for this SKU.")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Model Performance")
                if f_metrics is not None:
                    sku_metric = f_metrics[f_metrics["sku_id"].astype(str) == selected_sku]
                    if not sku_metric.empty:
                        m = sku_metric.iloc[0]
                        st.metric("MAE", f"{m.get('mae', 0):.2f}")
                        st.metric("RMSE", f"{m.get('rmse', 0):.2f}")
                        st.metric("MAPE", format_pct(m.get('mape', 0)))
                        st.metric("WAPE", format_pct(m.get('wape', 0)))
                    else:
                        st.info("Metrics not found.")
            with col2:
                st.subheader("8-Week Forecast Table")
                if not sku_fcst.empty:
                    display_fcst = sku_fcst[["week_start", "forecast_units", "lower_bound", "upper_bound"]].copy()
                    display_fcst["week_start"] = display_fcst["week_start"].dt.strftime("%Y-%m-%d")
                    display_fcst = display_fcst.round(2)
                    st.dataframe(display_fcst, use_container_width=True)
                
            st.markdown("---")
            if df_backtest is not None and not df_backtest.empty:
                st.markdown("---")
                st.subheader("🧪 Rolling-Origin Backtest Model Evaluation")
                st.markdown("Strict out-of-sample comparison across 3 historical origins (8-week test horizon per fold, zero future leakage):")
                summary_agg = df_backtest.groupby("model")[["wape", "mae", "rmse"]].mean().round(2).reset_index()
                col_b1, col_b2 = st.columns([1, 1])
                with col_b1:
                    st.dataframe(summary_agg, use_container_width=True)
                with col_b2:
                    fig_bt = px.bar(summary_agg, x="model", y="wape", color="model", text="wape", title="Average Backtest WAPE (%)")
                    st.plotly_chart(fig_bt, use_container_width=True)
            
        else:
            st.warning("No SKUs available for selected filters.")
    else:
        st.error("Required forecast data missing.")


elif selection == "Inventory Risk":
    st.header("Inventory Risk")
    
    f_stockout = filter_by_cat(df_stockout)
    if f_stockout is not None:
        st.subheader("Risk Heatmap")
        
        def color_risk(val):
            color = RISK_COLORS.get(val, "black")
            return f'color: {color}; font-weight: bold;'
        
        display_risk = f_stockout.copy()
        if "risk_level" in display_risk.columns:
            st.dataframe(display_risk.style.map(color_risk, subset=['risk_level']), use_container_width=True)
            
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Risk by Category")
            if "category" in display_risk.columns or df_sku is not None:
                if "category" not in display_risk.columns and df_sku is not None:
                    display_risk = pd.merge(display_risk, df_sku[["sku_id", "category"]], on="sku_id", how="left")
                
                if "category" in display_risk.columns and "risk_level" in display_risk.columns:
                    risk_cat = display_risk.groupby(["category", "risk_level"]).size().reset_index(name="count")
                    fig = px.bar(risk_cat, x="category", y="count", color="risk_level", barmode="stack", color_discrete_map=RISK_COLORS)
                    st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Days of Supply Distribution")
            if "days_of_supply" in display_risk.columns:
                fig2 = px.histogram(display_risk, x="days_of_supply", nbins=30, color_discrete_sequence=[COLOR_SCHEME["info"]])
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Stockout risk data not available.")

elif selection == "Reorder & Markdown":
    st.header("Reorder & Markdown Actions")
    
    f_reorder = filter_by_cat(df_reorder)
    f_markdown = filter_by_cat(df_markdown)
    
    st.subheader("Reorder Recommendations")
    if f_reorder is not None and not f_reorder.empty:
        total_cost = f_reorder.get("estimated_cost", pd.Series([0])).sum()
        st.metric("Total Estimated Reorder Cost", format_currency(total_cost))
        
        if "priority" in f_reorder.columns:
            priorities = ["ALL"] + f_reorder["priority"].dropna().unique().tolist()
            sel_pri = st.selectbox("Filter by Priority", priorities)
            if sel_pri != "ALL":
                display_reorder = f_reorder[f_reorder["priority"] == sel_pri]
            else:
                display_reorder = f_reorder
        else:
            display_reorder = f_reorder
            
        fmt_reorder = display_reorder.copy()
        if "estimated_cost" in fmt_reorder.columns:
            fmt_reorder["estimated_cost"] = fmt_reorder["estimated_cost"].apply(format_currency)
        st.dataframe(fmt_reorder, use_container_width=True)
    else:
        st.info("No reorder recommendations available.")
        
    st.markdown("---")
    
    st.subheader("Markdown Candidates")
    if f_markdown is not None and not f_markdown.empty:
        total_recov = f_markdown.get("potential_recovery", pd.Series([0])).sum()
        st.metric("Total Potential Recovery", format_currency(total_recov))
        
        fmt_markdown = f_markdown.copy()
        if "potential_recovery" in fmt_markdown.columns:
            fmt_markdown["potential_recovery"] = fmt_markdown["potential_recovery"].apply(format_currency)
        if "suggested_discount_pct" in fmt_markdown.columns:
            fmt_markdown["suggested_discount_pct"] = fmt_markdown["suggested_discount_pct"].apply(format_pct)
        elif "suggested_discount" in fmt_markdown.columns:
            fmt_markdown["suggested_discount"] = fmt_markdown["suggested_discount"].apply(format_pct)
            
        st.dataframe(fmt_markdown, use_container_width=True)
    else:
        st.info("No markdown candidates available.")

    st.markdown("---")
    st.subheader("📋 Unified 4-Tier Inventory Action Recommendations")
    st.markdown("Operational category for every SKU: **Reorder**, **Markdown**, **Watch**, or **Healthy**.")
    f_actions = filter_by_cat(df_actions)
    if f_actions is not None and not f_actions.empty:
        rec_opts = ["ALL"] + sorted(f_actions["recommendation"].dropna().unique().tolist())
        selected_rec = st.selectbox("Filter by Recommendation Category", rec_opts)
        if selected_rec != "ALL":
            display_actions = f_actions[f_actions["recommendation"] == selected_rec]
        else:
            display_actions = f_actions
            
        show_cols = ["sku_id", "product_name", "category", "recommendation", "risk_level", 
                     "days_of_supply", "weeks_of_supply", "stockout_risk_score", "overstock_risk_score", "action_description"]
        avail_cols = [c for c in show_cols if c in display_actions.columns]
        st.dataframe(display_actions[avail_cols], use_container_width=True)
    else:
        st.info("Unified action recommendations data not available.")

elif selection == "Financial Impact":

    st.header("Financial Impact")
    
    if risk_summary:
        rev_risk = risk_summary.get("total_revenue_at_risk", 0)
        exc_cap = risk_summary.get("total_excess_capital", 0)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Revenue at Risk", format_currency(rev_risk))
        c2.metric("Total Excess Capital", format_currency(exc_cap))
        c3.metric("Net Financial Impact", format_currency(rev_risk + exc_cap))
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Revenue at Risk by Category")
            if "by_category" in risk_summary and "revenue_at_risk" in risk_summary["by_category"]:
                cat_data = risk_summary["by_category"]["revenue_at_risk"]
                if cat_data:
                    df_cat_rev = pd.DataFrame(list(cat_data.items()), columns=["Category", "Value"])
                    # Apply global category filter
                    if selected_categories:
                        df_cat_rev = df_cat_rev[df_cat_rev["Category"].isin(selected_categories)]
                    fig = px.bar(df_cat_rev, x="Value", y="Category", orientation='h', color_discrete_sequence=[COLOR_SCHEME["danger"]])
                    st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Excess Capital by Category")
            if "by_category" in risk_summary and "excess_capital" in risk_summary["by_category"]:
                cat_data2 = risk_summary["by_category"]["excess_capital"]
                if cat_data2:
                    df_cat_exc = pd.DataFrame(list(cat_data2.items()), columns=["Category", "Value"])
                    if selected_categories:
                        df_cat_exc = df_cat_exc[df_cat_exc["Category"].isin(selected_categories)]
                    fig2 = px.bar(df_cat_exc, x="Value", y="Category", orientation='h', color_discrete_sequence=[COLOR_SCHEME["warning"]])
                    st.plotly_chart(fig2, use_container_width=True)
                    
        st.subheader("Top SKUs by Financial Impact")
        
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Top 10 Revenue at Risk**")
            if "top_10_skus" in risk_summary and "revenue_at_risk" in risk_summary["top_10_skus"]:
                top_rev = pd.DataFrame(risk_summary["top_10_skus"]["revenue_at_risk"])
                if not top_rev.empty:
                    if "value" in top_rev.columns:
                        top_rev["value"] = top_rev["value"].apply(format_currency)
                    st.dataframe(top_rev, use_container_width=True)
        with t2:
            st.markdown("**Top 10 Excess Capital**")
            if "top_10_skus" in risk_summary and "excess_capital" in risk_summary["top_10_skus"]:
                top_exc = pd.DataFrame(risk_summary["top_10_skus"]["excess_capital"])
                if not top_exc.empty:
                    if "value" in top_exc.columns:
                        top_exc["value"] = top_exc["value"].apply(format_currency)
                    st.dataframe(top_exc, use_container_width=True)

    else:
        st.error("Risk summary JSON not available.")

st.markdown("---")
st.markdown(f"<div style='text-align: center; color: gray; padding: 10px;'>PROJECT FORESIGHT | NorthBay Living | Data refreshed: {datetime.now().strftime('%Y-%m-%d')}</div>", unsafe_allow_html=True)
