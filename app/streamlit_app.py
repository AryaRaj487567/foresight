import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

# Set up project root in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

st.set_page_config(page_title="PROJECT FORESIGHT", page_icon="📊", layout="wide")

# Polish metric typography to prevent Rupee value clipping
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 1.40rem !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

ACTION_COLORS = {
    "Reorder": "#d62728",
    "Markdown": "#ff7f0e",
    "Watch": "#ffca28",
    "Healthy": "#2ca02c"
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
            with open(path, 'r', encoding='utf-8') as f:
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
    return f"{val:.1f}%" if abs(val) > 1 else f"{val*100:.1f}%"

# Load data
df_merged = load_dataframe("data/processed/merged_analysis_ready.csv")
df_sales_daily = load_dataframe("data/processed/sales_daily_cleaned.csv")
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
df_inventory = load_dataframe("data/processed/inventory_snapshots_cleaned.csv")
risk_summary = load_json("outputs/risk/risk_summary.json")

# Precompute Seasonal-Naive baseline trajectory for all SKUs (period=52 weeks, MA4 fallback)
@st.cache_data
def compute_seasonal_naive_trajectories(weekly_df, horizon=8, seasonal_period=52):
    if weekly_df is None or weekly_df.empty:
        return pd.DataFrame()
    forecasts = []
    for sku_id, group in weekly_df.groupby('sku_id'):
        grp = group.sort_values('week_start_date').copy()
        grp['week_start_date'] = pd.to_datetime(grp['week_start_date'])
        history_map = dict(zip(grp['week_start_date'], grp['weekly_units']))
        
        if len(grp) >= 4:
            ma_fallback = grp['weekly_units'].iloc[-4:].mean()
        else:
            ma_fallback = grp['weekly_units'].mean() if not grp.empty else 0.0
            
        last_date = grp['week_start_date'].max()
        for i in range(1, horizon + 1):
            fc_date = last_date + pd.Timedelta(weeks=i)
            target_date = fc_date - pd.Timedelta(weeks=seasonal_period)
            pred_val = history_map.get(target_date, ma_fallback)
            forecasts.append({
                'sku_id': sku_id,
                'week_start': fc_date,
                'seasonal_naive_units': max(0.0, float(pred_val))
            })
    return pd.DataFrame(forecasts)

df_seasonal_naive = compute_seasonal_naive_trajectories(df_weekly)

# Sidebar
st.sidebar.title("🛋️ NorthBay Living")
st.sidebar.markdown("**PROJECT FORESIGHT**")

pages = [
    "Executive Overview",
    "Demand Forecasts",
    "Inventory Risk",
    "Stockout vs Overstock Grid",
    "Action Center",
    "SKU 360° Details",
    "Financial Impact"
]
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

# Filter helper
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

# Helper to enrich actions with SKU-level financial metrics
def get_enriched_actions():
    if df_actions is None: return None
    df = df_actions.copy()
    
    # Merge list_price from sku_master
    if df_sku is not None and 'list_price' in df_sku.columns:
        df = df.merge(df_sku[['sku_id', 'list_price']], on='sku_id', how='left')
    else:
        df['list_price'] = 0.0
        
    # Merge overstock excess values
    if df_overstock is not None and not df_overstock.empty:
        df = df.merge(df_overstock[['sku_id', 'excess_units', 'excess_value']], on='sku_id', how='left')
        df['excess_units'] = df['excess_units'].fillna(0.0)
        df['excess_inventory_value'] = df['excess_value'].fillna(0.0).round(2)
        df.drop(columns=['excess_value'], inplace=True, errors='ignore')
    else:
        df['excess_units'] = 0.0
        df['excess_inventory_value'] = 0.0
        
    # Calculate sales at risk for CRITICAL / HIGH
    df['stockout_days'] = (df['lead_time_days'] - df['days_of_supply']).clip(lower=0)
    df['sales_at_risk'] = (df['stockout_days'] * df['avg_daily_demand'] * df['list_price']).round(2)
    df.loc[~df['risk_level'].isin(['CRITICAL', 'HIGH']), 'sales_at_risk'] = 0.0
    
    df['total_financial_exposure'] = (df['sales_at_risk'] + df['excess_inventory_value']).round(2)
    return df

# Main app pages
if selection == "Executive Overview":
    st.header("Executive Overview")
    
    # 1. Compute catalog KPIs dynamically
    total_skus = 0
    total_revenue = 0.0
    total_revenue_30d = 0.0
    total_units_sold = 0
    skus_at_risk = 0
    excess_cap = 0.0
    rev_at_risk = 0.0
    
    f_sku = filter_by_cat(df_sku)
    if f_sku is not None:
        total_skus = len(f_sku)
        
    # Total revenue and units sold from sales data
    f_merged = filter_by_cat(df_merged)
    if f_merged is not None and "revenue" in f_merged.columns and "units_sold" in f_merged.columns:
        total_revenue = float(f_merged["revenue"].sum())
        total_units_sold = int(f_merged["units_sold"].sum())
        if "date" in f_merged.columns:
            try:
                date_series = pd.to_datetime(f_merged["date"])
                max_date = date_series.max()
                last_30d = f_merged[date_series >= max_date - pd.Timedelta(days=30)]
                total_revenue_30d = float(last_30d["revenue"].sum())
            except Exception:
                pass
    elif df_sales_daily is not None:
        f_sales = filter_by_cat(df_sales_daily)
        if f_sales is not None:
            total_revenue = float(f_sales["revenue"].sum())
            total_units_sold = int(f_sales["units_sold"].sum())
            
    f_stockout = filter_by_cat(df_stockout)
    if f_stockout is not None and "risk_level" in f_stockout.columns:
        skus_at_risk = len(f_stockout[f_stockout["risk_level"].isin(["CRITICAL", "HIGH"])])
        
    f_overstock = filter_by_cat(df_overstock)
    if f_overstock is not None and "excess_value" in f_overstock.columns:
        excess_cap = float(f_overstock["excess_value"].sum())
        
    if risk_summary and "revenue_at_risk" in risk_summary and "total_revenue_at_risk" in risk_summary["revenue_at_risk"]:
        rev_at_risk = float(risk_summary["revenue_at_risk"]["total_revenue_at_risk"])
    elif risk_summary and "total_revenue_at_risk" in risk_summary:
        rev_at_risk = float(risk_summary["total_revenue_at_risk"])
        
    # Row 1: Core Financial & Volume KPIs
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Active SKUs", f"{total_skus:,}")
    kpi2.metric("Total Revenue (All Time)", format_currency(total_revenue))
    kpi3.metric("Total Units Sold", f"{total_units_sold:,}")
    kpi4.metric("Trailing 30D Revenue", format_currency(total_revenue_30d))
    kpi5.metric("SKUs at Stockout Risk", f"{skus_at_risk:,}")
    
    # Row 2: Working Capital & Financial Exposure KPIs
    st.markdown("##### 💼 Working Capital & Risk Exposure")
    rk1, rk2, rk3 = st.columns(3)
    rk1.metric("Sales at Risk (Potential Stockout)", format_currency(rev_at_risk))
    rk2.metric("Excess Capital Locked (Overstock)", format_currency(excess_cap))
    rk3.metric("Total Financial Exposure", format_currency(rev_at_risk + excess_cap))
    
    # Row 3: 4-Tier Operational Status Counts (Derived Dynamically)
    st.markdown("##### 🎯 4-Tier Operational Action Status")
    enriched_actions = get_enriched_actions()
    f_actions = filter_by_cat(enriched_actions)
    rec_counts = {"Reorder": 0, "Markdown": 0, "Watch": 0, "Healthy": 0}
    if f_actions is not None and "recommendation" in f_actions.columns:
        actual_counts = f_actions["recommendation"].value_counts().to_dict()
        for k in rec_counts:
            rec_counts[k] = actual_counts.get(k, 0)
    elif risk_summary and "recommendations_breakdown" in risk_summary:
        rec_counts = risk_summary["recommendations_breakdown"]
        
    c_rec1, c_rec2, c_rec3, c_rec4 = st.columns(4)
    c_rec1.metric("🔴 Reorder (Urgent / High)", f"{rec_counts.get('Reorder', 0)} SKUs")
    c_rec2.metric("🟠 Markdown (Clearance)", f"{rec_counts.get('Markdown', 0)} SKUs")
    c_rec3.metric("🟡 Watch (Monitor Buffer)", f"{rec_counts.get('Watch', 0)} SKUs")
    c_rec4.metric("🟢 Healthy (Optimal)", f"{rec_counts.get('Healthy', 0)} SKUs")
    
    # Section D: Executive Key Actions
    st.markdown("##### ⚡ Key Actions & Executive Priorities")
    f_reorder = filter_by_cat(df_reorder)
    f_markdown = filter_by_cat(df_markdown)
    reorder_cost = float(f_reorder["estimated_cost"].sum()) if (f_reorder is not None and not f_reorder.empty and "estimated_cost" in f_reorder.columns) else 0.0
    markdown_recov = float(f_markdown["potential_recovery"].sum()) if (f_markdown is not None and not f_markdown.empty and "potential_recovery" in f_markdown.columns) else 0.0
    
    # Identify positive exposure SKU subsets
    pos_sar = f_actions[f_actions["sales_at_risk"] > 0] if f_actions is not None else pd.DataFrame()
    pos_exc = f_actions[f_actions["excess_inventory_value"] > 0] if f_actions is not None else pd.DataFrame()
    
    # Prioritize top financial driver
    if rev_at_risk >= excess_cap:
        primary_driver = f"**Sales-at-Risk Exposure ({format_currency(rev_at_risk)})** represents NorthBay Living's immediate revenue vulnerability across **{len(pos_sar)} high-risk SKU(s)** facing lead-time stockouts."
        secondary_driver = f"**Excess Capital Locked ({format_currency(excess_cap)})** in **{len(pos_exc)} overstocked SKU(s)** creates avoidable carrying cost and working-capital drag."
    else:
        primary_driver = f"**Excess Capital Locked ({format_currency(excess_cap)})** represents the largest working-capital trap, tied up across **{len(pos_exc)} overstocked SKU(s)**."
        secondary_driver = f"**Sales-at-Risk Exposure ({format_currency(rev_at_risk)})** threatens imminent stockout disruption across **{len(pos_sar)} high-risk SKU(s)**."

    st.info(
        f"**Executive Action Summary:**\n\n"
        f"- 🔴 **Replenishment Orders Required:** **{rec_counts.get('Reorder', 0)} SKUs** need purchase order execution (Total capital required: **{format_currency(reorder_cost)}**).\n"
        f"- 🟠 **Markdown Clearance Candidates:** **{rec_counts.get('Markdown', 0)} SKUs** recommended for structured discount clearance (Potential liquidity recovery: **{format_currency(markdown_recov)}**).\n"
        f"- ⚠️ {primary_driver}\n"
        f"- 📦 {secondary_driver}"
    )

    st.markdown("---")
    
    # Section E: Visual Analytics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Level Distribution")
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
            fig = px.bar(cat_rev, x="category", y="revenue", color="category", text_auto='.2s')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Revenue data not available.")
            
    # Section F: Top Positive Risk Tables (Filtered for positive values only)
    st.markdown("---")
    t1, t2 = st.columns(2)
    with t1:
        st.subheader("Top Positive Sales at Risk SKUs")
        if f_actions is not None and not f_actions.empty:
            top_pos_sar = f_actions[f_actions["sales_at_risk"] > 0].sort_values("sales_at_risk", ascending=False).head(10)
            if not top_pos_sar.empty:
                df_disp = top_pos_sar[["sku_id", "product_name", "sales_at_risk"]].copy()
                df_disp["sales_at_risk"] = df_disp["sales_at_risk"].apply(format_currency)
                df_disp.columns = ["SKU ID", "Product Name", "Sales at Risk (₹)"]
                st.dataframe(df_disp, use_container_width=True)
            else:
                st.success("✅ No SKUs currently have positive sales at risk under selected filters.")
        else:
            st.info("Risk data not available.")
            
    with t2:
        st.subheader("Top Positive Excess Capital SKUs")
        if f_actions is not None and not f_actions.empty:
            top_pos_exc = f_actions[f_actions["excess_inventory_value"] > 0].sort_values("excess_inventory_value", ascending=False).head(10)
            if not top_pos_exc.empty:
                df_disp2 = top_pos_exc[["sku_id", "product_name", "excess_inventory_value"]].copy()
                df_disp2["excess_inventory_value"] = df_disp2["excess_inventory_value"].apply(format_currency)
                df_disp2.columns = ["SKU ID", "Product Name", "Excess Capital (₹)"]
                st.dataframe(df_disp2, use_container_width=True)
            else:
                st.success("✅ No SKUs currently hold excess inventory capital under selected filters.")
        else:
            st.info("Excess capital data not available.")

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
            sku_sn = df_seasonal_naive[df_seasonal_naive["sku_id"].astype(str) == selected_sku].copy() if not df_seasonal_naive.empty else pd.DataFrame()
            
            if not sku_weekly.empty and not sku_fcst.empty:
                date_col = "week_start_date" if "week_start_date" in sku_weekly.columns else "week_start"
                sku_weekly["week_start"] = pd.to_datetime(sku_weekly[date_col])
                sku_fcst["week_start"] = pd.to_datetime(sku_fcst["week_start"])
                if not sku_sn.empty:
                    sku_sn["week_start"] = pd.to_datetime(sku_sn["week_start"])
                
                fig = go.Figure()
                # 1. Historical Actual Sales
                fig.add_trace(go.Scatter(
                    x=sku_weekly["week_start"], y=sku_weekly["weekly_units"],
                    mode='lines', name='Historical Sales', line=dict(color=COLOR_SCHEME["primary"], width=2)
                ))
                # 2. Seasonal Naive Baseline Trajectory
                if not sku_sn.empty:
                    fig.add_trace(go.Scatter(
                        x=sku_sn["week_start"], y=sku_sn["seasonal_naive_units"],
                        mode='lines+markers', name='Seasonal Naive Baseline',
                        line=dict(color="#9467bd", dash='dot', width=1.8)
                    ))
                # 3. Selected / Final Forecast
                fig.add_trace(go.Scatter(
                    x=sku_fcst["week_start"], y=sku_fcst["forecast_units"],
                    mode='lines+markers', name='Selected Forecast (MA4)',
                    line=dict(color=COLOR_SCHEME["warning"], dash='dash', width=2.5)
                ))
                # 4. Uncertainty prediction interval (80% confidence)
                fig.add_trace(go.Scatter(
                    x=sku_fcst["week_start"].tolist() + sku_fcst["week_start"].tolist()[::-1],
                    y=sku_fcst["upper_bound"].tolist() + sku_fcst["lower_bound"].tolist()[::-1],
                    fill='toself', fillcolor='rgba(255, 127, 14, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip", showlegend=True, name='80% Prediction Interval'
                ))
                
                fig.update_layout(
                    title=f"Demand Forecast Comparison for {selected_sku_display}",
                    xaxis_title="Week Starting Date",
                    yaxis_title="Demand (Units)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data to plot chart for this SKU.")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Model Performance & Bias")
                if f_metrics is not None:
                    sku_metric = f_metrics[f_metrics["sku_id"].astype(str) == selected_sku]
                    if not sku_metric.empty:
                        m = sku_metric.iloc[0]
                        # Calculate Forecast Bias if not in metrics
                        if 'forecast_bias' in m:
                            bias_val = float(m['forecast_bias'])
                        else:
                            # Derive historical bias from weekly sales trailing 4 weeks vs avg
                            w_units = sku_weekly["weekly_units"].values if not sku_weekly.empty else np.array([])
                            if len(w_units) >= 8:
                                act = w_units[-4:]
                                pred = np.array([w_units[-8:-4].mean()] * 4)
                                bias_val = ((pred.sum() - act.sum()) / (act.sum() + 1e-6)) * 100
                            else:
                                bias_val = 0.0
                                
                        st.metric("WAPE", format_pct(m.get('wape', 0)))
                        st.metric("MAE", f"{m.get('mae', 0):.2f} units")
                        st.metric("RMSE", f"{m.get('rmse', 0):.2f}")
                        st.metric("MAPE", format_pct(m.get('mape', 0)))
                        st.metric("Forecast Bias", f"{bias_val:+.1f}%")
                    else:
                        st.info("Metrics not found.")
            with col2:
                st.subheader("8-Week Forecast Table")
                if not sku_fcst.empty:
                    display_fcst = sku_fcst[["week_start", "forecast_units", "lower_bound", "upper_bound"]].copy()
                    display_fcst["week_start"] = display_fcst["week_start"].dt.strftime("%Y-%m-%d")
                    display_fcst.columns = ["Week Start", "Forecast Units", "Lower Bound (80%)", "Upper Bound (80%)"]
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

# ==============================================================================
# 4. STOCKOUT vs OVERSTOCK DECISION GRID
# ==============================================================================
elif selection == "Stockout vs Overstock Grid":
    st.header("Stockout vs Overstock Decision Grid")
    st.markdown("Interactive 2D decision quadrant mapping **Overstock Risk (0–100)** vs. **Stockout Risk (0–100)**. Point size indicates total financial exposure in Indian Rupees (₹).")
    
    enriched_actions = get_enriched_actions()
    f_grid = filter_by_cat(enriched_actions)
    
    if f_grid is not None and not f_grid.empty:
        # Scale bubble size safely for Plotly
        f_grid = f_grid.copy()
        f_grid['marker_size'] = np.sqrt(f_grid['total_financial_exposure']) + 8.0
        f_grid['financial_exposure_fmt'] = f_grid['total_financial_exposure'].apply(format_currency)
        f_grid['sales_at_risk_fmt'] = f_grid['sales_at_risk'].apply(format_currency)
        f_grid['excess_value_fmt'] = f_grid['excess_inventory_value'].apply(format_currency)
        
        fig_grid = px.scatter(
            f_grid,
            x='overstock_risk_score',
            y='stockout_risk_score',
            color='recommendation',
            size='marker_size',
            color_discrete_map=ACTION_COLORS,
            hover_name='product_name',
            hover_data={
                'sku_id': True,
                'category': True,
                'recommendation': True,
                'stockout_risk_score': ':.1f',
                'overstock_risk_score': ':.1f',
                'financial_exposure_fmt': True,
                'sales_at_risk_fmt': True,
                'excess_value_fmt': True,
                'marker_size': False,
            },
            labels={
                'overstock_risk_score': 'Overstock Risk Score (0 = Lean, 100 = Severe Bloat)',
                'stockout_risk_score': 'Stockout Risk Score (0 = Safe Buffer, 100 = Immediate Stockout)',
                'recommendation': 'Recommended Action',
                'financial_exposure_fmt': 'Total Financial Exposure',
                'sales_at_risk_fmt': 'Sales at Risk',
                'excess_value_fmt': 'Excess Capital'
            }
        )
        
        # Quadrant threshold lines at 30
        fig_grid.add_vline(x=30, line_dash="dash", line_color="gray", opacity=0.5)
        fig_grid.add_hline(y=30, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Quadrant annotations
        fig_grid.add_annotation(x=10, y=95, text="High Stockout / Low Overstock (Critical Reorder)", showarrow=False, font=dict(color="#d62728", size=11))
        fig_grid.add_annotation(x=85, y=95, text="High Stockout / High Overstock (Volatile Mismatch)", showarrow=False, font=dict(color="gray", size=11))
        fig_grid.add_annotation(x=85, y=5, text="Low Stockout / High Overstock (Markdown Candidates)", showarrow=False, font=dict(color="#ff7f0e", size=11))
        fig_grid.add_annotation(x=10, y=5, text="Low Stockout / Low Overstock (Healthy / Watch)", showarrow=False, font=dict(color="#2ca02c", size=11))
        
        fig_grid.update_layout(
            xaxis=dict(range=[-5, 105]),
            yaxis=dict(range=[-5, 105]),
            height=650
        )
        st.plotly_chart(fig_grid, use_container_width=True)
        
        # Summary table of selected items in the grid
        st.subheader("Quadrant SKU Details")
        disp_cols = ['sku_id', 'product_name', 'category', 'recommendation', 'stockout_risk_score', 'overstock_risk_score', 'sales_at_risk', 'excess_inventory_value', 'total_financial_exposure']
        f_table = f_grid[disp_cols].copy().sort_values('total_financial_exposure', ascending=False)
        f_table['sales_at_risk'] = f_table['sales_at_risk'].apply(format_currency)
        f_table['excess_inventory_value'] = f_table['excess_inventory_value'].apply(format_currency)
        f_table['total_financial_exposure'] = f_table['total_financial_exposure'].apply(format_currency)
        f_table.columns = ['SKU ID', 'Product Name', 'Category', 'Action Tier', 'Stockout Score', 'Overstock Score', 'Sales at Risk (₹)', 'Excess Capital (₹)', 'Total Exposure (₹)']
        st.dataframe(f_table, use_container_width=True)
    else:
        st.info("Grid data not available for selected categories.")

# ==============================================================================
# 5. ACTION CENTER
# ==============================================================================
elif selection == "Action Center":
    st.header("Action Center & Replenishment Guidance")
    
    f_reorder = filter_by_cat(df_reorder)
    f_markdown = filter_by_cat(df_markdown)
    
    st.subheader("Reorder Replenishment Orders")
    if f_reorder is not None and not f_reorder.empty:
        total_cost = f_reorder.get("estimated_cost", pd.Series([0])).sum()
        st.metric("Total Estimated Reorder Cost", format_currency(total_cost))
        
        if "priority" in f_reorder.columns:
            priorities = ["ALL"] + f_reorder["priority"].dropna().unique().tolist()
            sel_pri = st.selectbox("Filter Reorders by Priority", priorities)
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
    
    st.subheader("Markdown Clearance Candidates")
    if f_markdown is not None and not f_markdown.empty:
        total_recov = f_markdown.get("potential_recovery", pd.Series([0])).sum()
        st.metric("Total Potential Capital Recovery", format_currency(total_recov))
        
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
    st.subheader("📋 Consolidated 4-Tier Operational Recommendations")
    st.markdown("Every catalog SKU classified into: **Reorder**, **Markdown**, **Watch**, or **Healthy** with calculated financial exposure.")
    
    enriched_actions = get_enriched_actions()
    f_actions = filter_by_cat(enriched_actions)
    
    if f_actions is not None and not f_actions.empty:
        rec_opts = ["ALL"] + sorted(f_actions["recommendation"].dropna().unique().tolist())
        selected_rec = st.selectbox("Filter Recommendations by Tier", rec_opts)
        if selected_rec != "ALL":
            display_actions = f_actions[f_actions["recommendation"] == selected_rec]
        else:
            display_actions = f_actions
            
        # Sort by total financial exposure descending for maximum executive clarity
        if "total_financial_exposure" in display_actions.columns:
            display_actions = display_actions.sort_values("total_financial_exposure", ascending=False)
            
        show_cols = [
            "sku_id", "product_name", "category", "recommendation", "risk_level", 
            "days_of_supply", "weeks_of_supply", "stockout_risk_score", "overstock_risk_score",
            "sales_at_risk", "excess_inventory_value", "total_financial_exposure", "action_description"
        ]
        avail_cols = [c for c in show_cols if c in display_actions.columns]
        fmt_actions = display_actions[avail_cols].copy()
        fmt_actions["sales_at_risk"] = fmt_actions["sales_at_risk"].apply(format_currency)
        fmt_actions["excess_inventory_value"] = fmt_actions["excess_inventory_value"].apply(format_currency)
        fmt_actions["total_financial_exposure"] = fmt_actions["total_financial_exposure"].apply(format_currency)
        fmt_actions.columns = [
            "SKU ID", "Product Name", "Category", "Recommendation", "Risk Level",
            "Days of Supply", "Weeks of Supply", "Stockout Score", "Overstock Score",
            "Sales at Risk (₹)", "Excess Value (₹)", "Total Exposure (₹)", "Action Description"
        ]
        st.dataframe(fmt_actions, use_container_width=True)
    else:
        st.info("Unified action recommendations data not available.")

# ==============================================================================
# 6. SKU 360° DETAILS
# ==============================================================================
elif selection == "SKU 360° Details":
    st.header("SKU 360° Intelligence View")
    st.markdown("Deep dive into SKU-level catalog metadata, inventory position, demand forecast trajectory, risk scores, and financial exposure.")
    
    if df_sku is not None and not df_sku.empty:
        f_sku = filter_by_cat(df_sku)
        if f_sku is not None and not f_sku.empty:
            sku_options = (f_sku["sku_id"].astype(str) + " - " + f_sku["product_name"].fillna("Unknown")).tolist()
            selected_sku_display = st.selectbox("Select SKU to Inspect", sku_options)
            sel_sku_id = selected_sku_display.split(" - ")[0]
            
            sku_meta = f_sku[f_sku["sku_id"] == sel_sku_id].iloc[0]
            
            # 1. Catalog Identity
            st.subheader("1. Catalog Identity & Specifications")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("SKU ID", str(sku_meta.get("sku_id", "")))
            m2.metric("Product Name", str(sku_meta.get("product_name", "")))
            m3.metric("Category", str(sku_meta.get("category", "")))
            m4.metric("Launch Date", str(sku_meta.get("launch_date", "N/A")))
            m5.metric("List Price", format_currency(sku_meta.get("list_price", 0)))
            m6.metric("Unit Cost", format_currency(sku_meta.get("unit_cost", 0)))
            
            st.markdown("---")
            
            # 2. Inventory Position
            enriched = get_enriched_actions()
            sku_action = enriched[enriched["sku_id"] == sel_sku_id].iloc[0] if (enriched is not None and not enriched[enriched["sku_id"] == sel_sku_id].empty) else None
            
            st.subheader("2. Current Inventory Position")
            i1, i2, i3, i4, i5, i6 = st.columns(6)
            if sku_action is not None:
                i1.metric("On Hand Units", f"{int(sku_action.get('on_hand_units', 0)):,}")
                i2.metric("On Order Units", f"{int(sku_action.get('on_order_units', 0)):,}")
                i3.metric("Lead Time", f"{int(sku_action.get('lead_time_days', 0))} days")
                i4.metric("Reorder Point", f"{int(sku_action.get('reorder_point', 0)):,} units")
                i5.metric("Days of Supply", f"{sku_action.get('days_of_supply', 0):.1f} days")
                i6.metric("Weeks of Supply", f"{sku_action.get('weeks_of_supply', 0):.1f} wks")
            
            st.markdown("---")

            # 3. Demand & Forecast Trajectory
            st.subheader("3. Demand History & 8-Week Forecast Trajectory")
            sku_weekly = df_weekly[df_weekly["sku_id"] == sel_sku_id].copy() if df_weekly is not None else pd.DataFrame()
            sku_fcst = df_forecast[df_forecast["sku_id"] == sel_sku_id].copy() if df_forecast is not None else pd.DataFrame()
            sku_sn = df_seasonal_naive[df_seasonal_naive["sku_id"] == sel_sku_id].copy() if not df_seasonal_naive.empty else pd.DataFrame()
            
            if not sku_weekly.empty and not sku_fcst.empty:
                date_col = "week_start_date" if "week_start_date" in sku_weekly.columns else "week_start"
                sku_weekly["week_start"] = pd.to_datetime(sku_weekly[date_col])
                sku_fcst["week_start"] = pd.to_datetime(sku_fcst["week_start"])
                if not sku_sn.empty:
                    sku_sn["week_start"] = pd.to_datetime(sku_sn["week_start"])
                    
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sku_weekly["week_start"], y=sku_weekly["weekly_units"],
                    mode='lines', name='Historical Sales', line=dict(color=COLOR_SCHEME["primary"], width=2)
                ))
                if not sku_sn.empty:
                    fig.add_trace(go.Scatter(
                        x=sku_sn["week_start"], y=sku_sn["seasonal_naive_units"],
                        mode='lines+markers', name='Seasonal Naive Baseline', line=dict(color="#9467bd", dash='dot', width=1.8)
                    ))
                fig.add_trace(go.Scatter(
                    x=sku_fcst["week_start"], y=sku_fcst["forecast_units"],
                    mode='lines+markers', name='Selected Forecast (MA4)', line=dict(color=COLOR_SCHEME["warning"], dash='dash', width=2.5)
                ))
                fig.add_trace(go.Scatter(
                    x=sku_fcst["week_start"].tolist() + sku_fcst["week_start"].tolist()[::-1],
                    y=sku_fcst["upper_bound"].tolist() + sku_fcst["lower_bound"].tolist()[::-1],
                    fill='toself', fillcolor='rgba(255, 127, 14, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip", showlegend=True, name='80% Prediction Interval'
                ))
                fig.update_layout(title=f"8-Week Demand Trajectory for {selected_sku_display}", xaxis_title="Week Starting Date", yaxis_title="Units")
                st.plotly_chart(fig, use_container_width=True)
                
                # 8-week forecast table
                df_table = sku_fcst[["week_start", "forecast_units", "lower_bound", "upper_bound"]].copy()
                df_table["week_start"] = df_table["week_start"].dt.strftime("%Y-%m-%d")
                df_table.columns = ["Forecast Week Start", "Forecast Units", "Lower Bound", "Upper Bound"]
                st.dataframe(df_table.round(2), use_container_width=True)

            st.markdown("---")

            # 4. Risk Intelligence & Action Recommendation
            st.subheader("4. Risk Intelligence & Operational Action")
            r1, r2, r3, r4 = st.columns(4)
            if sku_action is not None:
                r1.metric("Action Tier", str(sku_action.get("recommendation", "")))
                r2.metric("Stockout Risk Level", str(sku_action.get("risk_level", "")))
                r3.metric("Stockout Risk Score", f"{sku_action.get('stockout_risk_score', 0):.1f} / 100")
                r4.metric("Overstock Risk Score", f"{sku_action.get('overstock_risk_score', 0):.1f} / 100")
                st.caption(f"**Operational Rationale:** {sku_action.get('action_description', '')}")

            st.markdown("---")

            # 5. SKU Financial Exposure
            st.subheader("5. SKU Financial Exposure & Capital Impact")
            f1, f2, f3 = st.columns(3)
            if sku_action is not None:
                f1.metric("Sales at Risk (Potential Stockout)", format_currency(sku_action.get("sales_at_risk", 0)))
                f2.metric("Excess Capital (Overstock)", format_currency(sku_action.get("excess_inventory_value", 0)))
                f3.metric("Total Financial Exposure", format_currency(sku_action.get("total_financial_exposure", 0)))
        else:
            st.warning("No SKUs available for selected filters.")
    else:
        st.error("SKU master data not found.")

# ==============================================================================
# 7. FINANCIAL IMPACT
# ==============================================================================
elif selection == "Financial Impact":
    st.header("Working Capital & Financial Impact")
    
    if risk_summary:
        rev_risk = 0.0
        exc_cap = 0.0
        
        if "revenue_at_risk" in risk_summary and "total_revenue_at_risk" in risk_summary["revenue_at_risk"]:
            rev_risk = float(risk_summary["revenue_at_risk"]["total_revenue_at_risk"])
        elif "total_revenue_at_risk" in risk_summary:
            rev_risk = float(risk_summary["total_revenue_at_risk"])
            
        if "excess_capital" in risk_summary and "total_excess_capital" in risk_summary["excess_capital"]:
            exc_cap = float(risk_summary["excess_capital"]["total_excess_capital"])
        elif "total_excess_capital" in risk_summary:
            exc_cap = float(risk_summary["total_excess_capital"])
        
        # Executive Financial Impact KPIs (Row 1: 3 macro exposure KPIs; Row 2: 2 operational liquidity/capital KPIs)
        f_reorder = filter_by_cat(df_reorder)
        f_markdown = filter_by_cat(df_markdown)
        reorder_capital = float(f_reorder["estimated_cost"].sum()) if (f_reorder is not None and not f_reorder.empty and "estimated_cost" in f_reorder.columns) else 0.0
        markdown_recovery = float(f_markdown["potential_recovery"].sum()) if (f_markdown is not None and not f_markdown.empty and "potential_recovery" in f_markdown.columns) else 0.0

        st.markdown("##### 💼 Working Capital & Risk Exposure")
        c1, c2, c3 = st.columns(3)
        c1.metric("Sales at Risk (Potential Stockout)", format_currency(rev_risk))
        c2.metric("Excess Capital Locked (Overstock)", format_currency(exc_cap))
        c3.metric("Net Financial Exposure", format_currency(rev_risk + exc_cap))

        st.markdown("##### ⚡ Capital Allocation & Liquidity Opportunities")
        c4, c5 = st.columns(2)
        c4.metric("Reorder Capital Needed (Replenishment)", format_currency(reorder_capital))
        c5.metric("Markdown Liquidity Potential (Clearance)", format_currency(markdown_recovery))
        
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sales at Risk by Category")
            if "revenue_at_risk" in risk_summary and "by_category" in risk_summary["revenue_at_risk"]:
                cat_data = risk_summary["revenue_at_risk"]["by_category"]
                if cat_data:
                    df_cat_rev = pd.DataFrame(list(cat_data.items()), columns=["Category", "Value"])
                    if selected_categories:
                        df_cat_rev = df_cat_rev[df_cat_rev["Category"].isin(selected_categories)]
                    fig = px.bar(df_cat_rev, x="Value", y="Category", orientation='h', color_discrete_sequence=[COLOR_SCHEME["danger"]], text_auto='.2s')
                    st.plotly_chart(fig, use_container_width=True)
            elif "by_category" in risk_summary and "revenue_at_risk" in risk_summary["by_category"]:
                cat_data = risk_summary["by_category"]["revenue_at_risk"]
                if cat_data:
                    df_cat_rev = pd.DataFrame(list(cat_data.items()), columns=["Category", "Value"])
                    if selected_categories:
                        df_cat_rev = df_cat_rev[df_cat_rev["Category"].isin(selected_categories)]
                    fig = px.bar(df_cat_rev, x="Value", y="Category", orientation='h', color_discrete_sequence=[COLOR_SCHEME["danger"]], text_auto='.2s')
                    st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Excess Capital by Category")
            if "excess_capital" in risk_summary and "by_category" in risk_summary["excess_capital"]:
                cat_data2 = risk_summary["excess_capital"]["by_category"]
                if cat_data2:
                    df_cat_exc = pd.DataFrame(list(cat_data2.items()), columns=["Category", "Value"])
                    if selected_categories:
                        df_cat_exc = df_cat_exc[df_cat_exc["Category"].isin(selected_categories)]
                    fig2 = px.bar(df_cat_exc, x="Value", y="Category", orientation='h', color_discrete_sequence=[COLOR_SCHEME["warning"]], text_auto='.2s')
                    st.plotly_chart(fig2, use_container_width=True)
            elif "by_category" in risk_summary and "excess_capital" in risk_summary["by_category"]:
                cat_data2 = risk_summary["by_category"]["excess_capital"]
                if cat_data2:
                    df_cat_exc = pd.DataFrame(list(cat_data2.items()), columns=["Category", "Value"])
                    if selected_categories:
                        df_cat_exc = df_cat_exc[df_cat_exc["Category"].isin(selected_categories)]
                    fig2 = px.bar(df_cat_exc, x="Value", y="Category", orientation='h', color_discrete_sequence=[COLOR_SCHEME["warning"]], text_auto='.2s')
                    st.plotly_chart(fig2, use_container_width=True)
                    
        st.markdown("---")
        st.subheader("Top SKUs by Financial Impact (Positive Exposure)")
        
        enriched_actions = get_enriched_actions()
        f_actions = filter_by_cat(enriched_actions)

        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Top Positive Sales at Risk SKUs**")
            if f_actions is not None and not f_actions.empty:
                top_pos_sar = f_actions[f_actions["sales_at_risk"] > 0].sort_values("sales_at_risk", ascending=False).head(10)
                if not top_pos_sar.empty:
                    df_top_rev = top_pos_sar[["sku_id", "product_name", "sales_at_risk"]].copy()
                    df_top_rev["sales_at_risk"] = df_top_rev["sales_at_risk"].apply(format_currency)
                    df_top_rev.columns = ["SKU ID", "Product Name", "Sales at Risk (₹)"]
                    st.dataframe(df_top_rev, use_container_width=True)
                else:
                    st.success("✅ No SKUs currently have positive sales at risk under selected filters.")
            else:
                st.info("Risk data not available.")
        with t2:
            st.markdown("**Top Positive Excess Capital SKUs**")
            if f_actions is not None and not f_actions.empty:
                top_pos_exc = f_actions[f_actions["excess_inventory_value"] > 0].sort_values("excess_inventory_value", ascending=False).head(10)
                if not top_pos_exc.empty:
                    df_top_exc = top_pos_exc[["sku_id", "product_name", "excess_inventory_value"]].copy()
                    df_top_exc["excess_inventory_value"] = df_top_exc["excess_inventory_value"].apply(format_currency)
                    df_top_exc.columns = ["SKU ID", "Product Name", "Excess Capital (₹)"]
                    st.dataframe(df_top_exc, use_container_width=True)
                else:
                    st.success("✅ No SKUs currently hold excess inventory capital under selected filters.")
            else:
                st.info("Excess capital data not available.")
    else:
        st.error("Risk summary JSON not available.")

st.markdown("---")
st.markdown(f"<div style='text-align: center; color: gray; padding: 10px;'>PROJECT FORESIGHT | NorthBay Living | Operational Decision Dashboard | Data refreshed: {datetime.now().strftime('%Y-%m-%d')}</div>", unsafe_allow_html=True)
