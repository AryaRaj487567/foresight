import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
from src.utils import setup_logging, PROCESSED_DATA_DIR, OUTPUTS_DIR

logger = logging.getLogger(__name__)

def get_latest_inventory(inv_df: pd.DataFrame) -> pd.DataFrame:
    """Get the most recent inventory snapshot per SKU.
    Returns: sku_id, on_hand_units, on_order_units, lead_time_days, reorder_point, snapshot_date"""
    inv_df = inv_df.copy()
    inv_df['date'] = pd.to_datetime(inv_df['date'])
    latest = inv_df.sort_values('date').groupby('sku_id').last().reset_index()
    latest = latest.rename(columns={'date': 'snapshot_date'})
    return latest[['sku_id', 'on_hand_units', 'on_order_units', 'lead_time_days', 'reorder_point', 'snapshot_date']]

def calculate_weekly_demand(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate average weekly forecasted demand per SKU.
    Returns: sku_id, avg_weekly_demand, total_8week_demand"""
    demand = forecast_df.groupby('sku_id').agg(
        avg_weekly_demand=('forecast_units', 'mean'),
        total_8week_demand=('forecast_units', 'sum')
    ).reset_index()
    return demand

def calculate_days_of_supply(inventory: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    """Calculate days of supply = (on_hand_units + on_order_units) / (avg_weekly_demand / 7).
    Returns: sku_id, on_hand_units, on_order_units, total_available, avg_daily_demand, days_of_supply"""
    df = pd.merge(inventory, demand, on='sku_id', how='left')
    
    if df['avg_weekly_demand'].isnull().any():
        logger.warning("Some SKUs have missing forecasts. Treating missing demand as 0.")
        df['avg_weekly_demand'] = df['avg_weekly_demand'].fillna(0)
        df['total_8week_demand'] = df['total_8week_demand'].fillna(0)
        
    df['total_available'] = df['on_hand_units'] + df['on_order_units']
    df['avg_daily_demand'] = df['avg_weekly_demand'] / 7
    
    df['days_of_supply'] = np.where(
        df['avg_daily_demand'] > 0,
        df['total_available'] / df['avg_daily_demand'],
        np.inf
    )
    return df

def classify_stockout_risk(dos_df: pd.DataFrame) -> pd.DataFrame:
    """Classify SKUs by stockout risk level and compute risk scores.
    
    Risk levels:
    - CRITICAL: days_of_supply < lead_time_days (will stockout before reorder arrives)
    - HIGH: days_of_supply < lead_time_days * 1.5 (at risk during lead time)
    - MEDIUM: days_of_supply < lead_time_days * 2.5 (buffer is thin)
    - LOW: days_of_supply >= lead_time_days * 2.5 (healthy stock)
    
    Risk Scores (0-100):
    - stockout_risk_score: 100 is immediate stockout, 0 is fully secure stock
    - overstock_risk_score: 100 is severely bloated inventory, 0 is lean
    """
    df = dos_df.copy()
    df['weeks_of_supply'] = df['days_of_supply'] / 7
    
    conditions = [
        df['days_of_supply'] < df['lead_time_days'],
        df['days_of_supply'] < (df['lead_time_days'] * 1.5),
        df['days_of_supply'] < (df['lead_time_days'] * 2.5),
        df['days_of_supply'] >= (df['lead_time_days'] * 2.5)
    ]
    choices = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    df['risk_level'] = np.select(conditions, choices, default='LOW')
    
    # 0 to 100 stockout risk score
    # Ratio of remaining supply to safe threshold (2.5 * lead_time)
    safe_days = df['lead_time_days'] * 2.5
    stockout_score = ((safe_days - df['days_of_supply']) / safe_days) * 100
    df['stockout_risk_score'] = stockout_score.clip(lower=0, upper=100).round(1)
    
    # 0 to 100 overstock risk score
    # 8 weeks of supply is baseline healthy, 26+ weeks is maximum overstock score 100
    overstock_score = ((df['weeks_of_supply'] - 8) / 18) * 100
    df['overstock_risk_score'] = overstock_score.clip(lower=0, upper=100).round(1)
    
    return df


def identify_overstock(dos_df: pd.DataFrame, weeks_threshold: int = 12) -> pd.DataFrame:
    """Identify overstocked SKUs where weeks_of_supply > threshold.
    Returns: sku_id, on_hand_units, avg_weekly_demand, weeks_of_supply, excess_units, excess_value
    excess_units = on_hand - (avg_weekly_demand * weeks_threshold)
    excess_value = excess_units * unit_cost"""
    df = dos_df[dos_df['weeks_of_supply'] > weeks_threshold].copy()
    df['excess_units'] = df['on_hand_units'] - (df['avg_weekly_demand'] * weeks_threshold)
    df['excess_units'] = df['excess_units'].clip(lower=0)
    
    if 'unit_cost' in df.columns:
        df['excess_value'] = (df['excess_units'] * df['unit_cost']).round(2)
    else:
        df['excess_value'] = 0.0
        
    return df[['sku_id', 'on_hand_units', 'avg_weekly_demand', 'weeks_of_supply', 'excess_units', 'excess_value']]

def generate_reorder_recommendations(risk_df: pd.DataFrame, sku_master: pd.DataFrame) -> pd.DataFrame:
    """Generate reorder recommendations for CRITICAL and HIGH risk SKUs.
    
    Recommended order qty = (target_weeks_of_supply * avg_weekly_demand) - total_available
    target_weeks_of_supply = (lead_time_days / 7) * 2 + 4 (lead time + safety + 4 weeks buffer)
    
    Returns: sku_id, product_name, category, risk_level, on_hand_units, recommended_order_qty,
             estimated_cost (qty * unit_cost), priority (URGENT for CRITICAL, HIGH for HIGH)"""
    df = risk_df[risk_df['risk_level'].isin(['CRITICAL', 'HIGH'])].copy()
    # Only merge product_name and category; unit_cost is already in risk_df from pipeline
    merge_cols = ['sku_id', 'product_name', 'category']
    if 'unit_cost' not in df.columns:
        merge_cols.append('unit_cost')
    df = df.merge(sku_master[merge_cols], on='sku_id', how='left')
    
    df['target_weeks_of_supply'] = (df['lead_time_days'] / 7) * 2 + 4
    df['recommended_order_qty'] = (df['target_weeks_of_supply'] * df['avg_weekly_demand']) - df['total_available']
    df['recommended_order_qty'] = df['recommended_order_qty'].apply(np.ceil).clip(lower=0)
    
    df['estimated_cost'] = (df['recommended_order_qty'] * df['unit_cost']).round(2)
    df['priority'] = np.where(df['risk_level'] == 'CRITICAL', 'URGENT', 'HIGH')
    
    return df[['sku_id', 'product_name', 'category', 'risk_level', 'on_hand_units', 'recommended_order_qty', 'estimated_cost', 'priority']]

def identify_markdown_candidates(overstock_df: pd.DataFrame, weekly_sales_df: pd.DataFrame,
                                 sku_master: pd.DataFrame, weeks_threshold: int = 20) -> pd.DataFrame:
    """Identify SKUs for markdown/clearance.
    
    Criteria:
    1. weeks_of_supply > weeks_threshold (extremely overstocked), OR
    2. weeks_of_supply > 12 AND declining trend (last 4 weeks < previous 4 weeks average)
    
    Returns: sku_id, product_name, category, weeks_of_supply, excess_units,
             suggested_discount_pct (based on severity: 15-50%), reason,
             potential_recovery (excess_units * list_price * (1 - discount))"""
    df = overstock_df.merge(sku_master[['sku_id', 'product_name', 'category', 'list_price']], on='sku_id', how='left')
    
    if not weekly_sales_df.empty:
        # Handle both column naming conventions
        date_col = 'week_start_date' if 'week_start_date' in weekly_sales_df.columns else 'week_start'
        sales = weekly_sales_df.sort_values(['sku_id', date_col])
        def is_declining(grp):
            if len(grp) < 8: return False
            last_4 = grp['weekly_units'].iloc[-4:].mean()
            prev_4 = grp['weekly_units'].iloc[-8:-4].mean()
            return last_4 < prev_4
            
        trends = sales.groupby('sku_id').apply(is_declining).reset_index(name='declining_trend')
        df = df.merge(trends, on='sku_id', how='left')
    else:
        df['declining_trend'] = False
        
    df['declining_trend'] = df.get('declining_trend', False).fillna(False)
    
    cond1 = df['weeks_of_supply'] > weeks_threshold
    cond2 = (df['weeks_of_supply'] > 12) & df['declining_trend']
    
    candidates = df[cond1 | cond2].copy()
    
    conditions = [
        candidates['weeks_of_supply'] > 26,
        candidates['weeks_of_supply'] > 20,
        candidates['weeks_of_supply'] > 12
    ]
    choices = [0.50, 0.30, 0.15]
    candidates['suggested_discount_pct'] = np.select(conditions, choices, default=0.15)
    
    candidates['reason'] = np.where(candidates['weeks_of_supply'] > weeks_threshold, 
                                    'Extremely overstocked', 'Overstocked + Declining trend')
                                    
    candidates['potential_recovery'] = (candidates['excess_units'] * candidates['list_price'] * (1 - candidates['suggested_discount_pct'])).round(2)
    
    return candidates[['sku_id', 'product_name', 'category', 'weeks_of_supply', 'excess_units', 'suggested_discount_pct', 'reason', 'potential_recovery']]

def generate_action_recommendations(risk_df: pd.DataFrame, 
                                     overstock_df: pd.DataFrame,
                                     reorder_df: pd.DataFrame,
                                     markdown_df: pd.DataFrame,
                                     sku_master: pd.DataFrame) -> pd.DataFrame:
    """Generate consolidated 4-tier operational recommendations for all SKUs:
    1. Reorder: Imminent or high stockout risk requiring replenishment purchase orders.
    2. Markdown: Excessive inventory with low or declining velocity requiring promotional clearance.
    3. Watch: Thin stock buffer (medium risk) or mild surplus requiring active monitoring.
    4. Healthy: Well-balanced stock aligned with forecasted run-rate.
    """
    df = risk_df.copy()
    if 'product_name' not in df.columns or 'category' not in df.columns:
        df = df.merge(sku_master[['sku_id', 'product_name', 'category']], on='sku_id', how='left')
        
    reorder_skus = set(reorder_df['sku_id']) if not reorder_df.empty else set()
    markdown_skus = set(markdown_df['sku_id']) if not markdown_df.empty else set()
    
    recommendations = []
    actions = []
    
    for _, row in df.iterrows():
        sku = row['sku_id']
        risk_lvl = row.get('risk_level', 'LOW')
        wos = row.get('weeks_of_supply', 0)
        
        if sku in reorder_skus:
            rec = 'Reorder'
            action = 'Issue purchase order to prevent stockout'
        elif sku in markdown_skus:
            rec = 'Markdown'
            action = 'Initiate promotional clearance/discount to recover working capital'
        elif risk_lvl == 'MEDIUM' or (wos > 10 and wos <= 16):
            rec = 'Watch'
            action = 'Monitor sales velocity and supplier lead-time buffers'
        else:
            rec = 'Healthy'
            action = 'Inventory position is optimal'
            
        recommendations.append(rec)
        actions.append(action)
        
    df['recommendation'] = recommendations
    df['action_description'] = actions
    return df

def calculate_revenue_at_risk(risk_df: pd.DataFrame, forecast_df: pd.DataFrame,
                              sku_master: pd.DataFrame) -> dict:
    """Calculate rupee revenue at risk from potential stockouts."""
    at_risk = risk_df[risk_df['risk_level'].isin(['CRITICAL', 'HIGH'])].copy()
    at_risk = at_risk.merge(sku_master[['sku_id', 'product_name', 'category', 'list_price']], on='sku_id', how='left')
    
    at_risk['stockout_days'] = (at_risk['lead_time_days'] - at_risk['days_of_supply']).clip(lower=0)
    at_risk['revenue_at_risk'] = (at_risk['stockout_days'] * at_risk['avg_daily_demand'] * at_risk['list_price']).round(2)
    
    total_rar = float(at_risk['revenue_at_risk'].sum())
    by_cat = at_risk.groupby('category')['revenue_at_risk'].sum().to_dict()
    by_risk = at_risk.groupby('risk_level')['revenue_at_risk'].sum().to_dict()
    
    top_10 = at_risk.nlargest(10, 'revenue_at_risk')[['sku_id', 'product_name', 'revenue_at_risk']]
    top_10_list = list(top_10.itertuples(index=False, name=None))
    
    return {
        'currency': 'INR (₹)',
        'total_revenue_at_risk': total_rar,
        'total_revenue_at_risk_inr_formatted': f"₹{total_rar:,.2f}",
        'by_category': by_cat,
        'by_risk_level': by_risk,
        'top_10_skus': top_10_list
    }

def calculate_excess_capital(overstock_df: pd.DataFrame, sku_master: pd.DataFrame) -> dict:
    """Calculate rupee working capital locked in excess inventory."""
    df = overstock_df.merge(sku_master[['sku_id', 'product_name', 'category', 'unit_cost']], on='sku_id', how='left')
    df['excess_capital'] = (df['excess_units'] * df['unit_cost']).round(2)
    
    total_cap = float(df['excess_capital'].sum())
    by_cat = df.groupby('category')['excess_capital'].sum().to_dict()
    
    top_10 = df.nlargest(10, 'excess_capital')[['sku_id', 'product_name', 'excess_capital']]
    top_10_list = list(top_10.itertuples(index=False, name=None))
    
    return {
        'currency': 'INR (₹)',
        'total_excess_capital': total_cap,
        'total_excess_capital_inr_formatted': f"₹{total_cap:,.2f}",
        'by_category': by_cat,
        'top_10_skus': top_10_list
    }


def run_risk_pipeline(processed_dir: Path = None, forecast_dir: Path = None,
                      output_dir: Path = None) -> dict:
    """End-to-end risk assessment pipeline.
    
    1. Load inventory, forecasts, SKU master, weekly sales
    2. Get latest inventory per SKU
    3. Calculate weekly demand from forecasts
    4. Calculate days of supply
    5. Classify stockout risk
    6. Identify overstock
    7. Generate reorder recommendations
    8. Identify markdown candidates
    9. Calculate revenue at risk
    10. Calculate excess capital
    11. Save all outputs to output_dir/risk/
    12. Return comprehensive summary
    """
    logger.info("Starting risk assessment pipeline...")
    
    if processed_dir is None: processed_dir = PROCESSED_DATA_DIR
    if forecast_dir is None: forecast_dir = OUTPUTS_DIR / 'forecasts'
    if output_dir is None: output_dir = OUTPUTS_DIR / 'risk'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading input data...")
    inv_df = pd.read_csv(processed_dir / 'inventory_snapshots_cleaned.csv')
    forecast_df = pd.read_csv(forecast_dir / 'forecast_8week.csv')
    sku_master = pd.read_csv(processed_dir / 'sku_master_cleaned.csv')
    try:
        weekly_sales_df = pd.read_csv(forecast_dir / 'weekly_sales.csv')
    except FileNotFoundError:
        logger.warning("weekly_sales.csv not found. Using empty data.")
        weekly_sales_df = pd.DataFrame(columns=['sku_id', 'week_start', 'weekly_units'])
    
    logger.info("Processing latest inventory...")
    latest_inv = get_latest_inventory(inv_df)
    
    logger.info("Calculating weekly demand...")
    demand = calculate_weekly_demand(forecast_df)
    
    logger.info("Calculating days of supply...")
    dos_df = calculate_days_of_supply(latest_inv, demand)
    dos_df = dos_df.merge(sku_master[['sku_id', 'unit_cost']], on='sku_id', how='left')
    
    logger.info("Classifying stockout risk...")
    risk_df = classify_stockout_risk(dos_df)
    
    logger.info("Identifying overstock...")
    overstock_df = identify_overstock(risk_df, weeks_threshold=12)
    
    logger.info("Generating reorder recommendations...")
    reorder_df = generate_reorder_recommendations(risk_df, sku_master)
    
    logger.info("Identifying markdown candidates...")
    markdown_df = identify_markdown_candidates(overstock_df, weekly_sales_df, sku_master)
    
    logger.info("Generating unified action recommendations (Reorder / Markdown / Watch / Healthy)...")
    action_recs_df = generate_action_recommendations(risk_df, overstock_df, reorder_df, markdown_df, sku_master)
    
    # Merge action recommendations into risk_df
    risk_df['recommendation'] = action_recs_df['recommendation']
    risk_df['action_description'] = action_recs_df['action_description']

    logger.info("Calculating revenue at risk...")
    rar_summary = calculate_revenue_at_risk(risk_df, forecast_df, sku_master)
    
    logger.info("Calculating excess capital...")
    cap_summary = calculate_excess_capital(overstock_df, sku_master)
    
    summary = {
        'revenue_at_risk': rar_summary,
        'excess_capital': cap_summary,
        'recommendations_breakdown': action_recs_df['recommendation'].value_counts().to_dict()
    }
    
    logger.info("Saving outputs...")
    risk_df.to_csv(output_dir / 'stockout_risk.csv', index=False)
    action_recs_df.to_csv(output_dir / 'inventory_action_recommendations.csv', index=False)
    overstock_df.to_csv(output_dir / 'overstock_analysis.csv', index=False)
    reorder_df.to_csv(output_dir / 'reorder_recommendations.csv', index=False)
    markdown_df.to_csv(output_dir / 'markdown_candidates.csv', index=False)
    
    with open(output_dir / 'risk_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
        
    logger.info("Risk assessment pipeline completed successfully.")
    return summary

