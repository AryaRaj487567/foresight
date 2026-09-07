import pytest
import pandas as pd
import numpy as np
from src.risk import (
    calculate_days_of_supply,
    classify_stockout_risk,
    identify_overstock,
    generate_reorder_recommendations,
    identify_markdown_candidates
)

def test_calculate_days_of_supply():
    inv = pd.DataFrame([{
        'sku_id': 'SKU-001',
        'on_hand_units': 70.0,
        'on_order_units': 0.0,
        'lead_time_days': 14,
        'reorder_point': 30
    }])
    demand = pd.DataFrame([{
        'sku_id': 'SKU-001',
        'avg_weekly_demand': 14.0,  # 2 units per day
        'total_8week_demand': 112.0
    }])
    dos_df = calculate_days_of_supply(inv, demand)
    assert dos_df['days_of_supply'].iloc[0] == 35.0  # 70 / 2 = 35 days

def test_classify_stockout_risk():
    dos_df = pd.DataFrame([
        {'sku_id': 'SKU-1', 'days_of_supply': 5.0, 'lead_time_days': 14.0},   # < lead time -> CRITICAL
        {'sku_id': 'SKU-2', 'days_of_supply': 18.0, 'lead_time_days': 14.0},  # < 1.5x lead time -> HIGH
        {'sku_id': 'SKU-3', 'days_of_supply': 30.0, 'lead_time_days': 14.0},  # < 2.5x lead time -> MEDIUM
        {'sku_id': 'SKU-4', 'days_of_supply': 60.0, 'lead_time_days': 14.0},  # >= 2.5x lead time -> LOW
    ])
    risk_df = classify_stockout_risk(dos_df)
    assert risk_df.loc[risk_df['sku_id'] == 'SKU-1', 'risk_level'].iloc[0] == 'CRITICAL'
    assert risk_df.loc[risk_df['sku_id'] == 'SKU-2', 'risk_level'].iloc[0] == 'HIGH'
    assert risk_df.loc[risk_df['sku_id'] == 'SKU-3', 'risk_level'].iloc[0] == 'MEDIUM'
    assert risk_df.loc[risk_df['sku_id'] == 'SKU-4', 'risk_level'].iloc[0] == 'LOW'

def test_identify_overstock():
    dos_df = pd.DataFrame([
        {'sku_id': 'SKU-1', 'weeks_of_supply': 15.0, 'on_hand_units': 150.0, 'avg_weekly_demand': 10.0, 'unit_cost': 20.0},
        {'sku_id': 'SKU-2', 'weeks_of_supply': 5.0, 'on_hand_units': 50.0, 'avg_weekly_demand': 10.0, 'unit_cost': 20.0}
    ])
    overstock = identify_overstock(dos_df, weeks_threshold=12)
    assert len(overstock) == 1
    assert overstock.iloc[0]['sku_id'] == 'SKU-1'
    # 150 - (10 * 12) = 30 units excess
    assert overstock.iloc[0]['excess_units'] == 30.0
    assert overstock.iloc[0]['excess_value'] == 600.0

def test_generate_reorder_recommendations():
    risk_df = pd.DataFrame([
        {
            'sku_id': 'SKU-1',
            'risk_level': 'CRITICAL',
            'lead_time_days': 14.0,
            'avg_weekly_demand': 20.0,
            'total_available': 10.0,
            'on_hand_units': 10.0,
            'unit_cost': 25.0
        }
    ])
    sku_master = pd.DataFrame([
        {'sku_id': 'SKU-1', 'product_name': 'Desk', 'category': 'Furniture'}
    ])
    reorders = generate_reorder_recommendations(risk_df, sku_master)
    assert len(reorders) == 1
    assert reorders.iloc[0]['priority'] == 'URGENT'
    assert reorders.iloc[0]['recommended_order_qty'] > 0
    assert reorders.iloc[0]['estimated_cost'] > 0

def test_risk_scores():
    dos_df = pd.DataFrame([
        {'sku_id': 'SKU-1', 'days_of_supply': 0.0, 'lead_time_days': 14.0},   # 0 days of supply -> 100 stockout score
        {'sku_id': 'SKU-2', 'days_of_supply': 70.0, 'lead_time_days': 14.0},  # 70 days (10 wks) -> 0 stockout score
    ])
    scored = classify_stockout_risk(dos_df)
    assert 'stockout_risk_score' in scored.columns
    assert 'overstock_risk_score' in scored.columns
    assert scored.loc[scored['sku_id'] == 'SKU-1', 'stockout_risk_score'].iloc[0] == 100.0
    assert scored.loc[scored['sku_id'] == 'SKU-2', 'stockout_risk_score'].iloc[0] == 0.0

def test_generate_action_recommendations():
    from src.risk import generate_action_recommendations
    risk_df = pd.DataFrame([
        {'sku_id': 'SKU-1', 'risk_level': 'CRITICAL', 'weeks_of_supply': 1.0},
        {'sku_id': 'SKU-2', 'risk_level': 'LOW', 'weeks_of_supply': 25.0},
        {'sku_id': 'SKU-3', 'risk_level': 'MEDIUM', 'weeks_of_supply': 4.0},
        {'sku_id': 'SKU-4', 'risk_level': 'LOW', 'weeks_of_supply': 6.0}
    ])
    overstock_df = pd.DataFrame([{'sku_id': 'SKU-2'}])
    reorder_df = pd.DataFrame([{'sku_id': 'SKU-1'}])
    markdown_df = pd.DataFrame([{'sku_id': 'SKU-2'}])
    sku_master = pd.DataFrame([
        {'sku_id': f'SKU-{i}', 'product_name': f'P{i}', 'category': 'Cat'} for i in range(1, 5)
    ])
    
    recs = generate_action_recommendations(risk_df, overstock_df, reorder_df, markdown_df, sku_master)
    assert len(recs) == 4
    assert set(recs['recommendation']) == {'Reorder', 'Markdown', 'Watch', 'Healthy'}
    assert recs.loc[recs['sku_id'] == 'SKU-1', 'recommendation'].iloc[0] == 'Reorder'
    assert recs.loc[recs['sku_id'] == 'SKU-2', 'recommendation'].iloc[0] == 'Markdown'
    assert recs.loc[recs['sku_id'] == 'SKU-3', 'recommendation'].iloc[0] == 'Watch'
    assert recs.loc[recs['sku_id'] == 'SKU-4', 'recommendation'].iloc[0] == 'Healthy'

