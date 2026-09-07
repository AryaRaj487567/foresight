import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_cleaning import (
    validate_schema,
    clean_sku_master,
    clean_sales_daily,
    clean_calendar,
    clean_inventory_snapshots
)

def test_validate_schema_valid():
    datasets = {
        'sales_daily': pd.DataFrame([{'date': '2024-01-01', 'sku_id': 'S1', 'units_sold': 1, 'revenue': 10, 'unit_price': 10, 'promo_flag': 0}]),
        'sku_master': pd.DataFrame([{'sku_id': 'S1', 'product_name': 'Desk', 'category': 'Furniture', 'subcategory': 'Tables', 'launch_date': '2024-01-01', 'unit_cost': 50, 'list_price': 100}]),
        'calendar': pd.DataFrame([{'date': '2024-01-01', 'week': 1, 'month': 1, 'year': 2024, 'day_of_week': 'Monday', 'season': 'Winter', 'is_holiday': 1, 'holiday_name': 'New Year', 'promo_event': ''}]),
        'inventory_snapshots': pd.DataFrame([{'date': '2024-01-01', 'sku_id': 'S1', 'on_hand_units': 100, 'on_order_units': 0, 'lead_time_days': 14, 'reorder_point': 50}])
    }
    # Should find no missing column issues
    issues = validate_schema(datasets)
    assert len(issues) == 0

def test_validate_schema_missing():
    datasets = {
        'sales_daily': pd.DataFrame(columns=['date', 'sku_id']),
    }
    issues = validate_schema(datasets)
    assert len(issues) > 0

def test_clean_sku_master():
    raw_sku = pd.DataFrame([
        {'sku_id': 'SKU-001', 'product_name': 'Desk', 'category': 'furniture', 'subcategory': 'tables', 'launch_date': '2024-01-01', 'unit_cost': 50.0, 'list_price': 100.0},
        {'sku_id': 'SKU-001', 'product_name': 'Desk', 'category': 'furniture', 'subcategory': 'tables', 'launch_date': '2024-01-01', 'unit_cost': 50.0, 'list_price': 100.0}, # duplicate
        {'sku_id': 'SKU-002', 'product_name': 'Lamp', 'category': 'DECOR', 'subcategory': 'lighting', 'launch_date': '2024-02-01', 'unit_cost': 20.0, 'list_price': 45.0}
    ])
    log = []
    cleaned = clean_sku_master(raw_sku, log)
    assert len(cleaned) == 2
    assert cleaned.iloc[0]['category'] == 'Furniture'
    assert cleaned.iloc[0]['subcategory'] == 'Tables'
    assert cleaned.iloc[1]['category'] == 'Decor'
    assert cleaned.iloc[1]['subcategory'] == 'Lighting'

def test_clean_sales_daily():
    raw_sales = pd.DataFrame([
        {'date': '2024-01-01', 'sku_id': 'SKU-001', 'units_sold': -2.0, 'revenue': 20.0, 'unit_price': 10.0, 'promo_flag': 0},
        {'date': '2024-01-02', 'sku_id': 'SKU-001', 'units_sold': np.nan, 'revenue': 0.0, 'unit_price': 10.0, 'promo_flag': 0},
        {'date': '2024-01-03', 'sku_id': 'SKU-001', 'units_sold': 5.0, 'revenue': np.nan, 'unit_price': 10.0, 'promo_flag': 1},
        {'date': 'invalid_date', 'sku_id': 'SKU-001', 'units_sold': 3.0, 'revenue': 30.0, 'unit_price': 10.0, 'promo_flag': 0},
        {'date': '2024-01-05', 'sku_id': 'SKU-999', 'units_sold': 1.0, 'revenue': 10.0, 'unit_price': 10.0, 'promo_flag': 0}, # orphan
    ])
    valid_skus = {'SKU-001'}
    log = []
    cleaned = clean_sales_daily(raw_sales, valid_skus, log)
    
    # Invalid date and orphan should be removed
    assert len(cleaned) == 3
    # Negative units should be clipped to 0
    assert (cleaned['units_sold'] >= 0).all()
    # Missing units should be 0
    assert not cleaned['units_sold'].isnull().any()
    # Missing revenue should be imputed
    assert not cleaned['revenue'].isnull().any()
    assert cleaned.loc[cleaned['date'] == '2024-01-03', 'revenue'].iloc[0] == 50.0

def test_clean_calendar():
    raw_cal = pd.DataFrame([
        {'date': '2024-01-01', 'season': ' winter ', 'holiday_name': np.nan, 'promo_event': np.nan}
    ])
    log = []
    cleaned = clean_calendar(raw_cal, log)
    assert cleaned.iloc[0]['season'] == 'Winter'
    assert cleaned.iloc[0]['holiday_name'] == ''
    assert cleaned.iloc[0]['promo_event'] == ''

def test_clean_inventory_snapshots():
    raw_inv = pd.DataFrame([
        {'date': '2024-01-01', 'sku_id': 'SKU-001', 'on_hand_units': 100.0, 'on_order_units': 0, 'lead_time_days': 14, 'reorder_point': 50},
        {'date': '2024-01-08', 'sku_id': 'SKU-001', 'on_hand_units': np.nan, 'on_order_units': 0, 'lead_time_days': 14, 'reorder_point': 50},
        {'date': '2024-01-01', 'sku_id': 'SKU-999', 'on_hand_units': 10.0, 'on_order_units': 0, 'lead_time_days': 7, 'reorder_point': 10}, # orphan
    ])
    valid_skus = {'SKU-001'}
    log = []
    cleaned = clean_inventory_snapshots(raw_inv, valid_skus, log)
    assert len(cleaned) == 2
    # Missing value forward filled from 100.0
    assert cleaned.iloc[1]['on_hand_units'] == 100.0
